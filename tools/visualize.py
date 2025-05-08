import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm #! 修改

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

# 递归的读取配置文件
def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()
    #TODO 目前同时显示针对bbox
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE") # 配置文件
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred", "gt_pred"]) # 可视化gt或pred或同时 #! 增加同时显示
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None) # 筛选
    parser.add_argument("--bbox-score", type=float, default=None) # 筛选
    parser.add_argument("--map-score", type=float, default=0.5) # 筛选
    parser.add_argument("--out-dir", type=str, default="viz") # 可视化结果保存路径
    args, opts = parser.parse_known_args()  # 包括一些额外参数opts

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader( #? 命名为dataflow
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred" or args.mode == "gt_pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval() # 改变某些层的模式，比如BN和Dropout 不会禁用梯度

    #TODO 以上是准备工作，以下是数据处理和可视化

    for sample_idx, data in tqdm(enumerate(dataflow)): # 一帧帧处理 
        metas = data["metas"].data[0][0] # 字典
        # name = "{}-{}".format(metas["timestamp"], metas["token"])
        name = f"{sample_idx:06d}" #! 更改输出名称
        if args.mode == "pred" or args.mode == "gt_pred":
            with torch.inference_mode(): # 相对于torch.no_grad()，torch.inference_mode()会完全禁用自动求导机制
                outputs = model(**data)
        #TODO bbox
        bboxes = None
        labels = None
        gt_bboxes = None
        gt_labels = None
        gt_scores = None    
        pred_bboxes = None
        pred_labels = None
        pred_scores = None
        if (args.mode == "gt" or args.mode == "gt_pred") and "gt_bboxes_3d" in data:
            #? 获取tensor属性[num_boxes, box_dim],并转换为numpy数组 box_dim=9 x y z dx dy dz yaw vx vy 在nuscenes_dataset.py中添加了vx,vy
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy() 
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2 # 将中心点从几何中心移动到底部中心
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            gt_bboxes = bboxes
            gt_labels = labels
            gt_scores = np.ones_like(labels).astype(np.float32)

        if (args.mode == "pred" or args.mode == "gt_pred") and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            pred_bboxes = bboxes
            pred_labels = labels
            pred_scores = scores
        if args.mode == "gt_pred":
            bboxes = LiDARInstance3DBoxes.cat([gt_bboxes, pred_bboxes])
            labels = np.concatenate([gt_labels, pred_labels])
            scores = np.concatenate([gt_scores, pred_scores])


        #TODO map
        masks = None
        gt_masks = None
        pred_masks = None
        if (args.mode == "gt" or args.mode == "gt_pred") and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
            gt_masks = masks
        if (args.mode == "pred" or args.mode == "gt_pred") and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
            pred_masks = masks
            
        #TODO 可视化
        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                if args.mode == "pred":
                    visualize_camera(
                        os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                        image,
                        bboxes=pred_bboxes,
                        labels=pred_labels,
                        scores=pred_scores,
                        transform=metas["lidar2image"][k],
                        classes=cfg.object_classes,
                    )
                elif args.mode == "gt":
                    visualize_camera(
                        os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                        image,
                        bboxes=gt_bboxes,
                        labels=gt_labels,
                        scores=gt_scores,
                        transform=metas["lidar2image"][k],
                        classes=cfg.object_classes,
                    )
                else:
                    visualize_camera(
                        os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                        image,
                        bboxes=bboxes,
                        labels=labels,
                        scores=scores,
                        transform=metas["lidar2image"][k],
                        classes=cfg.object_classes,
                    )


        if "points" in data:
            lidar = data["points"].data[0][0].numpy() # (num_obj,num_points,5)
            if args.mode == 'pred':
                visualize_lidar(
                    os.path.join(args.out_dir, "lidar", f"{name}.png"),
                    lidar,
                    bboxes=pred_bboxes,
                    labels=pred_labels,
                    scores=pred_scores,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]], # 0,3 是x_min,x_max
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]], # 1,4 是y_min,y_max
                    classes=cfg.object_classes,
                )
            elif args.mode == 'gt':
                visualize_lidar(
                    os.path.join(args.out_dir, "lidar", f"{name}.png"),
                    lidar,
                    bboxes=gt_bboxes,
                    labels=gt_labels,
                    scores=gt_scores,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]], # 0,3 是x_min,x_max
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]], # 1,4 是y_min,y_max
                    classes=cfg.object_classes,
                )
            else:
                visualize_lidar(
                    os.path.join(args.out_dir, "lidar", f"{name}.png"),
                    lidar,
                    bboxes=bboxes,
                    labels=labels,
                    scores=scores,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]], # 0,3 是x_min,x_max
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]], # 1,4 是y_min,y_max
                    classes=cfg.object_classes,
                )
                

        if masks is not None:
            if args.mode == 'pred':
                visualize_map(
                    os.path.join(args.out_dir, "map", f"{name}.png"),
                    pred_masks,
                    classes=cfg.map_classes,
                )
            elif args.mode == 'gt':
                visualize_map(
                    os.path.join(args.out_dir, "map", f"{name}.png"),
                    gt_masks,
                    classes=cfg.map_classes,
                )
            else:
                visualize_map(
                    os.path.join(args.out_dir, "map", f"{name}.png"),
                    masks,
                    classes=cfg.map_classes,
                )


if __name__ == "__main__":
    main()
