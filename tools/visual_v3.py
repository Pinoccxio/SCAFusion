
#* v1 原版 基于test.py from cx
#* v2 数据来源从dataset改为dataloader，转换矩阵直接由lidar2img,
#*    修改了投影函数，未检测函数，iou计算补充lidar坐标系
#*    添加了将bbox中心从几何中心移动到底部中心的操作
#* v3 更改绘图逻辑，先生成gt和pred的索引，再绘图
#*    添加lidar绘图和lidar的文字绘制，同时添加了with_text参数(store_true)

#TODO from cx
import matplotlib.pyplot as plt
import argparse
import copy
import os
import warnings
import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval
from mmdet3d.core.bbox import LiDARInstance3DBoxes, get_box_type, points_cam2img
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d
from pprint import pprint
import cv2
import numpy as np
from tqdm import tqdm

#! 添加dataloader
def visualize_topK_matches(data_loader, outputs, output_dir="match_vis", topK=4):
    """
    可视化每个GT对应的前k个预测结果
    Args:
        dataset: 数据集对象
        outputs: 模型输出列表 已经过box.decode处理
        output_dir: 输出目录
        topK: 显示前k个预测结果
        pred的文字顺序: k+1:label:score:iou
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = {
        'gt': (0, 255, 0),  # Green For GT
        'pred': [(0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255)]  # Yellow For Pred
    } # Top4 不同颜色

    for idx, data in enumerate(data_loader): #! 修改迭代方式
        print(f"Processing frame {idx}")

        # >>> Data preparation >>>  
        try:
            metas = data['metas'].data[0][0]
            output = outputs[idx]
            #! 修改img_path读取方式
            img_path = metas['filename'][0]
            img = cv2.imread(img_path)
            if img is None:
                print(f'Could not read image from {img_path}, skipping now...')
                continue
        except KeyError as e:
            print(f'Key Value is wrong: {e}, skipping frame{idx}')
            continue
        # <<< Data preparation <<<
   
        # >>> Calibration preparation >>>
        #!  坐标变换
        lidar2img = metas['lidar2image'][0]
        # <<< Calibration preparation <<<

        # >>> Data loading & Validation >>>
        try:
            pred_boxes = output['boxes_3d'].tensor.numpy()
            pred_labels = output['labels_3d']
            pred_scores = output['scores_3d']
            #! 修改gt_boxes 和 gt_labels 读取方式
            gt_boxes = data['gt_bboxes_3d'].data[0][0].tensor.numpy()
            gt_labels = data['gt_labels_3d'].data[0][0]
            #TODO 这就是偏移的原因之一
            #! 将3d框中心点从几何中心移动到底部中心
            pred_boxes[..., 2] -= pred_boxes[..., 5] / 2
            pred_boxes = LiDARInstance3DBoxes(pred_boxes, box_dim=9)
            gt_boxes[..., 2] -= gt_boxes[..., 5] / 2
            gt_boxes = LiDARInstance3DBoxes(gt_boxes, box_dim=9)

            # Empty case
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                print(f"With no valid pred or GT, skipping frame {idx}")
                continue
        except Exception as e:
            print(f"Error loading data：{e}, skipping frame {idx}")
            continue
        # <<< Data loading & Validation <<<

        # >>> IoU Calculation>>>
        try:
            # 3D-IoU_Matrix [num_gt, num_pred]
            iou_matrix = bbox_overlaps_3d(
                gt_boxes.tensor[:, :7],  # 7 parameters (x,y,z,w,l,h,ry)
                pred_boxes.tensor[:, :7],
                coordinate='lidar'
            ) # (num_gt, num_pred)
            assert iou_matrix.shape[0] == len(gt_boxes), "Unmatch for dimensions of IoU Matrix"
        except Exception as e:
            print(f"Error calculating IoU：{e}, skipping frame {idx}")
            continue
        # <<< IoU Calculation <<<

        # >>> Visualization >>>
        vis_img = img.copy()
        for gt_idx in range(iou_matrix.shape[0]):
            # 处理GT框
            gt_box = gt_boxes[gt_idx:gt_idx+1] # (1,7)
            gt_label = gt_labels[gt_idx].item()
            #! gt修改投影
            gt_corners = project_box(gt_box, lidar2img, img.shape[:2])
            if gt_corners:
                draw_boxes(vis_img, gt_corners, colors['gt'], f'{gt_label}')
                print(f'Drawing gt_box on with {colors["gt"]}')

            ious = iou_matrix[gt_idx]
            topk_values, topK_indices = torch.topk(ious, k=min(topK, len(ious)))
            #TODO 在此处做漏检、位置错误、类别错误的可视化操作
            for k in range(len(topK_indices)):
                # 处理pred框
                pred_idx = topK_indices[k].item()
                if ious[pred_idx] <= 0: # 1e-6
                    continue
                try:
                    pred_box = pred_boxes[pred_idx:pred_idx+1]
                    pred_label = pred_labels[pred_idx].item()
                    pred_score = pred_scores[pred_idx].item()
                    #! pred修改投影
                    pred_corners = project_box(pred_box, lidar2img, img.shape[:2])

                    if pred_corners:
                        # Draw
                        text = f'{k + 1}:{pred_label}:{pred_score:.2f}:{ious[pred_idx]:.2f}'
                        draw_boxes(vis_img, pred_corners, colors['pred'][k % 4], text)
                        print(f'Drawing pred_box at frame {idx}')
                    else:
                        print(f"No valid pred_corners, skipping frame {idx}")
                        continue
                except Exception as e:
                    print(f'Prediction Error: {e}, skipping frame {idx}')
        # Saving Results
        save_path = os.path.join(output_dir, f"frame{idx}.jpg")
        cv2.imwrite(save_path, vis_img)
        # <<< Visualization <<<

#! 添加misdetection可视化
def visualize_misdetections(data_loader, outputs, 
                            output_dir="mis_match_vis", 
                            threshold=0.5, with_text=True,
                            xlim=None, ylim=None):
    os.makedirs(output_dir, exist_ok=True)
    colors = {
        'gt': (0, 255, 0),  # Green For GT
        'pred': (0, 255, 255),  # Yellow For Pred
        'mis': (0, 0, 255)  # Red For Mis
    } # Top4 不同颜色

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)): #! 修改迭代方式
        # >>> Data preparation >>>  
        try:
            metas = data['metas'].data[0][0]
            output = outputs[idx]
            lidar = data['points'].data[0][0].numpy()
            #! 修改img_path读取方式
            img_path = metas['filename'][0]
            img = cv2.imread(img_path)
            if img is None:
                print(f'Could not read image from {img_path}, skipping now...')
                continue
        except KeyError as e:
            print(f'Key Value is wrong: {e}, skipping frame{idx}')
            continue
        # <<< Data preparation <<<
   
        # >>> Calibration preparation >>>
        #!  坐标变换
        lidar2img = metas['lidar2image'][0]
        # <<< Calibration preparation <<<

        # >>> Data loading & Validation >>>
        try:
            pred_boxes = output['boxes_3d'].tensor.numpy()
            pred_labels = output['labels_3d']
            pred_scores = output['scores_3d']
            #! 修改gt_boxes 和 gt_labels 读取方式
            gt_boxes = data['gt_bboxes_3d'].data[0][0].tensor.numpy()
            gt_labels = data['gt_labels_3d'].data[0][0]
            #TODO 这就是偏移的原因之一
            #! 将3d框中心点从几何中心移动到底部中心
            pred_boxes[..., 2] -= pred_boxes[..., 5] / 2
            pred_boxes = LiDARInstance3DBoxes(pred_boxes, box_dim=9)
            gt_boxes[..., 2] -= gt_boxes[..., 5] / 2
            gt_boxes = LiDARInstance3DBoxes(gt_boxes, box_dim=9)

            # Empty case
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                print(f"With no valid pred or GT, skipping frame {idx}")
                continue
        except Exception as e:
            print(f"Error loading data: {e}, skipping frame {idx}")
            continue
        # <<< Data loading & Validation <<<

        # >>> IoU Calculation>>>
        try:
            # 3D-IoU_Matrix [num_gt, num_pred]
            iou_matrix = bbox_overlaps_3d(
                gt_boxes.tensor[:, :7],  # 7 parameters (x,y,z,w,l,h,ry)
                pred_boxes.tensor[:, :7],
                coordinate='lidar'
            ) # (num_gt, num_pred)
            assert iou_matrix.shape[0] == len(gt_boxes), "Unmatch for dimensions of IoU Matrix"
        except Exception as e:
            print(f"Error calculating IoU: {e}, skipping frame {idx}")
            continue
        # <<< IoU Calculation <<<  

        # >>> Get Indices >>>
        gt_indices = []
        pred_indices = []
        pred_ious = []
        mis_indices = []
        mis_ious = []
        for gt_idx in range(iou_matrix.shape[0]):
            gt_indices.append(gt_idx)
            gt_label = gt_labels[gt_idx].item()
            ious = iou_matrix[gt_idx]
            topk_values, topK_indices = torch.topk(ious, k=min(1, len(ious)))
            pred_iou, pred_idx = topk_values[0], topK_indices[0]
            pred_label = pred_labels[pred_idx].item()
            if pred_iou == 0:
                continue
            if pred_iou <= threshold or pred_label != gt_label:
                mis_indices.append(pred_idx)
                mis_ious.append(pred_iou)
            else:
                pred_indices.append(pred_idx)
                pred_ious.append(pred_iou)
        # <<< Get Indices <<<

        # >>> Visualization >>>
        # img
        vis_img = img.copy()
        # lidar
        fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1)
        ax.set_axis_off()
        radius = 15
        lidar_width = 25
        lidar_fontsize = 75
        lidar_bias = -1 # 在plt中，图像y轴朝上
        if lidar is not None:
            plt.scatter(
                lidar[:, 0],
                lidar[:, 1],
                s=radius,
                c="white",
            )
        for gt_idx in gt_indices:
            gt_box = gt_boxes[gt_idx:gt_idx+1] # (1,7)
            gt_label = gt_labels[gt_idx].item()
            gt_corners = project_box(gt_box, lidar2img, img.shape[:2])
            if not gt_corners:
                continue
            text = f"{gt_label}" if with_text else ""
            draw_boxes(vis_img, gt_corners, colors['gt'], text)
            # lidar
            coords = gt_box.corners[0,[0,3,7,4,0],:2]
            plt.plot(
                coords[:,0],
                coords[:,1],
                color=np.array(colors['gt'])[::-1] / 255,
                linewidth=lidar_width,
            )
            center = coords[:-1].mean(axis=0)  # 计算前4个点的中心坐标
            plt.text(
                center[0],
                center[1]+lidar_bias,
                text,
                color=np.array(colors['gt'])[::-1] / 255,
                fontsize=lidar_fontsize,
                horizontalalignment='center',
                verticalalignment='center'
            )
        for k, pred_idx in enumerate(pred_indices):
            pred_box = pred_boxes[pred_idx:pred_idx+1]
            pred_label = pred_labels[pred_idx].item()
            pred_score = pred_scores[pred_idx].item()
            pred_iou = pred_ious[k]
            pred_corners = project_box(pred_box, lidar2img, img.shape[:2])
            if not pred_corners:
                continue
            text = f"{pred_label}:{pred_score:.2f}:{pred_iou:.2f}" if with_text else ""
            draw_boxes(vis_img, pred_corners, colors['pred'], text)
            # lidar
            coords = pred_box.corners[0,[0,3,7,4,0],:2]
            plt.plot(
                coords[:,0],
                coords[:,1],
                color=np.array(colors['pred'])[::-1] / 255,
                linewidth=lidar_width,
            )
            center = coords[:-1].mean(axis=0)  # 计算前4个点的中心坐标
            plt.text(
                center[0],
                center[1]+lidar_bias,
                text,
                color=np.array(colors['pred'])[::-1] / 255,
                fontsize=lidar_fontsize,
                horizontalalignment='center',
                verticalalignment='center'
            )
        for k, mis_idx in enumerate(mis_indices):
            mis_box = pred_boxes[mis_idx:mis_idx+1]
            mis_label = pred_labels[mis_idx].item()
            mis_score = pred_scores[mis_idx].item()
            mis_iou = mis_ious[k]
            mis_corners = project_box(mis_box, lidar2img, img.shape[:2])
            if not mis_corners:
                continue
            text = f"{mis_label}:{mis_score:.2f}:{mis_iou:.2f}" if with_text else ""
            draw_boxes(vis_img, mis_corners, colors['mis'], text)
            # lidar
            coords = mis_box.corners[0,[0,3,7,4,0],:2]
            plt.plot(
                coords[:,0],
                coords[:,1],
                color=np.array(colors['mis'])[::-1] / 255,
                linewidth=lidar_width,
            )
            center = coords[:-1].mean(axis=0)  # 计算前4个点的中心坐标
            plt.text(
                center[0],
                center[1]+lidar_bias,
                text,
                color=np.array(colors['mis'])[::-1] / 255,
                fontsize=lidar_fontsize,
                horizontalalignment='center',
                verticalalignment='center'
            )
        # <<< Visualization <<<

        # >>> Saving Results >>>
        # img
        img_path = os.path.join(output_dir,'camera-0')
        os.makedirs(img_path, exist_ok=True)
        save_path = os.path.join(img_path, f"frame{idx}.png")
        cv2.imwrite(save_path, vis_img)
        # lidar
        lidar_path = os.path.join(output_dir,'lidar')
        os.makedirs(lidar_path, exist_ok=True)
        save_path = os.path.join(lidar_path, f"frame{idx}.png")
        fig.savefig(
            save_path,
            dpi=10,
            facecolor="black",
            format="png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
        # <<< Saving Results <<<

#! 利用lidar2img 重写角点映射函数
def project_box(box, lidar2img, img_shape):
    """
    将3D边界框投影到2D图像平面
    Args:
        box: LiDARInstance3DBoxes对象,包含一个3D边界框
        lidar2img: 雷达到图像的投影矩阵 (4x4)
        img_shape: 图像尺寸 (H, W)
    Returns:
        list: 包含投影坐标的列表,每个元素为 tensor 形状 (8,2)
              如果投影无效则返回空列表
    """
    try:
        corners = box.corners[0]  # (8,3)
        device = corners.device
        ones = torch.ones(corners.shape[0], 1, device=device)
        coords = torch.cat([corners, ones], dim=-1)  # (8,4)

        lidar2img = torch.tensor(
            lidar2img, 
            dtype=torch.float32,
            device=device
        )

        coords = coords @ lidar2img.T  # (8,4)
        coords[:, 2] = torch.clamp(coords[:, 2], min=1e-5, max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]
        
        # 检查投影点是否在图像范围内
        valid = (coords[:, 0] >= 0) & (coords[:, 0] < img_shape[1]) & \
                (coords[:, 1] >= 0) & (coords[:, 1] < img_shape[0])
        # 如果有任何点在图像范围内,则返回投影坐标
        if valid.any():
            return [coords[..., :2]]  # 返回tensor (8,2)
        else:
            return []
            
    except Exception as e:
        print(f"投影失败: {e}")
        return []
    

def draw_boxes(img, corners, color, text):
    """
    Args:
        corners:  [N,2] Numpy array
        color: BGR tuple
    """
    # 提取张量并转换为 NumPy 数组
    if isinstance(corners, list) and len(corners) == 1:
        corners = corners[0].cpu().numpy()  # 转换为 [8,2] 的 NumPy 数组
    else:
        print(f"警告：corners 格式错误，期望为包含单个张量的列表，实际为 {type(corners)}")
        return
    if len(corners) < 8:
        print(f"警告：角点数量不足，期望8个，实际为 {len(corners)}")
        return

    # 3D Box lines
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7)  # Side
    ]
    for (i, j) in connections:
        try:
            pt1 = tuple(np.round(corners[i]).astype(int))
            pt2 = tuple(np.round(corners[j]).astype(int))
            cv2.line(img, pt1, pt2, color, 2)
        except Exception as e:
            print(f"绘制边线失败：{e}")
            continue

    try:
        center = np.mean(corners[:4], axis=0).astype(int)
        cv2.putText(img, text, (center[0], center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except:
        pass


def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框的IoU。

    Args:
        bbox1 (list): 第一个边界框的坐标。
        bbox2 (list): 第二个边界框的坐标。

    Returns:
        float: IoU值。
    """
    # Assuming bbox1 and bbox2 are [x1, y1, x2, y2]
    x_max_1 = np.max(bbox1[:, 0])
    y_max_1 = np.max(bbox1[:, 1])
    x_min_1 = np.min(bbox1[:, 0])
    y_min_1 = np.min(bbox1[:, 1])

    x_max_2 = np.max(bbox2[:, 0])
    y_max_2 = np.max(bbox2[:, 1])
    x_min_2 = np.min(bbox2[:, 0])
    y_min_2 = np.min(bbox2[:, 1])

    x1 = max(x_min_1, x_min_2)
    y1 = max(y_min_1, y_min_2)
    x2 = min(x_max_1, x_max_2)
    y2 = min(y_max_1, y_max_2)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    bbox2_area = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)

    iou = inter_area / (bbox1_area + bbox2_area - inter_area)
    return iou


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--output_dir", type=str, default="match_vis_result") # 可视化结果保存路径
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--with_text", action="store_true", help="show results")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    pprint(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    distributed = False
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model)
    
    # model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

        # visualize_topK_matches(data_loader, outputs, topK=4, output_dir=args.output_dir)
        xlim = [cfg.point_cloud_range[d] for d in [0, 3]]
        ylim = [cfg.point_cloud_range[d] for d in [1, 4]]
        visualize_misdetections(data_loader, outputs, 
                                output_dir=args.output_dir, 
                                threshold=0.3, with_text=args.with_text,
                                xlim=xlim, ylim=ylim)

if __name__ == "__main__":
    main()
