from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32  # 类方法装饰器
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_attention,
    build_head,
    build_neck,
    build_vtransform,
)
# 动态散射，当max_num_points=-1时，使用动态散射, 相比固定大小的体素化,动态散射可以更灵活地处理点的分布
from mmdet3d.ops import Voxelization, DynamicScatter  #? DynamicScatter()先不看
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.utils.visual import *
from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    #!
    sample_cnt = 0
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        attention: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            # 是否对体素内的点特征进行聚合 cfg中无，默认是True
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True) 

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        #TODO 这一部分是相对于原版的bevfusion 增加的
        if attention is not None: 
            self.attention = build_attention(attention)
        else:
            self.attention = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )

        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])
       
        # 针对各head的loss_scale 只检查顶层键 cfg顶层键无loss_scale
        #TODO 可配置各个头的损失权重
        if "loss_scale" in kwargs:  
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights() #自动调用
        
        #! aux
        self.aux_flag = kwargs.get("aux_flag", False)
        if self.aux_flag:
            self.camera_aux_net(kwargs)

    def init_weights(self) -> None: # 仅针对相机backbone初始化
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def freeze_params(self, cfg):
        """根据配置冻结模型参数"""
        frozen_stage = cfg.model_freeze_config.frozen_stages
        if frozen_stage == -1:
            return
        frozen_params = cfg.model_freeze_config.frozen_params[f'stage{frozen_stage}']
        if getattr(frozen_params, "encoder_camera_backbone", False):
            self.encoders["camera"]["backbone"].requires_grad_(False)
            self.encoders["camera"]["backbone"].eval()
        if getattr(frozen_params, "encoder_camera_neck", False):
            self.encoders["camera"]["neck"].requires_grad_(False)
            self.encoders["camera"]["neck"].eval()
        if getattr(frozen_params, "encoder_camera_vtransform", False):
            self.encoders["camera"]["vtransform"].requires_grad_(False)
            self.encoders["camera"]["vtransform"].eval()
        if getattr(frozen_params, "encoder_lidar_backbone", False):
            self.encoders["lidar"]["backbone"].requires_grad_(False)
            self.encoders["lidar"]["backbone"].eval()

        if getattr(frozen_params, "fuser", False):
            self.fuser.requires_grad_(False)
            self.fuser.eval()
        if getattr(frozen_params, "attention", False):
            self.attention.requires_grad_(False)
            self.attention.eval()
        if getattr(frozen_params, "decoder_backbone", False):
            self.decoder["backbone"].requires_grad_(False)
            self.decoder["backbone"].eval()
        if getattr(frozen_params, "decoder_neck", False):
            self.decoder["neck"].requires_grad_(False)
            self.decoder["neck"].eval()

        if getattr(frozen_params, "head_object", False):
            self.heads["object"].requires_grad_(False)
            self.heads["object"].eval()
        if getattr(frozen_params, "head_map", False):
            self.heads["map"].requires_grad_(False)
            self.heads["map"].eval()

    #! aux
    def camera_aux_net(self, kwargs):
        decoder = kwargs['aux_decoder']
        heads = kwargs['aux_heads']
        self.aux_decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.aux_heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.aux_heads[name] = build_head(heads[name])  

        if "loss_scale" in kwargs:  
            self.aux_loss_scale = kwargs["loss_scale"]
        else:
            self.aux_loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.aux_loss_scale[name] = 0.05


    def extract_camera_features( # 经过pipeline后输出的data_dict
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        #TODO 输入 (B,cam,3,256,704)
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W) 
        #? 把相机个数放入batch维度,因此在camera backbone中就不用区分camera个数
        x = self.encoders["camera"]["backbone"](x)
        #TODO 输出 (B*cam,192,32,88) (B*cam,384,16,44) (B*cam,768,8,22)
        x = self.encoders["camera"]["neck"](x)
        #TODO 输出 (B*cam,256,32,88) (B*cam,256,16,44)

        # 有的neck输出一个特征列表，测试选择第一个(通常是最高分辨率下的特征图)
        if not isinstance(x, torch.Tensor):
            x = x[0] 

        #? 恢复到相机个数维度 vtransform中需要用到相机个数
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W) 
        #TODO 输入 (B,cam,256,32,88)
        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x
        #TODO 输出 (B,80,180,180)

    def extract_lidar_features(self, x) -> torch.Tensor:  # 输入x是点云数据points list(tensor[num_points,5]) list长度为batch_size   
        feats, coords, sizes = self.voxelize(x)
        # feats [M_total,ndim]
        # coords [M_total,4] # batch_index x y z 
        # sizes [M_total]
        batch_size = coords[-1, 0] + 1 # 最后一个batch_index + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes) 
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):  # res tensor[num_points,5]
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3: # Voxelization()
                # f [M,max_points_per,ndim]
                # c [M,3]
                # n [M] 元素表示每个体素的实际点数
                f, c, n = ret 
            else: # DynamicScatter() #? 暂不处理
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            # F为torch.nn.functional
            # 把[M,3]的坐标扩充为[M,4] 第1维为batch_index
            # (1,0)表示在左边扩充1维，右边扩充0维
            # mode="constant"表示扩充的值为常数，value=k表示扩充的值为k
            # 结果为[M,4] 第[k,x,y,z]为体素坐标
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))  #? 因为待会要cat 这里保存batch_index信息
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0) # 沿batch维度拼接, [M_total,max_points_per,ndim]
        coords = torch.cat(coords, dim=0) # [M_total,4]
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0) # [M_total]
            if self.voxelize_reduce: # 把每个体素内的点特征进行聚合到以体素为单位 [M_total,ndim]
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous() # 确保数据在内存中连续存储，提高后续处理效率

        # feats [M_total,ndim]
        # coords [M_total,4]
        # sizes [M_total]
        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points")) # 封装了forward_single()
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # img [batch_size,camera_num,3,H,W]
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    #TODO 具体处理过程看这   按batchsize处理
    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None, # 相对于lidar坐标系
        gt_labels_3d=None,
        **kwargs,
    ): 
        #? self.training在EvalHook中会设置为False
        features = []   #self.training是nn.Module的内置属性 [start:end:step],step=-1表示反向
        # 在推理时先处理点云数据，优化内存占用
        for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor == "camera":
                #TODO 输入 (B,cam_num,3,H,W)
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                ) 
                #TODO 输出 (B,80,180,180)
                #! aux
                if self.aux_flag:
                    cam_aux_feature = feature
            elif sensor == "lidar":
                #TODO 输入 list(tensor[num_points,5]) list长度为batch_size  类似于(B,num_points,5)
                feature = self.extract_lidar_features(points)
                #TODO 输出 (B,256,180,180)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature) #TODO 输出 list[(B,80,180,180), (B,256,180,180)]
            

        
        if not self.training: # 如果不是训练的话，再倒回来
            # avoid OOM
            features = features[::-1]
   
        if self.fuser is not None: 
            #TODO 输入 list[(B,80,180,180), (B,256,180,180)]
            x = self.fuser(features)  # 先相机后雷达
            #TODO 输出 (B,256,180,180)
        else:
            assert len(features) == 1, features
            x = features[0]

        # #!
        # if self.sample_cnt == 10:
        #     visualize_bev_feature(features[0],'/root/bs/__tmp__/cam_bev.png',cv2.COLORMAP_JET)
        #     visualize_bev_feature(features[1],'/root/bs/__tmp__/lid_bev.png',cv2.COLORMAP_JET)
        #     visualize_bev_feature(x,'/root/bs/__tmp__/fuse_bev.png',cv2.COLORMAP_JET)

        #TODO 输入 (B,256,180,180)
        if self.attention is not None:
            x = self.attention(x)  
        #TODO 输出 (B,256,180,180)
        # #!
        # if self.sample_cnt == 10:
        #     visualize_bev_feature(x,'/root/bs/__tmp__/attn_bev.png',cv2.COLORMAP_JET)
        # self.sample_cnt += 1

        batch_size = x.shape[0]

        #TODO 输入 (B,256,180,180)
        x = self.decoder["backbone"](x)
        #TODO 输出 tuple[(B,128,180,180), (B,256,90,90)]

        #TODO 输入 tuple[(B,128,180,180), (B,256,90,90)]
        x = self.decoder["neck"](x)
        #TODO 输出 [(B,512,180,180)]

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    #TODO 输出 dict(loss_heatmap,loss_cls,loss_bbox,loss_matched_ious)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]  #默认是1.0
                    else:
                        outputs[f"stats/{type}/{name}"] = val

            #! align
            if hasattr(self.encoders["camera"]["vtransform"], "align_loss"):
                outputs["loss/align_loss"] = self.encoders["camera"]["vtransform"].align_loss
            
            #! aux
            if self.aux_flag:
                #TODO 输入 (B,80,180,180)
                x_aux = self.aux_decoder["backbone"](cam_aux_feature)
                #TODO 输出 （B,128,90,90） (B,256,45,45) (B,512,45,45)
                #TODO 输入  (B,128,90,90） (B,512,45,45)
                x_aux = self.aux_decoder["neck"](x_aux)
                #TODO 输出 (B,256,180,180)
                for type, head in self.aux_heads.items():
                    if type == "object":
                        pred_dict = head(x_aux, metas)
                        losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                        #TODO 输出 dict(loss_heatmap,loss_cls,loss_bbox,loss_matched_ious)
                    elif type == "map":
                        losses = head(x_aux, gt_masks_bev)
                    else:
                        raise ValueError(f"unsupported head: {type}")
                    for name, val in losses.items():
                        if val.requires_grad:
                            outputs[f"aux_loss/{type}/{name}"] = val * self.aux_loss_scale[type]  #默认是0.05
                        else:
                            outputs[f"aux_stats/{type}/{name}"] = val
                
            return outputs
        

        else:
            outputs = [{} for _ in range(batch_size)] # 为每个batch创建也给字典来存储结果
            for type, head in self.heads.items():
                if type == "object":
                    #TODO 输入 (B,512,180,180)和metas
                    pred_dict = head(x, metas) # cls_num可在cfg中指定
                    #TODO 输出[0][0] tuple(list(dict()))  其中list层有B个dict()
                    #TODO center (B,2,num_proposals)
                    #TODO height (B,1,num_proposals)
                    #TODO dim (B,2,num_proposals)
                    #TODO rot (B,2,num_proposals)
                    #TODO vel (B,2,num_proposals)
                    #TODO heatmap (B,cls_num,num_queries)

                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                        #TODO 输出 list[dict(boxes_3d,scores_3d,labels_3d)] 长度为B
                elif type == "map":
                    # 另外Logits这个命名一般是指模型最后一层线性层的原始输出
                    logits = head(x) # 当不传入gt_masks_bev时，head(x)返回的是[batch_size,map_classes,H,W]
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
