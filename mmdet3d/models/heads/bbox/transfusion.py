import copy  # 用于深拷贝对象

import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch函数式接口
from mmcv.cnn import ConvModule, build_conv_layer  # MMCV中的卷积模块
from mmcv.runner import force_fp32  # 强制使用FP32精度的装饰器
from torch import nn  # PyTorch神经网络模块

from mmdet3d.core import (
    PseudoSampler,  # 伪采样器
    circle_nms,  # 圆形NMS
    draw_heatmap_gaussian,  # 绘制高斯热图
    gaussian_radius,  # 计算高斯半径
    xywhr2xyxyr,  # 坐标系转换
)
from mmdet3d.models.builder import HEADS, build_loss  # 模型构建器
from mmdet3d.models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer  # Transformer相关组件
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu  # 3D IoU计算和NMS
from mmdet.core import (
    AssignResult,  # 分配结果 用于存储分配结果。分配结果通常包括每个预测框与真实框的匹配信息，比如匹配的索引、最大重叠度和标签等
    build_assigner,  # 构建分配器 分配器的作用是根据一定的策略（如IOU、距离等）将预测框与真实框进行匹配。不同的分配器可以实现不同的匹配策略。
    build_bbox_coder,  # 构建边界框编码器 ）。边界框编码器的作用是将边界框的坐标进行编码和解码，以便于在网络中进行回归任务。编码器可以将实际的边界框坐标转换为网络输出的格式，反之亦然。
    build_sampler,  # 构建采样器 采样器的作用是在训练过程中从预测框中选择正样本和负样本，以便于计算损失。采样策略可以影响模型的训练效果。
    multi_apply,  # 这是一个工具函数，用于将一个函数应用到多个输入上。它可以简化对多个输入进行相同操作的代码编写，常用于批量处理。
)
__all__ = ["TransFusionHead"]


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

@HEADS.register_module()
class TransFusionHead(nn.Module):
    # sample_count = 0 #!
    def __init__(
        self,
        num_proposals=128, # 200
        auxiliary=True,
        in_channels=128 * 3, # 512
        hidden_channel=128, 
        num_classes=4, # 10
        # config for Transformer
        num_decoder_layers=3, # 1
        num_heads=8, 
        nms_kernel_size=1, # 3
        ffn_channel=256, 
        dropout=0.1,
        bn_momentum=0.1,
        activation="relu",
        # config for FFN
        common_heads=dict(), # e.g. center [2,2] height [1,2] dim [3,2] rot [2,2] vel [2,2] 特征维度，卷积层数
        num_heatmap_convs=2,
        conv_cfg=dict(type="Conv1d"),
        norm_cfg=dict(type="BN1d"),
        bias="auto",
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"), # FocalLoss
        loss_iou=dict(type="VarifocalLoss", use_sigmoid=True, iou_weighted=True, reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean"), # L1Loss
        loss_heatmap=dict(type="GaussianFocalLoss", reduction="mean"),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None, # TransFusionBBoxCoder
    ):
        super(TransFusionHead, self).__init__()
        self.fp16_enabled = False
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False) # true
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls) # Focalloss
        self.loss_bbox = build_loss(loss_bbox) # L1Loss
        self.loss_iou = build_loss(loss_iou) # VarifocalLoss
        self.loss_heatmap = build_loss(loss_heatmap) # GaussianFocalLoss

        self.bbox_coder = build_bbox_coder(bbox_coder) # TransFusionBBoxCoder
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type="Conv2d"),
            in_channels, # 512
            hidden_channel, # 128
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel, # 128
                hidden_channel, # 128
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="BN2d"),
            )
        )
        layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                hidden_channel, # 128
                num_classes, # 10
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, # 128
                    num_heads, # 8
                    ffn_channel, # 256
                    dropout, # 0.1
                    activation, # relu
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel), # (2,128)
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel), # (2,128)
                )
            )

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs))) #? heatmap : (10,2) 
            self.prediction_heads.append(
                FFN(
                    hidden_channel, # 128
                    heads,  # e.g. center [2,2] height [1,2] dim [3,2] rot [2,2] vel [2,2] 特征维度，卷积层数
                    conv_cfg=conv_cfg, # Conv1d
                    norm_cfg=norm_cfg, # BN1d
                    bias=bias, # auto
                )
            )

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        # grid_size=[1440, 1440, 41] output_size_factor=8
        x_size = self.test_cfg["grid_size"][0] // self.test_cfg["out_size_factor"] # 1440 // 8 = 180
        y_size = self.test_cfg["grid_size"][1] // self.test_cfg["out_size_factor"] # 1440 // 8 = 180
        self.bev_pos = self.create_2D_grid(x_size, y_size) # (1,180^2,2)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]] 
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        # +0.5偏置，使得坐标在每个网格的中心而不是边界上
        batch_x = batch_x + 0.5 # [180,180] 每行x相同   [0.5, 1.5, 2.5, ..., 179.5]
        batch_y = batch_y + 0.5 
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None] # [1,2,180,180]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1) # [1,180^2,2]
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return
        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            # PseudoSampler在这里更像是一个占位符，
            # 它的主要作用是将assigner的分配结果直接转换为采样结果的格式，而不进行实际的采样操作。
            self.bbox_sampler = PseudoSampler() 
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner) 
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    # 对应一个level  默认只处理一个layer
    def forward_single(self, inputs, img_inputs, metas):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs) # 这里lidar_feat就是fused_feat

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # [BS, C, H*W]   (B,128,180^2)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device) # (B,180^2,2)

        #################################
        # image guided query initialization
        #################################
        dense_heatmap = self.heatmap_head(lidar_feat) # (B,10,180,180)  
        dense_heatmap_img = None
        heatmap = dense_heatmap.detach().sigmoid()
        #!
        # if self.sample_count == 10:
        #     heatmap_save = copy.deepcopy(heatmap)
        #     np.save(f'/root/bs/__tmp__/heatmap.npy', heatmap_save.cpu().numpy())
        # self.sample_count += 1

        padding = self.nms_kernel_size // 2 # 3//2=1
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        ) # (B,10,178,178)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner # (B,10,180,180)
        ## for Pedestrian & Traffic_cone in nuScenes #? 对小目标(行人，交通锥)进行特殊处理
        #! 对meteor特殊处理
        if self.test_cfg["dataset"] == "nuScenes":
            # local_max[
            #     :,
            #     0,
            # ] = F.max_pool2d(heatmap[:, 0], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                8,
            ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                9,
            ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
            local_max[
                :,
                1,
            ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[
                :,
                2,
            ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max) # 只保留局部最大值 其他置0
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1) # (B,10,180^2) #? 相当于是置信度图

        # top #num_proposals among all classes
        # (B,10*180^2) -> 返回降序后的索引(B,10*180^2) -> 取前200个 (B,200)
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1] # (B,200) // 180^2
        top_proposals_index = top_proposals % heatmap.shape[-1] # (B,200) % 180^2
        # (B,128,180^2)
        # (B,200) -> (B,1,200) -> (B,128,200),根据200对应的索引进行
        query_feat = lidar_feat_flatten.gather( 
            index=top_proposals_index[:, None, :].expand(
                -1, lidar_feat_flatten.shape[1], -1
            ),
            dim=-1,
        ) # (B,128,200)
        self.query_labels = top_proposals_class #? (B,200)

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
            0, 2, 1
        ) # (B,10,200)
        query_cat_encoding = self.class_encoding(one_hot.float()) # (B,10,200)->(B,128,200) 
        query_feat += query_cat_encoding #? (B,128,200)   class dependent embedding
        # (B,180^2,2)
        # (B,200) -> (B,200,1) -> (B,200,2)
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :]
            .permute(0, 2, 1)
            .expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        ) #? (B,200,2)

        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat, # (B,128,200)
                lidar_feat_flatten, # (B,128,180^2)
                query_pos, # (B,200,2)
                bev_pos, # (B,180^2,2)
            ) # (B, 128, 200)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)  # (B, dim, 200)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1) # 预测偏移 + 参考点
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1) # 变回来

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        # ret_dicts[0]["heatmap"] (B,10,200) # 经过sigmoid
        ret_dicts[0]["query_heatmap_score"] = heatmap.gather( # (B,10,180^2)
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), # (B,10,200)
            dim=-1,
        )  # (B,10,200)
        ret_dicts[0]["dense_heatmap"] = dense_heatmap # (B,10,180,180) 无sigmoid

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        # 除了dense_heatmap, dense_heatmap_old, query_heatmap_score 其他都concat
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1
                ) # (B,dim,200*num_decoder_layers)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res] 

    #TODO 输入(B,512,180,180)   
    def forward(self, feats, metas):
        """Forward pass.
        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
           #? tuple(list[dict]): Output results. first index by level, second index by layer 
           #? 只有一个level一个decoder.layer  ->  ret[0][0]   
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [None], [metas]) # img_inputs=None
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx : batch_idx + 1] # 使用切片会保留bs维度
            list_of_pred_dict.append(pred_dict) # list长度为batch_size 字典e.g. ['center']:[1, 3, num_proposal]

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply( # 对每个batch进行操作
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            list_of_pred_dict,
            np.arange(len(gt_labels_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels, # (B,200)
            label_weights, # (B,200)
            bbox_targets, # (B,200,10)
            bbox_weights, # (B,200,10)
            ious, # (B,200)
            num_pos, # batch_size*每个batch的正样本数量
            matched_ious, # 平均iou
            heatmap, # (B,10,180,180)
        )

    #TODO 用于训练阶段，包含预测尺度恢复
    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict["center"].shape[-1] # 200

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict["heatmap"].detach()) # (1,10,200)
        center = copy.deepcopy(preds_dict["center"].detach()) # (1,2,200)
        height = copy.deepcopy(preds_dict["height"].detach()) # (1,1,200)
        dim = copy.deepcopy(preds_dict["dim"].detach()) # (1,3,200)
        rot = copy.deepcopy(preds_dict["rot"].detach()) # (1,2,200)
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach()) # (1,2,200)
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height, vel
        )  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]["bboxes"]
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[
                self.num_proposals * idx_layer : self.num_proposals * (idx_layer + 1), :
            ]
            score_layer = score[
                ...,
                self.num_proposals * idx_layer : self.num_proposals * (idx_layer + 1),
            ]

            if self.train_cfg.assigner.type == "HungarianAssigner3D": # 这个
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer, # (200, 10)
                    gt_bboxes_tensor, 
                    gt_labels_3d, 
                    score_layer, # (200, 10)
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == "HeuristicAssigner":
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result) # assign_result_list长度为num_layer 每个元素为AssignResult

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]), # gt总数
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]), # 每个proposal对应的gt索引
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]), # 每个proposal对应的gt最大iou
            labels=torch.cat([res.labels for res in assign_result_list]), # 每个proposal对应的gt类别
        )
        sampling_result = self.bbox_sampler.sample(
            assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor
        )
        pos_inds = sampling_result.pos_inds # 正样本索引(gt_inds>0的位置)
        neg_inds = sampling_result.neg_inds # 负样本索引(gt_inds=0的位置)
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation (200,10)
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(
            center.device
        )
        ious = assign_result_ensemble.max_overlaps # (200)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) # (200)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) # (200)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes # 全部10

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0: #-1
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
        ).to(device)
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])
        feature_map_size = (
            grid_size[:2] // self.train_cfg["out_size_factor"]
        )  # [x_len, y_len]   # (180,180)
        heatmap = gt_bboxes_3d.new_zeros(
            self.num_classes, feature_map_size[1], feature_map_size[0]
        )  # (10,180,180)
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
            length = length / voxel_size[1] / self.train_cfg["out_size_factor"]
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width), min_overlap=self.train_cfg["gaussian_overlap"] # 0.1
                )
                radius = max(self.train_cfg["min_radius"], int(radius)) # 2
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.train_cfg["out_size_factor"]
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.train_cfg["out_size_factor"]
                )

                center = torch.tensor(
                    [coor_x, coor_y], dtype=torch.float32, device=device
                )
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
                # NOTE: fix
                draw_heatmap_gaussian(
                    heatmap[gt_labels_3d[idx]], center_int[[1, 0]], radius
                ) # 交换x,y坐标 因为heatmap坐标系是yx

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None], # (1,200)
            label_weights[None], # (1,200)
            bbox_targets[None], # (1,200,10)
            bbox_weights[None], # (1,200,10)
            ious[None], # (1,200)
            int(pos_inds.shape[0]), # 正样本数量
            float(mean_iou), # 平均iou
            heatmap[None], # (1,10,180,180)
        )

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        (
            labels, # (B,200)
            label_weights, # (B,200)
            bbox_targets, # (B,200,10)
            bbox_weights, # (B,200,10)
            ious, # (B,200)
            num_pos, # batch_size*每个batch的正样本数量
            matched_ious, # 平均iou
            heatmap,
        ) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        if hasattr(self, "on_the_image_mask"):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict() 
        # loss_heatmap/loss_matched_ious不分层
        # loss_cls/loss_bbox分层

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict["dense_heatmap"]),  # (B,10,180,180)
            heatmap,  # (B,10,180,180)
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1), # heatmap.eq(1)找到正样本, 计算平均因子
        )
        loss_dict["loss_heatmap"] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                idx_layer == 0 and self.auxiliary is False
            ):
                prefix = "layer_-1" # 由于只用了一层decoder layer 所以前缀为-1
            else:
                prefix = f"layer_{idx_layer}"
            #? cls
            layer_labels = labels[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1) # (B*200,)
            layer_label_weights = label_weights[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1) # (B*200,)
            layer_score = preds_dict["heatmap"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes) # (B*200,10)
            layer_loss_cls = self.loss_cls( # 仅用预测值训练而没有结合热力图置信度
                layer_cls_score,
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )


            #? bbox
            layer_center = preds_dict["center"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ] # (B,2,200)
            layer_height = preds_dict["height"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ] # (B,1,200)
            layer_rot = preds_dict["rot"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ] # (B,2,200)
            layer_dim = preds_dict["dim"][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ] # (B,3,200)
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot], dim=1
            ).permute(
                0, 2, 1
            )  # [BS, num_proposals, code_size] # (B,200,8)
            if "vel" in preds_dict.keys():
                layer_vel = preds_dict["vel"][
                    ...,
                    idx_layer
                    * self.num_proposals : (idx_layer + 1)
                    * self.num_proposals,
                ]
                preds = torch.cat(
                    [layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1
                ).permute(
                    0, 2, 1
                )  # [BS, num_proposals, code_size] # (B,200,10)
            code_weights = self.train_cfg.get("code_weights", None)
            layer_bbox_weights = bbox_weights[
                :,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
                :,
            ] # (B,200,10)
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(
                code_weights
            ) # (B,200,10)
            layer_bbox_targets = bbox_targets[
                :,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
                :,
            ] # (B,200,10)
            layer_loss_bbox = self.loss_bbox(
                preds,  # (B,200,10)
                layer_bbox_targets,  # (B,200,10)
                layer_reg_weights,  # (B,200,10)
                avg_factor=max(num_pos, 1) # 平均因子
            )
            
            #? iou
            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            loss_dict[f"{prefix}_loss_cls"] = layer_loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f"matched_ious"] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    #TODO 用于val阶段 包含预测尺度恢复
    def get_bboxes(self, preds_dicts, metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]["heatmap"].shape[0]
            # (B,10,200*num_decoder_layers) -> (B,10,200)
            batch_score = preds_dict[0]["heatmap"][..., -self.num_proposals :].sigmoid() 
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(
                self.query_labels, num_classes=self.num_classes
            ).permute(0, 2, 1)  # (B,10,200)
            # (B,10,200) * (B,10,200) * (B,10,200) 预测分数x置信度x类别掩码
            batch_score = batch_score * preds_dict[0]["query_heatmap_score"] * one_hot
            batch_center = preds_dict[0]["center"][..., -self.num_proposals :] # (B,2,200)
            batch_height = preds_dict[0]["height"][..., -self.num_proposals :] # (B,1,200)
            batch_dim = preds_dict[0]["dim"][..., -self.num_proposals :] # (B,3,200)
            batch_rot = preds_dict[0]["rot"][..., -self.num_proposals :] # (B,2,200)
            batch_vel = None
            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"][..., -self.num_proposals :] # (B,2,200)

            temp = self.bbox_coder.decode(  #TODO 这一步恢复了解的尺度
                batch_score, # (B,10,200)
                batch_rot, # (B,2,200)
                batch_dim, # (B,3,200)
                batch_center, # (B,2,200)
                batch_height, # (B,1,200)
                batch_vel, # (B,2,200)
                filter=True,
            ) # 将网络输出变为最终3D边界框的形式 list(dict(bboxes,scores,labels)) list长度为batch_size

            if self.test_cfg["dataset"] == "nuScenes":
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=["pedestrian"],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=["traffic_cone"],
                        indices=[9],
                        radius=0.175,
                    ),
                    #! 与数据集相关
                    # dict(
                    #     num_class=1,
                    #     class_names=["meteor"],
                    #     indices=[0],
                    #     radius=0.175,
                    # ),
                    # dict(
                    #     num_class=1,
                    #     class_names=["platform"],
                    #     indices=[1],
                    #     radius=-1,
                    # ),
                ]
            elif self.test_cfg["dataset"] == "Waymo":
                self.tasks = [
                    dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                    dict(
                        num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7
                    ),
                    dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]["bboxes"] # (200, 10)
                scores = temp[i]["scores"] # (200)
                labels = temp[i]["labels"] # (200)
                ## adopt circle nms for different categories
                if self.test_cfg["nms_type"] != None:  # null
                    # 这里步骤是实现了原形NMS和传统NMS  
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task["indices"]:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task["radius"] > 0:
                            if self.test_cfg["nms_type"] == "circle":
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task["radius"],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]["box_type_3d"](
                                        boxes3d[task_mask][:, :7], 7
                                    ).bev
                                )
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task["radius"],
                                    pre_maxsize=self.test_cfg["pre_maxsize"],
                                    post_max_size=self.test_cfg["post_maxsize"],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][
                                task_keep_indices
                            ]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1 # level = 1
        assert len(rets[0]) == 1  # 因为是测试用，所以batch_size=1
        res = [
            [
                metas[0]["box_type_3d"]( # LiDARInstance3DBoxes
                    rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1]
                ),
                rets[0][0]["scores"],
                rets[0][0]["labels"].int(),
            ]
        ]
        return res   # res[0][0]
