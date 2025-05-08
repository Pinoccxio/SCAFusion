import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from mmdet3d.models.builder import FUSERS

__all__ = ["UAFuser"]

# cam  (B,256,180,180)
# lid  (B,80,180,180)
# input [cam, lid]
# out  (B,256,180,180)

# @FUSERS.register_module()
class UAFuser(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super(UAFuser, self).__init__()
        # 距离预测器改进：使用多层感知机(MLP)结构
        self.f_dist_cam = nn.Sequential(
            nn.Conv2d(channels[0], channels[0]//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]//2),
            nn.ReLU(True),
            nn.Conv2d(channels[0]//2, channels[0]//4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]//4),
            nn.ReLU(True),
            nn.Conv2d(channels[0]//4, channels[0], 1, bias=False)
        )
        self.f_dist_lid = nn.Sequential(
            nn.Conv2d(channels[1], channels[1]//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]//2),
            nn.ReLU(True),
            nn.Conv2d(channels[1]//2, channels[1]//4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]//4),
            nn.ReLU(True),
            nn.Conv2d(channels[1]//4, channels[1], 1, bias=False)
        )
        # 定义前馈网络 FFN 
        self.ffn = nn.Sequential(
            nn.Conv2d(sum(channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # F_cam: (B,256,180,180), F_lid: (B,80,180,180)
        F_cam = inputs[0]
        F_lid = inputs[1]
        # 计算空间维度上的不确定
        dist_cam = self.f_dist_cam(F_cam)  # (B,256,180,180)
        dist_lid = self.f_dist_lid(F_lid)  # (B,80,180,180)
        # 在空间维度上计算不确定性
        U_cam = 1 - torch.exp(-torch.mean(dist_cam, dim=1, keepdim=True))  # (B,1,180,180)
        U_lid = 1 - torch.exp(-torch.mean(dist_lid, dim=1, keepdim=True))  # (B,1,180,180)
        # 不确定性加权融合
        weighted_F_cam = F_cam * (1 - U_cam)  # (B,256,180,180)
        weighted_F_lid = F_lid * (1 - U_lid)  # (B,80,180,180)
        # 拼接加权后的特征
        fused_features = torch.cat((weighted_F_cam, weighted_F_lid), dim=1)  # (B,336,180,180)
        # 通过前馈网络进行最终处理
        refined_features = self.ffn(fused_features)  # (B,256,180,180)
        return refined_features

# 测试代码
if __name__ == "__main__":
    channel_cam = 256
    channel_lid = 80
    channels = [channel_cam, channel_lid]
    batch_size = 4
    F_cam = torch.randn(batch_size, channel_cam, 180, 180)  # 模拟相机特征
    F_lid = torch.randn(batch_size, channel_lid, 180, 180)  # 模拟LiDAR特征
    uaf_module = UAFuser(channels, 256)
    refined_features = uaf_module([F_cam, F_lid])
    print("Refined Features Shape:", refined_features.shape)