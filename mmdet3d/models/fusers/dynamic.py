from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["DynamicFuser"]

@FUSERS.register_module()
class DynamicFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int)-> None:
        super(DynamicFuser, self).__init__()
        self.conv3x3 = nn.Conv2d(sum(in_channels), out_channels, kernel_size=3, padding=1) # 默认有bias
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # 通道拼接
        f_concat = torch.cat(inputs, dim=1)
        
        # 3x3卷积
        f_static = self.conv3x3(f_concat)
        
        # 自适应特征选择
        f_avg = self.global_avg_pool(f_static)
        f_weight = self.sigmoid(self.conv1x1(f_avg))
        
        # 通道加权
        f_adaptive = f_weight * f_static
        
        return f_adaptive
