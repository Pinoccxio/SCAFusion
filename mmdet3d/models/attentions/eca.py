import torch
from torch import nn
from mmdet3d.models.builder import ATTENTIONS
import math
__all__ = ["ECA", "DFM"]

@ATTENTIONS.register_module()
class DFM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DFM, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # (256,180,180) -> (256,1,1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自适应特征选择
        f_avg = self.global_avg_pool(x)
        f_weight = self.sigmoid(self.conv1x1(f_avg))
        # 通道加权
        f_adaptive = f_weight * x
        return f_adaptive

@ATTENTIONS.register_module()
class ECA(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super(ECA, self).__init__()
        # 自适应确定卷积核大小
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k_size = max(3, t if t % 2 else t + 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 使用1D卷积进行跨通道交互
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全局平均池化
        y = self.avg_pool(x)  # [B,C,H,W] -> [B,C,1,1]  
        # 处理维度以适应1D卷积
        y = y.squeeze(-1).transpose(-1, -2)  # [B,C,1,1] -> [B,1,C]   
        # 1D卷积实现局部跨通道交互
        y = self.conv(y)  # [B,1,C] 
        # 恢复维度并应用sigmoid
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B,C,1,1]
        y = self.sigmoid(y)
        # 通道加权
        return x * y