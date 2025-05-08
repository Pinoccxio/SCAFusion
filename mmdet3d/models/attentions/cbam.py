import torch
from torch import nn

from mmdet3d.models.builder import ATTENTIONS

__all__ = ["CBAM"]

@ATTENTIONS.register_module()
class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(CBAM, self).__init__()
        #* Channel Attention(CAM)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        #* Spatial Attention(SAM)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #* Channel Attention(CAM)
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out) # (256,1,1)
        x = channel_out * x # （256,180,180）
        
        #* Spatial Attention(SAM)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, 180, 180)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, 180, 180)
        spatial_out = torch.cat([avg_out, max_out], dim=1) # (B, 2, 180, 180)
        spatial_out = self.sigmoid(self.conv(spatial_out)) # (B, 1, 180, 180)
        
        out = spatial_out * x
        return out