# >>> “Coordinate Attention” >>>
import torch
from torch import nn

from mmdet3d.models.builder import ATTENTIONS

__all__ = ["CA",
        "SCA"
        ]


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# 暂时没用到 被认为是比ReLU更好的激活函数
class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


@ATTENTIONS.register_module()
class CA(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # h,w   h不变, w变为1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # h,w   w不变, h变为1

        mip = max(8, in_channels // groups) # 8

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x) #  B,256,180,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2) #  B,256,1,180 -> B,256,180,1

        y = torch.cat([x_h, x_w], dim=2) # B,256,360,1
        y = self.conv1(y) # B,8,360,1
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2) # B,8,180,1
        x_w = x_w.permute(0, 1, 3, 2) # B,8,1,180

        x_h = self.conv2(x_h).sigmoid() # B,256,180,1
        x_w = self.conv3(x_w).sigmoid() # B,256,1,180
        x_h = x_h.expand(-1, -1, h, w) # B,256,180,180
        x_w = x_w.expand(-1, -1, h, w) # B,256,180,180

        y = identity * x_w * x_h

        return y

class SCA(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        super(CSA, self).__init__()
        self.ca = CA(in_channels, out_channels, groups)
        self.relu = h_swish()

    def forward(self, x):
        # CA
        x = self.ca(x)
        # SA
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, 180, 180)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, 180, 180)
        spatial_out = torch.cat([avg_out, max_out], dim=1) # (B, 2, 180, 180)
        spatial_out = self.relu(self.conv(spatial_out)) # (B, 1, 180, 180)

        x = x * spatial_out
        return x

