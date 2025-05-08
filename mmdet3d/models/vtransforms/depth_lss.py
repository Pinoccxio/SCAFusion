from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,  # 2
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2), # o = (i-5+4)/4+1
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), # o = ((i-5+4)/2+1)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential( # 256+64
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1), # (118+80)
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d): 
        #? depth (B,N,1,H,W) 由点云投影而来
        #? x (B,N,256,fH,fW)
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:]) # (BN,1,H,W) # 256 704
        x = x.view(B * N, C, fH, fW) # (BN,C,fH,fW)

        d = self.dtransform(d) # (BN,64,32,88)   
        #! 这一步是否可以先对齐再cat？
        x = torch.cat([d, x], dim=1) # (BN,320,32,88)
        x = self.depthnet(x) # (BN,198,32,88)

        depth = x[:, : self.D].softmax(dim=1) # (BN,118,32,88)
        # (BN,1,118,32,88)  (BN,80,1,32,88)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
        # (BN,80,118,32,88)

        x = x.view(B, N, self.C, self.D, fH, fW) # (B,N,80,118,32,88)
        x = x.permute(0, 1, 3, 4, 5, 2) # (B,N,118,32,88,80) 
        return x # (B,N,D,fH,fW,C)

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs) # ()
        x = self.downsample(x) # (B,80,360,360)
        return x # (B,80,180,180)