from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS

__all__ = ["BEVSegmentationHead"]
# vanilla表示基本模型，没有使用任何复杂的结构


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)

#? 融入BCELoss而非softmax
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

#? 将特征变为标准的BEV表示
class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]], # [-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]
        output_scope: List[Tuple[float, float, float]], # [-50, 50, 0.5], [-50, 50, 0.5]
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = [] # x,y 各一个tensor （长度200）
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep) # /2使得采样点在网格中心
            v = (v - imin) / (imax - imin) * 2 - 1 # 归一化到[-1,1] 表示输出特征图上的点相对于输入特征图的相对位置
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)  # (B,200,200,2)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


@HEADS.register_module()
class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int, # 256
        grid_transform: Dict[str, Any],
        classes: List[str], # drivable_area, ped_crossing, walkway, stop_line, carpark_area, divider
        loss: str, # focal
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss = loss

        self.transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, in_channels, 1),  #! 增加一体化det/seg头
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),
        )

    #TODO 输入(B,512,180,180)
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = self.transform(x)
        x = self.classifier(x)

        if self.training:
            losses = {}
            for index, name in enumerate(self.classes):
                if self.loss == "xent":
                    loss = sigmoid_xent_loss(x[:, index], target[:, index])
                elif self.loss == "focal":
                    loss = sigmoid_focal_loss(x[:, index], target[:, index])
                else:
                    raise ValueError(f"unsupported loss: {self.loss}")
                losses[f"{name}/{self.loss}"] = loss
            return losses
        else:
            return torch.sigmoid(x)
