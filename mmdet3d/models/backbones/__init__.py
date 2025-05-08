# 从mmdet中导入2D检测的backbone网络
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt

# 导入3D检测的backbone网络
from .resnet import *  # 通用ResNet网络
from .second import *  # SECOND网络,用于点云检测
from .sparse_encoder import *  # 稀疏编码器,用于处理稀疏点云数据  更加通用，但也更复杂
from .pillar_encoder import *  # 点柱编码器,用于PointPillars 适用于PointPillars架构
from .vovnet import *  # VoVNet网络,用于特征提取 以OSA为基本模块，不像DenseNet那样密集连接,而是最后一次性聚合所有特征
from .dla import *  # DLA网络,用于特征提取