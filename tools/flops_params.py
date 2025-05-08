import argparse                                                   # 命令行参数解析
import copy                                                       # 对象拷贝
import os                                                         # 操作系统接口
import warnings                                                   # 警告控制
import mmcv                                                       # MMCV工具包
import torch                                                      # PyTorch深度学习框架
from torchpack.utils.config import configs                        # torchpack配置工具
from torchpack import distributed as dist                         # 分布式训练工具
from mmcv import Config, DictAction                              # MMCV配置类和字典操作
from mmcv.cnn import fuse_conv_bn                                # 卷积层和BN层融合
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel  # 数据并行和分布式并行
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model  # 分布式信息、初始化、加载检查点等
from mmdet3d.apis import single_gpu_test                         # 单GPU测试接口
from mmdet3d.datasets import build_dataloader, build_dataset     # 数据加载器和数据集构建 这里build_dataloader来自mmdet.datasets.builder.py
from mmdet3d.models import build_model                           # 模型构建
from mmdet.apis import multi_gpu_test, set_random_seed          # 多GPU测试和随机种子设置
from mmdet.datasets import replace_ImageToTensor                 # 图像张量转换
from mmdet3d.utils import recursive_eval                         # 递归评估工具
from pprint import pprint #! 新增
from thop import profile#! 新增
from torchinfo import summary
from mmdet.models.backbones.swin import SwinTransformer


def get_model_flops_params(model, input):
    model.eval()
    flops, params = profile(model, (input), verbose=True)
    print('-' * 50)
    print(f'flops: {flops/1e9:.2f} G')
    print(f'params: {params/1e6:.2f} M')
    print('-' * 50)
    return flops, params

if __name__ == "__main__":
    model = SwinTransformer()
    input = torch.randn(1, 6, 3, 256, 704)
    print(model)
    get_model_flops_params(model, input)

    # summary(model, input_size=(6, 3, 256, 704))
    

