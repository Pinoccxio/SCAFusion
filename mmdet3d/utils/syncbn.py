import copy
import torch
# 导入双端队列数据结构,用于实现先进先出(FIFO)队列
# 在这里虽然导入了deque,但实际上并未使用,可以考虑删除该导入
from collections import deque


__all__ = ["convert_sync_batchnorm"]


def convert_sync_batchnorm(input_model, exclude=[]):
    """将模型中的BatchNorm层转换为SyncBatchNorm层
    BatchNorm: 在单个GPU内部计算
    SyncBatchNorm: 在所有GPU上同步计算
    
    Args:
        input_model: 需要转换的PyTorch模型
        exclude: 不需要转换的模块名称列表,这些模块将被跳过
        
    Returns:
        转换后的模型,其中除了exclude指定的模块外,所有BatchNorm层都被替换为SyncBatchNorm层
    """
    # 遍历模型中的所有模块
    for name, module in input_model._modules.items():
        # 检查当前模块名称是否在exclude列表中
        skip = sum([ex in name for ex in exclude])
        if skip:
            continue
        # 将当前模块的BatchNorm层转换为SyncBatchNorm层
        input_model._modules[name] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
    return input_model
    