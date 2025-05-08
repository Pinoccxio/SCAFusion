import platform                                           # 导入平台相关模块
from mmcv.utils import Registry, build_from_cfg           # 从mmcv导入注册表和构建函数

from mmdet.datasets import DATASETS                       # 从mmdet导入数据集注册表
from mmdet.datasets.builder import _concat_dataset        # 从mmdet导入数据集连接函数

#? 确保系统可以至少同时打开4096个文件
if platform.system() != "Windows":
    # 参考PyTorch issue #973: https://github.com/pytorch/pytorch/issues/973
    # 导入资源管理模块
    import resource

    # 获取当前系统文件描述符限制
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # 获取当前软限制
    base_soft_limit = rlimit[0]
    # 获取硬限制
    hard_limit = rlimit[1]
    # 设置新的软限制:
    # - 不小于4096
    # - 不小于当前软限制
    # - 不超过硬限制
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    # 更新系统的文件描述符限制
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

#? 对象采样器注册表，加入@regeister_module装饰器注册，使用cfg配置文件访问
OBJECTSAMPLERS = Registry("Object sampler")

#? 用包装器包装数据集，用于数据增强
def build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset                                          # 从mmdet3d导入CBGS数据集包装器
    from mmdet.datasets.dataset_wrappers import ClassBalancedDataset, ConcatDataset, RepeatDataset     # 从mmdet导入类平衡、连接和重复数据集包装器

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "ConcatDataset": # 连接多个数据集，允许分别评估
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    elif cfg["type"] == "RepeatDataset": # 重复数据一定次数，用于数据增强
        dataset = RepeatDataset(build_dataset(cfg["dataset"], default_args), cfg["times"])
    elif cfg["type"] == "ClassBalancedDataset": # 对少于阈值的类进行过采样
        dataset = ClassBalancedDataset(
            build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"]
        )
    elif cfg["type"] == "CBGSDataset": # 使用CBGS数据集包装器包装数据集
        dataset = CBGSDataset(build_dataset(cfg["dataset"], default_args))
    elif isinstance(cfg.get("ann_file"), (list, tuple)): # 连接多个数据集，允许分别评估 单个pkl文件时，cfg.get("ann_file")为字符串
        dataset = _concat_dataset(cfg, default_args)
    else: # 使用build_from_cfg函数根据自定义配置文件构建数据集 cfg中参数会覆盖default_args中的同名参数
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
