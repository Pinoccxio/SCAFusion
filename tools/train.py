import argparse                                                   # 命令行参数解析
import copy                                                       # 深拷贝
import os                                                        # 操作系统接口
import random                                                    # 随机数生成
import time                                                      # 时间相关功能

import numpy as np                                               # 科学计算库
import torch                                                     # PyTorch深度学习框架
from mmcv import Config                                          # MMCV配置类
from torchpack import distributed as dist                        # 分布式训练工具
from torchpack.environ import auto_set_run_dir, set_run_dir     # 运行目录设置
from torchpack.utils.config import configs                       # 配置工具

from mmdet3d.apis import train_model                            # 训练模型API
from mmdet3d.datasets import build_dataset                      # 数据集构建
from mmdet3d.models import build_model                          # 模型构建
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval  # 工具函数

def check_camera_backbone_weights(model):
    w = model.encoders.camera.backbone.named_parameters()
    print("=== Camera Backbone Weights ===")
    for name, param in w:
        print(f"Layer: {name}")
        print(f"Shape: {param.shape}")
        print(f"Values: {param.data}")
        print("---")

def main():
    dist.init()
    #TODO 包括一些指定参数和额外参数(如--load_from  --model.encoders.camera.backbone.init_cfg.checkpoint等)
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args() 
    # opts 包含其他未明确定义但以 --key=value 形式传入的参数
    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir() # 自动生成  运行程序时所在的工作目录/runs/年月日-时分秒 的文件夹  
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir # 通过args指定cfg的run_dir
    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}") # 是mmcv.Config配置对象的一个特有属性，会将配置信息转换成格式化的文本

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True # 使用确定的cuda操作选择
            torch.backends.cudnn.benchmark = False # 关闭cudnn的自动优化x选择

    #TODO 以上是基本配置，以下是数据和模型相关

    datasets = [build_dataset(cfg.data.train)]
    print("========")
    print(f'train dataset length: {len(datasets[0])}')
    print("========")

    model = build_model(cfg.model) # 用cfg.model配置构建模型
    # model.init_weights() # 初始化encoders.camera.backbone权重  #! 注销 因为在模型init中已经调用

    if cfg.get("sync_bn", None):  #? 暂无sync_bn配置
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model( #* 内部会有load_from和resume_from
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )



if __name__ == "__main__":
    main()
