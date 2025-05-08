
#? Hook: 钩子是用于在训练过程中插入额外操作的函数。它们可以用于在训练的特定阶段执行某些操作，例如在每个epoch结束时执行评估，或在训练开始前进行初始化。
import torch
from mmcv.parallel import MMDistributedDataParallel # 可用于包装model,使其支持分布式运行
from mmcv.runner import (
    EpochBasedRunner,           # 基于epoch的训练运行器
    DistSamplerSeedHook,        # 分布式采样器种子钩子,用于设置随机种子确保可重复性 before_train_epoch  父1
    GradientCumulativeFp16OptimizerHook,  # FP16梯度累积优化器钩子 after_train_iter  父3
    Fp16OptimizerHook,         # FP16优化器钩子  after_train_iter     父2
    OptimizerHook,             # 优化器钩子基类  after_train_iter     父1
    build_optimizer,           # 构建优化器
    build_runner,              # 构建训练运行器
)
from mmdet3d.runner import CustomEpochBasedRunner  # 自定义的基于epoch的训练运行器  #? 通过cfg.runner调用了

from mmdet3d.utils import get_root_logger  # 获取根日志记录器
from mmdet.core import DistEvalHook        # 分布式评估钩子  after_train_epoch
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor  # 数据加载器、数据集构建和图像张量替换

#! 阶段学习率设置
def stage_lr_setting(cfg):
    """根据配置文件设置不同阶段的训练参数"""
    frozen_stages = cfg.model_freeze_config.frozen_stages
    if frozen_stages == -1:
        return cfg
    else:
        cfg.optimizer.lr = cfg.model_freeze_config.frozen_params[f'stage{frozen_stages}']['lr']
        return cfg
    
def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    #TODO 数据加载器构建
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset] # 要用一层[]包装
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            1,  # None, 不重要
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]
    
    #TODO 模型构建
    # put model on gpus
    # 当设置为 True 时，DDP（DistributedDataParallel）会搜索未参与梯度计算的参数
    # 这些参数在反向传播时会被跳过，不进行梯度同步
    # 一般模型中有条件分支或训练过程中某些层需要冻结时设置
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    model.freeze_params(cfg) #! 冻结训练
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False, # 是否在GPU间同步缓冲区（非参数张量）,这部分在syncbn中已通过其他手段同步
        find_unused_parameters=find_unused_parameters,
    )
    

    #TODO 初始化runner配置
    # build runner runner每个epoch运行一次train()

    cfg = stage_lr_setting(cfg) #! 阶段学习率设置
    optimizer = build_optimizer(model, cfg.optimizer) 
    runner = build_runner(  #? 用的是CustomEpochBasedRunner
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir, #? 没有在cfg中定义, 通过tools/train.py --run-dir指定 用以存储训练文件
            logger=logger,
            meta={},
        ),
    )
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    #TODO 进一步配置runner的train部分
    #* fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            # 如果使用梯度累积,则使用支持梯度累积的FP16优化器Hook
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            # 否则使用普通的FP16优化器Hook
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        # 分布式训练且未指定优化器类型时使用默认的OptimizerHook
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        # 其他情况直接使用配置文件中的优化器配置
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,           # 学习率调整配置
        optimizer_config,        # 优化器配置,包含FP16等设置 # 在optimizer处理前的进一步处理
        cfg.checkpoint_config,   # 检查点保存配置
        cfg.log_config,         # 日志配置
        cfg.get("momentum_config", None),  # 动量更新配置,默认为None
    )
    if isinstance(runner, EpochBasedRunner): # runner是CustomEpochBasedRunner也行
        runner.register_hook(DistSamplerSeedHook()) # 确保不同GPU采用不同数据，且设置随机种子，保证训练的可重复性

    #TODO 进一步配置runner的eval部分
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1) #? 默认为1, 即验证时，使用一个GPU 
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner" # 因为是EpochBasedRunner,所以基于epoch验证
        eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    #TODO 断点续训和加载检查点
    if cfg.resume_from:
        runner.resume(cfg.resume_from) #? 断点续训  要想状态恢复， max_epochs 需要与之前一致
    elif cfg.load_from:   
        runner.load_checkpoint(cfg.load_from) #? 只加载模型参数,不加载优化器参数    #TODO load_from

    runner.run(data_loaders, [("train", 1)]) # workflow部分 每个阶段训练1个epoch，且仅训练 
    #? epoch验证过程是通过 DistEvalHook 来执行的，而不是通过 workflow 这样更加灵活
