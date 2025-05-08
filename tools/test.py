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
from pprint import pprint 
from thop import profile, clever_format

#! 加载部分权重
def load_partial_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # 获取模型当前的state_dict
    model_state_dict = model.state_dict()
    
    # 只加载非heads部分的权重
    for k in model_state_dict.keys():
        if not k.startswith('heads.'):
            if k in state_dict:
                model_state_dict[k] = state_dict[k]
    
    # 加载更新后的权重
    model.load_state_dict(model_state_dict, strict=False)

    # 加载部分权重后的验证
    verify_loading(model, checkpoint)

    return checkpoint

#! 加载部分权重后的验证
def verify_loading(model, checkpoint):
    loaded_keys = set(checkpoint['state_dict'].keys())
    model_keys = set(model.state_dict().keys())
    
    print("=== 未加载的权重 ===")
    print([k for k in model_keys if k.startswith('heads.')])
    
    print("\n=== 已加载的权重 ===")
    print([k for k in model_keys if not k.startswith('heads.')])

#! 计算模型复杂度(目前没有通过， 因 thop 在分析模型复杂度时无法正确处理自定义的 CUDA 算子， 如体素化)
def get_model_complexity_info_multimodal(model, data_loader):
    model.eval()
    # 获取一个batch的数据
    # 创建一个新的前向传播函数的类
    class WrapModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            return self.model(return_loss=False, rescale=True, **x)
    wrapped_model = WrapModel(model)
    data_batch = next(iter(data_loader))
    data = {}
    data['img'] = data_batch['img']._data[0]
    data['points'] = data_batch['points']._data[0]
    data['camera2ego'] = data_batch['camera2ego']._data[0]
    data['lidar2ego'] = data_batch['lidar2ego']._data[0]
    data['lidar2camera'] = data_batch['lidar2camera']._data[0]
    data['lidar2image'] = data_batch['lidar2image']._data[0]
    data['camera_intrinsics'] = data_batch['camera_intrinsics']._data[0]
    data['camera2lidar'] = data_batch['camera2lidar']._data[0]
    data['img_aug_matrix'] = data_batch['img_aug_matrix']._data[0]
    data['lidar_aug_matrix'] = data_batch['lidar_aug_matrix']._data[0]
    data['metas'] = data_batch['metas']._data[0]
    with torch.no_grad():
        flops, params = profile(wrapped_model, 
                              inputs=(data,),
                              verbose=True)
        # 转换为易读格式
        flops, params = clever_format([flops, params], "%.3f")
        print('-' * 50)
        print(f'模型总计算量: {flops}')
        print(f'模型总参数量: {params}')
        print('-' * 50)
    return flops, params

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    # 不带-- 这些是必需的位置参数 必须按照定义的顺序提供 在命令行中直接输入值，不需要参数名
    parser.add_argument("config", help="test config file path")  # 测试配置文件路径
    parser.add_argument("checkpoint", help="checkpoint file")  # 检查点文件
    # ---------------------------------
    parser.add_argument("--out", help="output result file in pickle format")  # 输出结果文件(pickle格式)
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )  # 仅格式化输出结果而不进行评估,用于将结果格式化为特定格式并提交到测试服务器
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )  # 评估指标,取决于数据集,如COCO的"bbox","segm","proposal",PASCAL VOC的"mAP","recall"
    parser.add_argument("--show", action="store_true", help="show results")  # 显示结果 #? 暂时没用
    parser.add_argument("--show-dir", help="directory where results will be saved")  # 结果保存目录 #? 暂时没用
    # ---------------------------------
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )  # 是否融合卷积和BN层,可以略微提高推理速度
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )  # 是否使用GPU收集结果
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )  # 用于从多个worker收集结果的临时目录,在未指定gpu-collect时可用
    parser.add_argument("--seed", type=int, default=0, help="random seed")  # 随机种子
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )  # 是否为CUDNN后端设置确定性选项
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )  # 覆盖配置文件中的某些设置,以xxx=yyy格式的键值对合并到配置文件中
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )  # 用于评估的自定义选项(已弃用),改用--eval-options
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )  # 用于评估的自定义选项,以xxx=yyy格式的键值对作为dataset.evaluate()函数的kwargs
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )  # 任务启动器,可选none/pytorch/slurm/mpi
    
    parser.add_argument("--local_rank", type=int, default=0) # 添加本地进程的rank参数,用于分布式训练
    args = parser.parse_args() # 解析命令行参数
    if "LOCAL_RANK" not in os.environ: # 如果环境变量中没有LOCAL_RANK,则将命令行参数中的local_rank设置为环境变量
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()  # 解析命令行参数
    print("================")
    print("Parsed arguments:")
    pprint(vars(args))  # 打印所有解析后的参数
    print("================")
    dist.init()  # 初始化分布式环境

    torch.backends.cudnn.benchmark = True  # 启用cudnn基准测试以提高性能
    torch.cuda.set_device(dist.local_rank())  # 设置当前进程使用的GPU设备

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )  # 确保至少指定了一个操作(保存/评估/格式化/显示结果)

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")  # eval和format_only不能同时指定

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")  # 输出文件必须是pkl格式

    #TODO 以上是args解析,以下是配置文件加载
    #? 加载配置文件 按目录递归继承configs的配置 默认每级(包括本级)目录是default.yaml 且同一键值对以最内层为准
    configs.load(args.config, recursive=True) 
    cfg = Config(recursive_eval(configs), filename=args.config)  # 创建配置对象
    pprint(cfg)  # 打印配置信息
    # cfg.dump('/root/bs/__tmp__/configs.yaml') #? 这样把整个配置保存后方便查看和更改

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)  # 合并命令行指定的配置选项(覆盖)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):   # cfg中默认是False
        torch.backends.cudnn.benchmark = True  # 根据配置设置cudnn_benchmark

    cfg.model.pretrained = None  # 禁用预训练模型
    # in case the test dataset is concatenated
    samples_per_gpu = 1  # 设置每个GPU的样本数
    if isinstance(cfg.data.test, dict):  # 一般单个数据集
        cfg.data.test.test_mode = True  # 设置测试模式 默认是True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)  # 获取每个GPU的样本数
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle' # 默认已是DefaultFormatBundle
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)  # 替换数据处理管道
    elif isinstance(cfg.data.test, list):  # 多个数据集
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True  # 设置每个数据集为测试模式
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )  # 获取最大的每GPU样本数
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)  # 替换所有数据集的处理管道

    # init distributed env first, since logger depends on the dist info.
    distributed = False  # 设置分布式环境标志  #! 是否分布式 默认True  测试时设置为False
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)  # 设置随机种子以确保可重复性

    #TODO 以上是配置文件加载,以下是数据集和数据加载器构建
    # build the dataloader
    # 构建数据集和数据加载器
    dataset = build_dataset(cfg.data.test) # builder -> GBGSDataset -> NuScenesDataset(Custom3D)  pipeline在get_item中使用   此时test_mode=True
    print("================")
    print(f'test dataset length: {len(dataset)}')
    print("================")
    # 封装pytorch的DataLoader而成，这里batch_size = num_gpus * samples_per_gpu
    #? 因为dataset的getitem返回的是字典，因而dataloader迭代时返回的也是字典，key保持不变，value变为批此列表的形式
    print("================")
    print(f'samples_per_gpu: {samples_per_gpu}')
    print("================")
    data_loader = build_dataloader( 
        dataset,
        samples_per_gpu=samples_per_gpu, 
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed, 
        shuffle=False,
    )

    #TODO 以上是数据集和数据加载器构建,以下是模型构建和加载检查点
    # build the model and load checkpoint
    # 构建模型并加载检查点
    cfg.model.train_cfg = None
    model = build_model(cfg.model) #! 没用到test_cfg
    # model.init_weights()
    # model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")) 

    # pprint(model) 
    fp16_cfg = cfg.get("fp16", None) 
    if fp16_cfg is not None:
        wrap_fp16_model(model)  # 使用FP16包装模型
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")  # 从CPU加载检查点(默认不严格匹配),之后根据需要移动
    # checkpoint = load_partial_checkpoint(model, args.checkpoint) # 加载部分权重并查看
    if args.fuse_conv_bn: # 参数指定的话，可以略微加速
        model = fuse_conv_bn(model)  # 融合卷积和BN层

    # 旧版本检查点中没有保存类别信息,这是为了向后兼容的解决方案
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
        print('============')
        print("from checkpoint")
        print(model.CLASSES)
        print('============')
    else:
        model.CLASSES = dataset.CLASSES
        print('============')
        print("from dataset")
        print(model.CLASSES)
        print('============')

    # 计算模型复杂度
    # flops, params = get_model_complexity_info_multimodal(model, data_loader) 
    #TODO 以上是模型构建和加载检查点,以下是测试
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])  # 单GPU模式
        outputs = single_gpu_test(model, data_loader) # mmdet3d.apis
    else:
        model = MMDistributedDataParallel(  # 多GPU分布式模式
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect) # mmdet.apis

    rank, _ = get_dist_info()  # 获取分布式训练的rank信息
    if rank == 0:  # 主进程执行以下操作
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)  # 保存结果到文件
        kwargs = {} if args.eval_options is None else args.eval_options #? eval_options传入
        if args.format_only:
            dataset.format_results(outputs, **kwargs)  # 仅格式化结果
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy() 
            # hard-code way to remove EvalHook args
            # 硬编码方式移除EvalHook参数
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            # 将args.eval和args.eval_options更新到eval_kwargs中
            # 连同原本的eval_kwargs（剔除EvalHook参数）
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))  # 评估模型性能


if __name__ == "__main__":
    main()
