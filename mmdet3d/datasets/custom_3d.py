import tempfile                                      # 导入临时文件模块
from os import path as osp                           # 导入路径处理模块

import mmcv                                          # 导入mmcv工具库
import numpy as np                                   # 导入numpy数值计算库
from torch.utils.data import Dataset                 # 导入PyTorch数据集基类

from mmdet.datasets import DATASETS                  # 导入数据集注册表

from ..core.bbox import get_box_type                # 导入3D边界框类型获取函数 core/bbox/structures/utils.py
from .pipelines import Compose                      # 导入数据处理流水线组合器 根本来源 from mmdet.datasets.pipelines import Compose
from .utils import extract_result_dict              # 导入结果字典提取工具

#TODO 此类作为NuScenesDataset的父类，主要是init/pipeline/get_item被使用了
@DATASETS.register_module()
class Custom3DDataset(Dataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    Args:
        dataset_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    #? init在子类用使用了
    def __init__(
        self,
        dataset_root,      # 数据集根目录路径
        ann_file,          # 标注文件路径
        pipeline=None,     # 数据处理流水线,默认为None
        classes=None,      # 数据集类别,默认为None
        modality=None,     # 传感器模态,默认为None
        box_type_3d="LiDAR", # 3D边界框类型,默认为"LiDAR"
        filter_empty_gt=True, # 是否过滤空的真值标注,默认为True
        test_mode=False,   # 是否为测试模式,默认为False
    ):
        super().__init__()
        # 初始化基本属性
        self.dataset_root = dataset_root  # 保存数据集根目录
        self.ann_file = ann_file          # 保存标注文件路径
        self.test_mode = test_mode        # 保存测试模式标志
        self.modality = modality          # 保存传感器模态
        self.filter_empty_gt = filter_empty_gt  # 保存是否过滤空真值标注
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)  # 获取3D框类型和模式

        # 初始化类别相关信息
        self.CLASSES = self.get_classes(classes)  # 获取数据集类别
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}  # 构建类别到ID的映射
        self.data_infos = self.load_annotations(self.ann_file)  # 加载标注信息
        
        # 构建数据处理流水线
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # 设置数据采样器的分组标志 非测试模式下设置
        if not self.test_mode:
            self._set_group_flag()

        self.epoch = -1  # 初始化训练轮次为-1
    
    #? 与pipeline配合使用,针对不同的epoch，数据增强的策略不同
    def set_epoch(self, epoch):   # 这里调用
        self.epoch = epoch
        if hasattr(self, "pipeline"): # 类中有某种属性/方法等
            for transform in self.pipeline.transforms:
                if hasattr(transform, "set_epoch"): 
                    transform.set_epoch(epoch) # 这个set_epoch是数据增强中的方法而非数据集中的方法
   
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        return mmcv.load(ann_file)
  
    def get_data_info(self, index):
        """获取指定索引的数据信息

        Args:
            index (int): 要获取的样本数据的索引

        Returns:
            dict: 将传递给数据预处理流水线的数据信息,包含以下键值:

                - sample_idx (str): 样本索引
                - lidar_path (str): 点云文件路径
                - file_name (str): 点云文件名
                - ann_info (dict): 标注信息
        """
        # 获取指定索引的数据信息
        info = self.data_infos[index]
        # 获取激光雷达点云的索引
        sample_idx = info["point_cloud"]["lidar_idx"]
        # 构建激光雷达点云文件的完整路径
        lidar_path = osp.join(self.dataset_root, info["pts_path"])

        # 构建输入字典,包含点云路径、样本索引和文件名
        input_dict = dict(
            lidar_path=lidar_path, sample_idx=sample_idx, file_name=lidar_path
        )

        # 如果不是测试模式,则加载标注信息
        if not self.test_mode:
            # 获取标注信息
            annos = self.get_ann_info(index)
            # 将标注信息添加到输入字典
            input_dict["ann_info"] = annos
            # 如果需要过滤空的真值标注,且当前样本没有有效标注,则返回None
            if self.filter_empty_gt and ~(annos["gt_labels_3d"] != -1).any():
                return None
        return input_dict

    #? 在prepare_train/test_data中调用
    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.
            results = dict(
                token=info["token"],
                sample_idx=info['token'],
                lidar_path=info["lidar_path"],
                sweeps=info["sweeps"],
                timestamp=info["timestamp"],
                location=info["location"],
                # image_paths
                # lidar2camera
                # lidar2image
                # lidar2ego
                # camera2ego
                # camera_intrinsics
                # camera2lidar
                # ann_info dict(
                            # gt_bboxes_3d=gt_bboxes_3d,
                            # gt_labels_3d=gt_labels_3d,
                            # gt_names=gt_names_3d,
                            # )
            )

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []                    # 图像相关字段列表
        results["bbox3d_fields"] = []                 # 3D边界框相关字段列表
        results["pts_mask_fields"] = []               # 点云掩码相关字段列表
        results["pts_seg_fields"] = []                # 点云分割相关字段列表
        results["bbox_fields"] = []                   # 2D边界框相关字段列表
        results["mask_fields"] = []                   # 2D掩码相关字段列表
        results["seg_fields"] = []                    # 2D分割相关字段列表
        results["box_type_3d"] = self.box_type_3d    # 3D边界框类型
        results["box_mode_3d"] = self.box_mode_3d    # 3D边界框模式(坐标系类型)
    
    #? 在__getitem__()中调用
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict) # 给input_dict添加一些key
        example = self.pipeline(input_dict)
        # 如果需要过滤空的真值标注,且当前样本没有任何有效标注,则返回None
        if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
            return None
        return example

    #? 在__getitem__()中调用
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results, \
                tmp_dir is the temporal directory created for saving json \
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, "results")
            out = f"{pklfile_prefix}.pkl"
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, "data loading pipeline is not provided"
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    #? 在__getitem__()中调用
    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    #? 调用了prepare_test_data和prepare_train_data
    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode: # 只有test的时候进入此路
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None: # 如果数据为空，则重新随机选择一个样本
                idx = self._rand_another(idx)
                continue
            return data

    #? 与__rand_another()方法耦合
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
