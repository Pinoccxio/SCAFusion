# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function  # 自定义自动求导函数
from torch.nn.modules.utils import _pair

from .voxel_layer import dynamic_voxelize, hard_voxelize


class _Voxelization(Function):
    @staticmethod  #? 其中ctx是上下文, 可以保存一些信息 可用ctx.save_for_backward()保存信息 用ctx.saved_tensors获取信息
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create. #? 这个参数可能会drop points
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1: #? 这一路是动态voxelize，暂时不管
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
            return coors
        else:
            voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1))) # (max_voxels, max_points, ndim)
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int) # (max_voxels, 3)
            num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int) # [max_voxels]
            voxel_num = hard_voxelize(  # max_voxels采用先来先服务策略，因而输入的点云需要shuffle
                points,
                voxels,
                coors,
                num_points_per_voxel,
                voxel_size,
                coors_range,
                max_points,
                max_voxels,
                3, # 用以确定xyz维度和索引
                deterministic,
            )
            # select the valid voxels
            voxels_out = voxels[:voxel_num] # [M, max_points, ndim]
            coors_out = coors[:voxel_num] # [M, 3]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num] # [M]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply # 转换为可直接使用的函数，相当于封装了forward()

class Voxelization(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time  #? 第一个元素是训练时，第二个是测试时
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels) # 如果输入int 则返回(int, int)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size  # num_x num_y num_z
        grid_size = torch.round(grid_size).long() # 取整数
        input_feat_shape = grid_size[:2] # 取前两个
        self.grid_size = grid_size  #? forward中没用到
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w] removed
        self.pcd_shape = [*input_feat_shape, 1] #[::-1] # [num_x, num_y, 1]  #? forward中没用到

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(
            input,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", max_num_points=" + str(self.max_num_points)
        tmpstr += ", max_voxels=" + str(self.max_voxels)
        tmpstr += ", deterministic=" + str(self.deterministic)
        tmpstr += ")"
        return tmpstr
