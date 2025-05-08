import os

import numpy as np
import torch

__all__ = ["load_augmented_point_cloud", "reduce_LiDAR_beams"]


def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    """加载增强的点云数据。
    
    Args:
        path (str): 点云文件路径
        virtual (bool): 是否加载虚拟点云
        reduce_beams (int): 激光雷达线数,默认32线
        
    Returns:
        points (np.ndarray): 增强后的点云数据
    """
    # 从文件加载原始点云数据,每个点包含5个值(x,y,z,intensity,ring)
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    
    # 构建虚拟点云文件路径
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points", 
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )
    assert os.path.exists(seg_path)
    
    # 加载虚拟点云数据
    data_dict = np.load(seg_path, allow_pickle=True).item()

    # 获取真实点云和虚拟点云
    virtual_points1 = data_dict["real_points"]  # 真实点云
    # 为虚拟点云添加反射强度值(0)
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],  # xyz坐标
            np.zeros([data_dict["virtual_points"].shape[0], 1]),  # 反射强度为0
            data_dict["virtual_points"][:, 3:],  # 其他特征
        ],
        axis=-1,
    )

    # 扩展原始点云维度以匹配虚拟点云
    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]), # +1 是标记
        ],
        axis=1,
    )
    # 为真实点云添加标记(0)
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )

    # 不同于Tianwei的实现,我们没有重复的前景真实点, 而是进行了语义信息的更新
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
        
    # 如果需要,添加虚拟点云
    if virtual:
        # 为虚拟点云添加标记(-1)
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        # 合并原始点云和虚拟点云
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
        
    return points


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    # print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius # 高度sin
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta) # 仰角
    phi = torch.atan2(pts[:, 1], pts[:, 0]) # 方位角

    top_ang = 0.1862 # rad 约为10.67度
    down_ang = -0.5353 # rad 约为-30.7度

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        #    以id为1为例
        #    beam_range[0] (≈0.1862°)
        #             ↓
        #             |←0.012° 偏移
        #     -----+-----  选择的点必须在
        #             |      这个范围内
        #     -----+-----
        #             |←0.012° 偏移
        #             ↓
        #     beam_range[1]
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    # [?] pick the 14th beam
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    else:
        raise NotImplementedError
    # points = copy.copy(pts)
    points = pts[mask]
    # print(points.size())
    return points.numpy()

