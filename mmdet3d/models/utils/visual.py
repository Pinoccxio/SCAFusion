import numpy as np
import cv2
import torch


def visualize_bev_feature(feature, save_path, map_type=cv2.COLORMAP_HOT):
    """
    使用cv2可视化BEV特征图并保存
    
    参数:
        feature: torch.Tensor, 形状为(B,C,H,W)的特征图
        save_path: str, 图像保存路径
        map_type: HOT/JET/AUTUMN/INFERNO...
    """
    print(f'特征形状为：{feature.shape}')
    feature = feature[0]  # (C,H,W)
    feature = feature.detach().cpu().numpy()
    # 计算特征图的平均值来降维
    feature_vis = np.mean(feature, axis=0)  # (H,W)
    
    # 归一化到0-255范围
    feature_min = feature_vis.min()
    feature_max = feature_vis.max()
    if feature_max - feature_min != 0:
        feature_vis = ((feature_vis - feature_min) / (feature_max - feature_min) * 255).astype(np.uint8)
    else:
        feature_vis = np.zeros_like(feature_vis, dtype=np.uint8)
    
    # 应用热力图颜色映射
    feature_vis = cv2.applyColorMap(feature_vis, map_type)
    
    # 保存图像
    cv2.imwrite(save_path, feature_vis)