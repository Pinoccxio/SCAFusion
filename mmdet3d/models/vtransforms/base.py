from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]


def gen_dx_bx(xbound, ybound, zbound): # [min max step]
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) # [xstep ystep zstep]
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]) # [xmin+xstep/2 ...]
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    ) # [xnum ynum znum]  [360,360,1]
    return dx, bx, nx


class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int, # 256
        out_channels: int, # 80
        image_size: Tuple[int, int], # 256 704
        feature_size: Tuple[int, int], # 32 88
        xbound: Tuple[float, float, float], # [-54 54 0.3]
        ybound: Tuple[float, float, float], # [-54 54 0.3]
        zbound: Tuple[float, float, float], # [-10 10 20]
        dbound: Tuple[float, float, float], # [1 60 0.5]
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum() # frustum.shape = (D,fH,fW,3) (118,32,88,3)
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    
    @force_fp32()
    def create_frustum(self): 
        #? 相机前的frustum离散空间的任意一点，对应的增强后的图像的[u,v,d(Zc)]
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        ) # 在每个fH层和fW层都有相同的深度值
        D, _, _ = ds.shape # D 深度采样点数量

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        ) # 在每个深度层和每个fH层都有相同的x坐标
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        ) # 在每个深度层和每个fW层都有相同的y坐标

        frustum = torch.stack((xs, ys, ds), -1)  # frustum.shape = (D,fH,fW,3)
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots, # (B,N,3,3)
        camera2lidar_trans, # (B,N,3)
        intrins, # (B,N,3,3)
        post_rots, # (B,N,3,3)
        post_trans, # (B,N,3)
        **kwargs,
    ):
        #? img_aug2img
        #? img2lidar
        #? lidar2lidar_aug
        B, N, _ = camera2lidar_trans.shape # (B,N,3)

        # undo post-transformation   #? frustum是aug后的
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3) # 广播运算
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        ) # (B,N,D,H,W,3)  (u*d, v*d, d)  (X Y Z)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3) # (B,N,D,H,W,3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
        #? 相机前方视锥离散空间任意一点，对应的增强后的雷达的点云坐标 (Xl,Yl,Zl)
        return points # (B,N,D,H,W,3) 

    # 在depth_lss子类中定义
    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long() # 转化为体素索引
        geom_feats = geom_feats.view(Nprime, 3) # 展开
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        ) # (Nprime,1)  内容 000...111......B-1
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # (Nprime,4)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept] # (num,80)
        geom_feats = geom_feats[kept] # (num,4)
        # (B,80,Z,H,W)
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix, # img2aug
        lidar_aug_matrix, # lidar2aug
        metas,
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # print(img.shape, self.image_size, self.feature_size)

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        ) #? (B,N,1,H,W) 每个值对应深度

        for b in range(batch_size): #? lidar2camera
            cur_coords = points[b][:, :3] # (P,3) P为点数
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :] # (N,P)
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] # (u,v) = (X/Z,Y/Z)

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2) # (N,P,2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]] # (x,y)2(y,x)

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            ) # (N,P)
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long() # (P,2)
                masked_dist = dist[c, on_img[c]] # (P,)
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        ) # (B,N,D,H,W,3)

        x = self.get_cam_feats(img, depth) #(B,N,118,H,W,80)
        x = self.bev_pool(geom, x) # (B,80,360,360)
        return x
