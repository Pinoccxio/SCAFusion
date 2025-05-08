import mmcv
import torch
import time

'''
#? pipeline处理前:
    dict(
        token=info["token"],
        sample_idx=info['token'],
        lidar_path=info["lidar_path"],
        sweeps=info["sweeps"],
        timestamp=info["timestamp"],
        location=info["location"],
        # image_paths
        # lidar2camera
        # lidar2image
        # camera2ego
        # camera_intrinsics
        # camera2lidar
        # ann_info dict(
                    # gt_bboxes_3d=gt_bboxes_3d,
                    # gt_labels_3d=gt_labels_3d,
                    # gt_names=gt_names_3d,
                    # )
#? pipeline处理后:
    dict(
        'img',  DataContainer._data[0]为tensor,尺寸[batch_size,camera_num,3,H,W]
        'points',  DataContainer._data[0]是一个len为batch_size的列表,每个元素为[num_points,5] x y z intensity timestamp
        'gt_bboxes_3d', DataContainer._data[0]是一个len为batch_size的列表,每个元素对应一个LiDARInstance3DBoxes对象
        'gt_labels_3d', DataContainer._data[0]是一个len为batch_size的列表,每个元素为num_boxes大小的tensor
        'gt_masks_bev', 为tensor [batch_size,map_classes,H,W]
        'camera_intrinsics',  DataContainer._data[0]为tensor,尺寸[batch_size,camera_num,4,4]
        'camera2ego',  DataContainer._data[0]为tensor,尺寸[batch_size,camera_num,4,4]
        'lidar2ego',  DataContainer._data[0]为tensor,尺寸[batch_size,4,4]
        'lidar2camera', DataContainer._data[0]为tensor,尺寸[batch_size,camera_num,4,4]
        'camera2lidar', DataContainer._data[0]为tensor,尺寸[batch_size,camera_num,4,4]
        'lidar2image', DataContainer._data[0]为tensor,尺寸[batch_size,camera_num,4,4]
        'img_aug_matrix', DataContainer._data[0]为tensor,尺寸[batch_size,camera_num,4,4]
        'lidar_aug_matrix', DataContainer._data[0]为tensor,尺寸[batch_size,4,4]
        'metas' DataContainer._data[0]是一个len为batch_size的列表,每个元素为dict,内涵一些元信息
    )
'''
#! 新增，用以测试经过pipeline处理后的数据结构
def test_data_structure(data):
    print("================")
    print('img')
    print(data['img']._data[0].shape)
    print("================")
    print('points')
    print(len(data['points']._data[0]))
    print(data['points']._data[0][0].shape)
    print("================")
    print('gt_bboxes_3d')
    print(len(data['gt_bboxes_3d']._data[0]))
    print(len(data['gt_bboxes_3d']._data[0][0])) #23
    print("================")
    print('gt_labels_3d')
    print(len(data['gt_labels_3d']._data[0]))
    print(len(data['gt_labels_3d']._data[0][0])) #23
    print("================")
    print('gt_masks_bev')
    print(data['gt_masks_bev'].shape)

    print("================") 
    print('camera_intrinsics')
    print(data['camera_intrinsics']._data[0].shape)
    print("================")
    print('camera2ego')
    print(data['camera2ego']._data[0].shape)
    print("================")
    print('lidar2ego')
    print(data['lidar2ego']._data[0].shape)
    print("================")
    print('lidar2camera')
    print(data['lidar2camera']._data[0].shape)
    print("================")
    print('camera2lidar')
    print(data['camera2lidar']._data[0].shape)
    print("================")
    print('lidar2image')
    print(data['lidar2image']._data[0].shape)
    print("================")
    print('img_aug_matrix')
    print(data['img_aug_matrix']._data[0].shape)
    print("================")
    print('lidar_aug_matrix')
    print(data['lidar_aug_matrix']._data[0].shape)
    print("================")
    print('metas')
    print(len(data['metas']._data[0]))
    
def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    total_time = 0
    total_samples = 0
    
    for data in data_loader:
        start_time = time.time()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        end_time = time.time()
        
        batch_time = end_time - start_time
        batch_size = len(result)
        total_time += batch_time
        total_samples += batch_size
        
        # 打印每个batch的平均时间
        print(f' | Batch size: {batch_size}, Average time per sample: {batch_time/batch_size:.4f}s')
        
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    
    # 打印总体平均时间
    avg_time = total_time / total_samples
    print(f'\nTotal samples: {total_samples}')
    print(f'Average time per sample: {avg_time:.4f}s')
    print(f'FPS: {1/avg_time:.2f}')
    
    return results
