from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS

# 在configs层的default.yaml中 train.py中有调用
@RUNNERS.register_module()
class CustomEpochBasedRunner(EpochBasedRunner):
    def set_dataset(self, dataset):
        self._dataset = dataset


    def train(self, data_loader, **kwargs):
        # update the schedule for data augmentation
        for dataset in self._dataset:
            dataset.set_epoch(self.epoch) # 通知数据集当前的epoch,以便数据集在训练过程中使用正确的epoch
        super().train(data_loader, **kwargs)

#? run的过程
# for epoch in range(self._epoch, self._max_epochs):
#     for i, flow in enumerate(workflow):
#         mode, epochs = flow  # 解包模式和轮次
        
#         # 检查是否应该执行当前阶段
#         if epoch % sum(epoch_len for _, epoch_len in workflow) >= \
#             sum(epoch_len for _, epoch_len in workflow[:i]):
            
#             # 训练模式
#             if mode == 'train':
#                 self.train(data_loaders[0])
#                 self.logger.info(f'Epoch[{epoch}][{mode}] 完成训练')
            
#             # 验证模式
#             elif mode == 'val':
#                 self.val(data_loaders[1])
#                 self.logger.info(f'Epoch[{epoch}][{mode}] 完成验证')

# workflow = [('train', 2), ('val', 1), ('train', 1)]
# 执行顺序将会是：
# Epoch 0: train -> train -> val -> train
# Epoch 1: train -> train -> val -> train

#? train的过程
# def train(self, data_loader):
#     """单个epoch的训练流程"""
#     # 1. 设置模型为训练模式
#     self.model.train()
    
#     # 2. 如果是epoch based runner，通知数据集当前epoch
#     if hasattr(self, '_dataset'):
#         for dataset in self._dataset:
#             dataset.set_epoch(self.epoch)
    
#     # 3. 初始化进度条
#     runner_iter = IterLoader(data_loader)
#     batch_size = data_loader.batch_size
#     len_loader = len(data_loader)
#     self.call_hook('before_train_epoch')  # 调用epoch开始前的钩子
    
#     # 4. 遍历每个batch
#     for i, data_batch in enumerate(runner_iter):
#         # 4.1 调用迭代开始前的钩子
#         self.call_hook('before_train_iter')
        
#         # 4.2 执行训练步骤
#         outputs = self.model.train_step(  #* 此处使用了train_step
#             data_batch, 
#             self.optimizer,
#             **kwargs
#         )
        
#         # 4.3 如果是分布式训练，等待所有进程
#         if self.world_size > 1:
#             torch.cuda.synchronize()
            
#         # 4.4 更新训练指标
#         if outputs is not None:
#             self.outputs = outputs  #* 存储输出，供optimizer hook使用
#             self.log_buffer.update(outputs['log_vars'])
#             self.num_samples += outputs['num_samples']
            
#         # 4.5 调用迭代结束后的钩子（包括optimizer hook）
#         self.call_hook('after_train_iter')
        
#         # 4.6 更新进度条
#         self._inner_iter += 1
#         self.iter += 1
    
#     # 5. 调用epoch结束后的钩子
#     self.call_hook('after_train_epoch')
#     self._epoch += 1