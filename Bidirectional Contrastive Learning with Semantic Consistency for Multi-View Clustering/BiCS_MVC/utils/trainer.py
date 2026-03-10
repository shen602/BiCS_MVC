"""
BiCS-MVC Trainer Module
Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering
"""
import torch
import torch.optim as optim
from collections import defaultdict


class BiCSMVCTrainer:
    """BiCS-MVC训练器"""

    def __init__(self, model, train_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = device

        from config.config import BiCSMVCConfig
        dataset_config = BiCSMVCConfig.get_dataset_config(model.dataset_name)
        actual_lr = dataset_config['learning_rate']
        total_epochs = dataset_config['total_epochs']

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=actual_lr,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        if model.dataset_name in ['NUSWIDE', 'Hdigit']:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=actual_lr * 0.01
            )
        elif model.dataset_name in ['MNIST_USPS', 'Fashion', 'Digit-Product']:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[total_epochs // 2, total_epochs * 3 // 4],
                gamma=0.5
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=actual_lr * 0.01
            )

        self.loss_history = defaultdict(list)

    def train_epoch(self, epoch):
        """
        训练一个epoch
        Args:
            epoch: 当前epoch
        Returns:
            平均损失和损失字典
        """
        self.model.train()
        total_loss = 0.0
        loss_dict = defaultdict(float)
        num_batches = 0

        for batch_idx, (xs, labels, idx) in enumerate(self.train_loader):
            if len(xs) == 0:
                continue

            xs_device = [x.to(self.device) for x in xs]
            labels_device = labels.to(self.device)

            # 前向传播
            outputs = self.model(xs_device)
            loss, losses = self.model.compute_loss(xs_device, labels_device, outputs)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for key, value in losses.items():
                loss_dict[key] += value.item()

            num_batches += 1

        if num_batches == 0:
            print("Warning: No valid batches processed in this epoch")
            return 0, loss_dict

        # 更新学习率
        self.scheduler.step()

        # 计算平均损失
        avg_loss = total_loss / num_batches
        for key in loss_dict:
            loss_dict[key] = loss_dict[key] / num_batches
            self.loss_history[key].append(loss_dict[key])

        self.loss_history['total'].append(avg_loss)

        return avg_loss, loss_dict
