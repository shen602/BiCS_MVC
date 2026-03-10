"""
BiCS-MVC Loss Functions Module
Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalContrastiveLoss(nn.Module):
    """双向对比学习损失函数"""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_list, labels=None):
        """
        计算双向对比损失
        Args:
            features_list: 多视图特征列表
            labels: 标签（可选）
        """
        if len(features_list) < 2:
            return torch.tensor(0.0, device=features_list[0].device if features_list else 'cpu')

        loss = 0
        num_views = len(features_list)
        valid_pairs = 0

        for i in range(num_views):
            for j in range(i + 1, num_views):
                feat1, feat2 = features_list[i], features_list[j]
                batch_size = feat1.shape[0]

                # 计算相似度矩阵
                sim_matrix = torch.matmul(feat1, feat2.T) / max(self.temperature, 0.1)
                labels_pos = torch.arange(batch_size, device=feat1.device)

                # 双向对比损失
                loss_i = F.cross_entropy(sim_matrix, labels_pos)
                loss_j = F.cross_entropy(sim_matrix.T, labels_pos)
                loss += (loss_i + loss_j) / 2
                valid_pairs += 1

        return loss / max(valid_pairs, 1)


class SemanticConsistencyLoss(nn.Module):
    """语义一致性损失函数"""

    def __init__(self):
        super().__init__()

    def forward(self, zs_list, labels):
        """
        计算语义一致性损失
        Args:
            zs_list: 多视图特征列表
            labels: 标签
        """
        if len(zs_list) < 2:
            return torch.tensor(0.0, device=zs_list[0].device if zs_list else 'cpu')

        loss = 0
        num_views = len(zs_list)
        valid_pairs = 0

        for i in range(num_views):
            for j in range(i + 1, num_views):
                z_i, z_j = zs_list[i], zs_list[j]

                # 归一化特征
                z_i_norm = F.normalize(z_i, dim=1)
                z_j_norm = F.normalize(z_j, dim=1)

                # 计算余弦相似度
                cos_sim = torch.sum(z_i_norm * z_j_norm, dim=1)

                # 一致性损失
                consistency_loss = torch.mean((1 - cos_sim) ** 2)
                loss += consistency_loss
                valid_pairs += 1

        return loss / max(valid_pairs, 1)
