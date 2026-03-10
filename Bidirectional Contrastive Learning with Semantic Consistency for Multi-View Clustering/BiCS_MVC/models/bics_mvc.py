"""
BiCS-MVC Main Model
Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.losses import BidirectionalContrastiveLoss, SemanticConsistencyLoss


class BiCSMVC(nn.Module):
    """
    BiCS-MVC: Bidirectional Contrastive Learning with Semantic Consistency
    for Multi-View Clustering
    """

    def __init__(self, view_dims, feature_dim=512, high_dim=128, class_num=10,
                 device='cuda', dataset_name='NUSWIDE', config=None):
        super().__init__()
        self.view_dims = view_dims
        self.num_views = len(view_dims)

        from config.config import BiCSMVCConfig
        dataset_config = BiCSMVCConfig.get_dataset_config(dataset_name)

        if config:
            dataset_config.update(config)

        self.feature_dim = dataset_config['feature_dim']
        self.high_dim = dataset_config['high_dim']
        self.class_num = class_num
        self.device = device
        self.dataset_name = dataset_name

        # 配置是否使用投影器
        self.use_projector = dataset_config.get('use_projector', True)
        print(f"  Use Projector: {self.use_projector}")

        # 权重配置
        self.contrastive_weight = dataset_config['contrastive_weight']
        self.semantic_weight = dataset_config['semantic_weight']

        # 视图编码器
        self.encoders = nn.ModuleList([
            self._create_encoder(dim, self.feature_dim) for dim in view_dims
        ])

        # 投影器（可选）
        if self.use_projector:
            self.projectors = nn.ModuleList([
                self._create_projector(self.feature_dim, self.high_dim)
                for _ in range(self.num_views)
            ])
        else:
            self.projectors = None

        # 损失函数
        self.contrastive_loss = BidirectionalContrastiveLoss(
            temperature=dataset_config['temperature']
        )
        self.semantic_loss = SemanticConsistencyLoss()

        print(f"\nBiCS-MVC Model for {dataset_name}:")
        print(f"  Feature dim: {self.feature_dim}, Projection dim: {self.high_dim}")
        print(f"  Loss weights: contrastive={self.contrastive_weight}, "
              f"semantic={self.semantic_weight}")
        print(f"  Temperature: {dataset_config['temperature']}")

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _create_encoder(self, input_dim, feature_dim):
        """创建视图编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

    def _create_projector(self, feature_dim, high_dim):
        """创建投影器"""
        return nn.Sequential(
            nn.Linear(feature_dim, high_dim),
            nn.BatchNorm1d(high_dim),
            nn.ReLU()
        )

    def forward(self, xs, missing_mask=None):
        """
        前向传播
        Args:
            xs: 多视图输入数据列表
            missing_mask: 缺失掩码（可选）
        Returns:
            包含特征和其他信息的字典
        """
        enhanced_views = xs
        confidence_scores = torch.ones(
            xs[0].shape[0], self.num_views, device=xs[0].device
        )

        zs, hs = [], []

        for v in range(self.num_views):
            # 编码
            h = self.encoders[v](enhanced_views[v])

            # 投影（可选）
            if self.use_projector:
                z = F.normalize(self.projectors[v](h), dim=1)
            else:
                z = F.normalize(h, dim=1)

            zs.append(z)
            hs.append(h)

        return {
            'zs': zs,
            'hs': hs,
            'enhanced_views': enhanced_views,
            'confidence_scores': confidence_scores,
        }

    def compute_loss(self, xs, labels, outputs, lambda_dict=None):
        """
        计算总损失
        Args:
            xs: 输入数据
            labels: 标签
            outputs: 模型输出
            lambda_dict: 损失权重字典（可选）
        Returns:
            总损失和损失字典
        """
        base_lambda_dict = {
            'contrastive': self.contrastive_weight,
            'semantic': self.semantic_weight,
        }

        if lambda_dict is not None:
            for key in base_lambda_dict:
                if key in lambda_dict:
                    base_lambda_dict[key] = base_lambda_dict[key] * lambda_dict[key]

        zs = outputs['zs']

        # 计算各项损失
        loss_contrastive = self.contrastive_loss(zs, labels)
        loss_semantic = self.semantic_loss(zs, labels)

        # 检查损失值有效性
        losses = [loss_contrastive, loss_semantic]
        for i, loss in enumerate(losses):
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss {i}, setting to 0")
                losses[i] = torch.tensor(0.0, device=self.device)

        # 总损失
        total_loss = (
            base_lambda_dict['contrastive'] * losses[0] +
            base_lambda_dict['semantic'] * losses[1]
        )

        loss_dict = {
            'total': total_loss,
            'contrastive': losses[0],
            'semantic': losses[1],
        }

        return total_loss, loss_dict
