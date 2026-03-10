"""
BiCS-MVC Configuration Module
Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering
"""
import torch
import os


class BiCSMVCConfig:
    """BiCS-MVC模型配置类"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_dir = 'models'

    @classmethod
    def get_dataset_config(cls, dataset_name):
        """获取数据集特定配置"""
        optimal_config = {
            'feature_dim': 512,
            'high_dim': 128,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'total_epochs': 50,
            'contrastive_weight': 1.0,
            'semantic_weight': 0.3,
            'temperature': 0.5,
            'use_projector': True
        }

        datasets = ['NUSWIDE', 'MNIST_USPS', 'Fashion', 'Hdigit', 'Digit-Product']
        configs = {dataset: optimal_config.copy() for dataset in datasets}

        return configs.get(dataset_name, optimal_config)


# 数据集配置
DATASET_CONFIGS = {
    'NUSWIDE': {
        'path': 'data/NUSWIDE.mat',
        'view': 5,
        'expected_classes': 5
    },
    'MNIST_USPS': {
        'path': 'data/MNIST_USPS.mat',
        'view': 2,
        'expected_classes': 10
    },
    'Fashion': {
        'path': 'data/Fashion.mat',
        'view': 3,
        'expected_classes': 10
    },
    'Hdigit': {
        'path': 'data/Hdigit.mat',
        'view': 2,
        'expected_classes': 10
    },
    'Digit-Product': {
        'path': 'data/Digit-Product.mat',
        'view': 2,
        'expected_classes': 10
    }
}


# 创建模型保存目录
if not os.path.exists(BiCSMVCConfig.model_save_dir):
    os.makedirs(BiCSMVCConfig.model_save_dir)
