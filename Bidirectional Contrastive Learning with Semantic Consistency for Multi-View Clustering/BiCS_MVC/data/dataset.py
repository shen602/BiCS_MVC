"""
BiCS-MVC Dataset Module
Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering
"""
import torch
import numpy as np
import scipy.io
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os


class MultiViewDataset(Dataset):
    """多视图数据集加载器"""

    def __init__(self, path, view, dataset_name='NUSWIDE'):
        self.dataset_name = dataset_name
        self.view = view

        print(f"Loading {dataset_name} dataset from {path}...")
        self.load_data(path, dataset_name)
        print(f"Successfully loaded {dataset_name}: {self.data_size} samples, "
              f"{self.class_num} classes, {len(self.multi_view)} views")

    def load_data(self, path, dataset_name):
        """加载数据集"""
        try:
            if os.path.isfile(path) or (path.endswith('.mat') and not os.path.isdir(path)):
                file_path = path
            else:
                file_path = os.path.join(path, f'{dataset_name}.mat')

            if dataset_name == 'NUSWIDE':
                self.load_nuswide(file_path)
            elif dataset_name == 'MNIST_USPS':
                self.load_mnist_usps(file_path)
            elif dataset_name == 'Fashion':
                self.load_fashion(file_path)
            elif dataset_name == 'Hdigit':
                self.load_hdigit(file_path)
            elif dataset_name == 'Digit-Product':
                self.load_digit_product(file_path)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            raise

    def load_nuswide(self, path):
        """加载NUSWIDE数据集"""
        data = scipy.io.loadmat(path)
        self.multi_view = []
        self.labels = data['Y'].reshape(-1)
        unique_labels = np.unique(self.labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        self.dims = []
        self.class_num = len(unique_labels)

        for i in range(self.view):
            view_data = data['X' + str(i + 1)][:, :-1].astype(np.float32)
            scaler = MinMaxScaler()
            self.multi_view.append(scaler.fit_transform(view_data))
            self.dims.append(view_data.shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def load_mnist_usps(self, path):
        """加载MNIST_USPS数据集"""
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.multi_view = []
        X1 = data['X1'].reshape(data['X1'].shape[0], -1).astype(np.float32)
        X2 = data['X2'].reshape(data['X2'].shape[0], -1).astype(np.float32)
        self.labels = np.squeeze(data['Y']).astype(np.int32)
        unique_labels = np.unique(self.labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        self.dims = []
        self.class_num = len(unique_labels)
        self.multi_view.append(scaler.fit_transform(X1))
        self.multi_view.append(scaler.fit_transform(X2))
        self.dims.append(X1.shape[1])
        self.dims.append(X2.shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def load_fashion(self, path):
        """加载Fashion数据集"""
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.multi_view = []

        if 'Y' in data:
            labels = np.squeeze(data['Y'])
        elif 'gt' in data:
            labels = np.squeeze(data['gt'])
        elif 'truth' in data:
            labels = np.squeeze(data['truth'])
        else:
            raise ValueError("Fashion.mat缺少标签字段")

        self.labels = labels.astype(np.int32)
        unique_labels = np.unique(self.labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        self.dims = []
        self.class_num = len(unique_labels)

        if 'X1' in data and 'X2' in data and 'X3' in data:
            X1 = data['X1'].astype(np.float32)
            X2 = data['X2'].astype(np.float32)
            X3 = data['X3'].astype(np.float32)
        elif 'x1' in data and 'x2' in data and 'x3' in data:
            X1 = data['x1'].astype(np.float32)
            X2 = data['x2'].astype(np.float32)
            X3 = data['x3'].astype(np.float32)
        else:
            raise ValueError("Fashion.mat缺少视图数据")

        if X1.ndim > 2:
            X1 = X1.reshape(X1.shape[0], -1)
        if X2.ndim > 2:
            X2 = X2.reshape(X2.shape[0], -1)
        if X3.ndim > 2:
            X3 = X3.reshape(X3.shape[0], -1)

        self.multi_view.append(scaler.fit_transform(X1))
        self.multi_view.append(scaler.fit_transform(X2))
        self.multi_view.append(scaler.fit_transform(X3))
        self.dims.append(X1.shape[1])
        self.dims.append(X2.shape[1])
        self.dims.append(X3.shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def load_hdigit(self, path):
        """加载Hdigit数据集"""
        self.multi_view = []
        data = scipy.io.loadmat(path)
        self.labels = data['Y'].reshape(-1).astype(np.int32)
        v1 = data['X'][0, 0].astype(np.float32)
        v2 = data['X'][0, 1].astype(np.float32)
        scaler = MinMaxScaler()
        v1 = scaler.fit_transform(v1)
        v2 = scaler.fit_transform(v2)
        self.multi_view = [v1, v2]
        self.dims = [v1.shape[1], v2.shape[1]]
        unique_labels = np.unique(self.labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        self.class_num = len(unique_labels)
        self.data_size = self.multi_view[0].shape[0]

    def load_digit_product(self, path):
        """加载Digit-Product数据集"""
        self.multi_view = []
        data = scipy.io.loadmat(path)
        self.labels = data['Y'].reshape(-1).astype(np.int32)
        v1 = data['X'][0, 0].astype(np.float32)
        v2 = data['X'][0, 1].astype(np.float32)
        scaler = MinMaxScaler()
        v1 = scaler.fit_transform(v1)
        v2 = scaler.fit_transform(v2)
        self.multi_view = [v1, v2]
        self.dims = [v1.shape[1], v2.shape[1]]
        unique_labels = np.unique(self.labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        self.class_num = len(unique_labels)
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        views = [view[idx] for view in self.multi_view]
        label = self.labels[idx]
        return views, label, idx


def multiview_collate_fn(batch):
    """多视图数据批处理函数"""
    views_data = [[] for _ in range(len(batch[0][0]))]
    labels = []
    indices = []

    for sample in batch:
        views, label, idx = sample
        for v, view_data in enumerate(views):
            views_data[v].append(torch.tensor(view_data, dtype=torch.float32))
        labels.append(label)
        indices.append(idx)

    views_tensor = [torch.stack(view_list) for view_list in views_data]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    return views_tensor, labels_tensor, indices_tensor


def get_dataloader(dataset_name, batch_size=128):
    """获取数据加载器"""
    from config.config import DATASET_CONFIGS

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = DATASET_CONFIGS[dataset_name].copy()
    print(f"Loading dataset: {dataset_name}")

    try:
        dataset = MultiViewDataset(
            config['path'], config['view'], dataset_name
        )

        actual_batch_size = min(batch_size, len(dataset))
        if actual_batch_size < batch_size:
            print(f"Warning: Reduced batch size from {batch_size} to {actual_batch_size}")

        dataloader = DataLoader(
            dataset, batch_size=actual_batch_size, shuffle=True,
            num_workers=0, collate_fn=multiview_collate_fn, drop_last=True
        )

        return {
            'dataloader': dataloader,
            'dataset': dataset,
            'view_dims': dataset.dims,
            'class_num': dataset.class_num,
            'num_views': len(dataset.multi_view)
        }

    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None
