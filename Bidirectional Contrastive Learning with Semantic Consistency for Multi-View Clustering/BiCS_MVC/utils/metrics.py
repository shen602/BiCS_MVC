"""
BiCS-MVC Evaluation Metrics Module
Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering
"""
import numpy as np
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader


def clustering_accuracy(y_true, y_pred):
    """
    计算聚类准确率
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    Returns:
        聚类准确率
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true = y_true - np.min(y_true) if np.min(y_true) > 0 else y_true
    y_pred = y_pred - np.min(y_pred) if np.min(y_pred) > 0 else y_pred

    n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i in range(len(y_true)):
        conf_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-conf_matrix)

    mapping = dict(zip(row_ind, col_ind))
    y_pred_mapped = np.array([mapping[x] for x in y_pred])

    acc = accuracy_score(y_true, y_pred_mapped)
    return acc


def evaluate_model(model, device, dataset, class_num):
    """
    评估模型性能
    Args:
        model: BiCS-MVC模型
        device: 设备
        dataset: 数据集
        class_num: 类别数
    Returns:
        评估指标字典
    """
    model.eval()

    from data.dataset import multiview_collate_fn
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=multiview_collate_fn
    )

    def extract_features(loader):
        """提取特征"""
        all_features = []
        all_labels = []
        with torch.no_grad():
            for xs, labels, idx in loader:
                if len(xs) == 0:
                    continue
                xs_device = [x.to(device) for x in xs]
                outputs = model(xs_device)
                features = torch.stack(outputs['zs']).mean(dim=0)
                all_features.append(features.cpu())
                all_labels.append(labels)

        if len(all_features) == 0:
            return None, None
        return torch.cat(all_features, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()

    features, labels = extract_features(dataloader)

    if features is None:
        return {'accuracy': 0.0, 'nmi': 0.0, 'ari': 0.0, 'purity': 0.0}

    try:
        # K-means聚类
        kmeans = KMeans(n_clusters=class_num, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # 计算评估指标
        acc = clustering_accuracy(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        ari = adjusted_rand_score(labels, cluster_labels)

        # 计算纯度
        cm = confusion_matrix(labels, cluster_labels)
        purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)

        return {
            'accuracy': acc,
            'nmi': nmi,
            'ari': ari,
            'purity': purity
        }

    except Exception as e:
        print(f"Evaluation error: {e}")
        return {'accuracy': 0.0, 'nmi': 0.0, 'ari': 0.0, 'purity': 0.0}
