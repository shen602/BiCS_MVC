# BiCS-MVC 快速开始指南

## 安装步骤

### 1. 克隆或下载项目

```bash
cd BiCS-MVC
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载数据集

从百度网盘下载数据集（详见 `data/README_DATASETS.md`），并将所有 `.mat` 文件放入 `data/` 目录。

## 运行实验

### 方式1：交互式运行（推荐新手）

```bash
python main.py
```

然后根据提示选择数据集编号（1-5）。

### 方式2：命令行指定数据集

```bash
# 在NUSWIDE数据集上运行
python main.py --dataset NUSWIDE

# 在MNIST-USPS数据集上运行
python main.py --dataset MNIST_USPS

# 在Fashion数据集上运行
python main.py --dataset Fashion

# 在Hdigit数据集上运行
python main.py --dataset Hdigit

# 在Digit-Product数据集上运行
python main.py --dataset Digit-Product
```

### 方式3：批量运行所有数据集

```bash
python main.py --batch_all
```

这将依次在所有5个数据集上运行实验，每个数据集运行10次，并报告最佳结果。

## 实验设置

- **运行次数：** 每个数据集默认运行10次
- **随机种子：** 42, 43, 44, ..., 51
- **报告指标：** 最大值、平均值、标准差

## 结果保存

实验结果会自动保存到 `results/` 目录：

```
results/
├── best_result_NUSWIDE.json
├── best_result_MNIST_USPS.json
├── best_result_Fashion.json
├── best_result_Hdigit.json
└── best_result_Digit-Product.json
```

每个JSON文件包含：
- `best_result`: 10次运行中的最佳结果
- `stats`: 统计信息（max, mean, std）

## 示例输出

```
================================================================================
BiCS-MVC: Bidirectional Contrastive Learning with Semantic Consistency
for Multi-View Clustering
================================================================================

Available datasets:
1. NUSWIDE
2. MNIST_USPS
3. Fashion
4. Hdigit
5. Digit-Product

Select dataset (1-5): 1

Loading dataset: NUSWIDE
Successfully loaded NUSWIDE: 5000 samples, 5 classes, 5 views

BiCS-MVC Model for NUSWIDE:
  Feature dim: 512, Projection dim: 128
  Loss weights: contrastive=1.0, semantic=0.3
  Temperature: 0.5
  Use Projector: True

Starting BiCS-MVC training on NUSWIDE...
Training for 50 epochs

Epoch 1/50:
  Total Loss: 2.3456
  Contrastive: 2.1234, Semantic: 0.7408
  Learning Rate: 0.000100

...

Run 1 Results:
ACC: 0.856
NMI: 0.742
ARI: 0.698
Purity: 0.863
```

## 自定义配置

如需修改模型配置，编辑 `config/config.py` 中的 `BiCSMVCConfig.get_dataset_config()` 方法：

```python
optimal_config = {
    'feature_dim': 512,        # 特征维度
    'high_dim': 128,           # 投影维度
    'batch_size': 64,          # 批大小
    'learning_rate': 1e-4,     # 学习率
    'total_epochs': 50,        # 训练轮数
    'contrastive_weight': 1.0, # 对比损失权重
    'semantic_weight': 0.3,    # 语义一致性损失权重
    'temperature': 0.5,        # 温度参数
    'use_projector': True      # 是否使用投影器
}
```

## 常见问题

### Q: CUDA out of memory 错误
A: 减小 `batch_size`，例如从64改为32。

### Q: 数据集加载失败
A: 检查 `.mat` 文件是否在 `data/` 目录中，文件名是否正确。

### Q: 如何只运行1次实验？
A: 修改 `main.py` 中的 `n_runs = 10` 为 `n_runs = 1`。

## 技术支持

如有问题，请提交GitHub Issue或联系作者。
