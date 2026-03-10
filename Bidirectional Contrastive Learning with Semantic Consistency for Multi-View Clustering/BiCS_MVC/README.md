# BiCS-MVC: Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering

Official PyTorch implementation of the paper "Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering".

## Overview

BiCS-MVC is a novel multi-view clustering method that leverages bidirectional contrastive learning and semantic consistency to learn discriminative representations from multiple views of data.

## Project Structure

```
BiCS-MVC/
├── config/
│   └── config.py          # Configuration parameters
├── data/
│   └── dataset.py         # Dataset loading and preprocessing
├── models/
│   ├── bics_mvc.py        # BiCS-MVC main model
│   └── losses.py          # Loss functions (Bidirectional Contrastive & Semantic Consistency)
├── utils/
│   ├── metrics.py         # Evaluation metrics
│   └── trainer.py         # Training utilities
├── main.py                # Main entry point
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- scikit-learn >= 0.24.0

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

The code supports the following datasets:

1. **NUSWIDE** (5 views, 5 classes)
2. **MNIST-USPS** (2 views, 10 classes)
3. **Fashion** (3 views, 10 classes)
4. **Hdigit** (2 views, 10 classes)
5. **Digit-Product** (2 views, 10 classes)

### Dataset Download

Download the datasets from Baidu Netdisk:

**Link:** [Your Baidu Netdisk Link Here]
**Password:** [Your Password Here]

After downloading, place the `.mat` files in the `data/` directory:

```
BiCS-MVC/
├── data/
│   ├── NUSWIDE.mat
│   ├── MNIST_USPS.mat
│   ├── Fashion.mat
│   ├── Hdigit.mat
│   └── Digit-Product.mat
```

## Usage

### Run on a Single Dataset

Interactive mode:
```bash
python main.py
```

Command-line mode:
```bash
python main.py --dataset NUSWIDE
```

### Run on All Datasets

```bash
python main.py --batch_all
```

This will run 10 experiments on each dataset and report the maximum and average results.

## Model Configuration

The model uses dataset-specific optimal configurations defined in `config/config.py`:

- **Feature dimension**: 512
- **Projection dimension**: 128
- **Batch size**: 64
- **Learning rate**: 1e-4
- **Epochs**: 50
- **Contrastive weight**: 1.0
- **Semantic weight**: 0.3
- **Temperature**: 0.5

## Evaluation Metrics

The model is evaluated using four standard clustering metrics:

- **ACC**: Clustering Accuracy
- **NMI**: Normalized Mutual Information
- **ARI**: Adjusted Rand Index
- **Purity**: Cluster Purity

## Results

Results are automatically saved to JSON files in the `results/` directory:

```
results/
├── best_result_NUSWIDE.json
├── best_result_MNIST_USPS.json
├── best_result_Fashion.json
├── best_result_Hdigit.json
└── best_result_Digit-Product.json
```

Each result file contains:
- Best result across all runs
- Statistics (max, mean, std) for each metric


