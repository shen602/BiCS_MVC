"""
BiCS-MVC Main Entry Point
Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering

Paper: Bidirectional Contrastive Learning with Semantic Consistency for Multi-View Clustering
"""
import torch
import numpy as np
import argparse
import time
import json
import os

from config.config import BiCSMVCConfig, DATASET_CONFIGS
from data.dataset import MultiViewDataset, get_dataloader
from models.bics_mvc import BiCSMVC
from utils.trainer import BiCSMVCTrainer
from utils.metrics import evaluate_model


def train_model(model, train_loader, config, device, dataset_name):
    """
    训练BiCS-MVC模型
    Args:
        model: BiCS-MVC模型
        train_loader: 训练数据加载器
        config: 配置
        device: 设备
        dataset_name: 数据集名称
    Returns:
        训练好的模型和损失历史
    """
    trainer = BiCSMVCTrainer(model, train_loader, config, device)
    total_epochs = config['total_epochs']

    print(f"\nStarting BiCS-MVC training on {dataset_name}...")
    print(f"Training for {total_epochs} epochs")

    train_losses = []
    start_time = time.time()

    for epoch in range(total_epochs):
        try:
            avg_loss, loss_dict = trainer.train_epoch(epoch)
            train_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == total_epochs - 1:
                lr = trainer.scheduler.get_last_lr()[0]
                print(f'Epoch {epoch + 1}/{total_epochs}:')
                print(f'  Total Loss: {avg_loss:.4f}')
                print(f'  Contrastive: {loss_dict["contrastive"]:.4f}, '
                      f'Semantic: {loss_dict["semantic"]:.4f}')
                print(f'  Learning Rate: {lr:.6f}')

        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            continue

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds.")

    return model, train_losses


def run_experiment(dataset_name, config=None, seed=None):
    """
    运行单次实验
    Args:
        dataset_name: 数据集名称
        config: 配置（可选）
        seed: 随机种子（可选）
    Returns:
        评估结果
    """
    try:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        dataset_config = BiCSMVCConfig.get_dataset_config(dataset_name)
        experiment_config = dataset_config.copy()

        if config:
            experiment_config.update(config)

        # 加载数据
        data_info = get_dataloader(dataset_name, experiment_config['batch_size'])
        if data_info is None:
            return None

        # 创建模型
        model = BiCSMVC(
            view_dims=data_info['view_dims'],
            feature_dim=experiment_config['feature_dim'],
            high_dim=experiment_config['high_dim'],
            class_num=data_info['class_num'],
            device=BiCSMVCConfig.device,
            dataset_name=dataset_name,
            config=experiment_config
        )
        model.to(BiCSMVCConfig.device)

        # 训练模型
        model, train_losses = train_model(
            model, data_info['dataloader'], experiment_config,
            BiCSMVCConfig.device, dataset_name
        )

        # 评估模型
        results = evaluate_model(
            model, BiCSMVCConfig.device, data_info['dataset'], data_info['class_num']
        )

        return results

    except Exception as e:
        print(f"Experiment error: {e}")
        return None


def round3(x):
    """保留3位小数"""
    return float(np.round(x, 3))


def main():
    """主函数"""
    datasets = ['NUSWIDE', 'MNIST_USPS', 'Fashion', 'Hdigit', 'Digit-Product']
    parser = argparse.ArgumentParser(
        description='BiCS-MVC: Bidirectional Contrastive Learning with Semantic Consistency'
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name')
    parser.add_argument('--batch_all', action='store_true',
                        help='Run experiments on all datasets')
    args = parser.parse_args()

    print("=" * 80)
    print("BiCS-MVC: Bidirectional Contrastive Learning with Semantic Consistency")
    print("for Multi-View Clustering")
    print("=" * 80)
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")

    if args.batch_all:
        print("\nRunning experiments on all datasets (10 runs each, reporting max)...")
        summary_results = {}

        for dataset_name in datasets:
            print(f"\n{'=' * 80}")
            print(f"Dataset: {dataset_name}")
            print('=' * 80)

            all_results = []
            n_runs = 10

            for i in range(n_runs):
                print(f"\n{'=' * 60}")
                print(f"{dataset_name} - Run {i + 1}/{n_runs}")
                print('=' * 60)

                current_seed = 42 + i
                try:
                    results = run_experiment(dataset_name, seed=current_seed)
                    if results:
                        print(f"\nRun {i + 1} Results:")
                        print(f"ACC: {results['accuracy']:.3f}")
                        print(f"NMI: {results['nmi']:.3f}")
                        print(f"ARI: {results['ari']:.3f}")
                        print(f"Purity: {results['purity']:.3f}")
                        all_results.append(results)
                except Exception as e:
                    print(f"{dataset_name} Run {i + 1} error: {e}")
                    continue

            if all_results:
                print(f"\n{'=' * 60}")
                print(f"{dataset_name} - Summary Statistics ({n_runs} runs)")
                print('=' * 60)

                metrics = ['accuracy', 'nmi', 'ari', 'purity']
                final_stats = {}

                for metric in metrics:
                    values = [r[metric] for r in all_results]
                    max_val = round3(max(values))
                    mean_val = round3(np.mean(values))
                    std_val = round3(np.std(values))
                    final_stats[metric] = {
                        'max': max_val,
                        'mean': mean_val,
                        'std': std_val
                    }
                    print(f"{metric.upper()}: Max={max_val:.3f}, "
                          f"Mean={mean_val:.3f}, Std={std_val:.3f}")

                # 保存最佳结果
                best_run = max(all_results, key=lambda r: r['accuracy'])
                best_run_rounded = {
                    'accuracy': round3(best_run['accuracy']),
                    'nmi': round3(best_run['nmi']),
                    'ari': round3(best_run['ari']),
                    'purity': round3(best_run['purity'])
                }

                save_obj = {
                    'dataset': dataset_name,
                    'best_result': best_run_rounded,
                    'stats': final_stats
                }

                output_filename = f'best_result_{dataset_name}.json'
                results_dir = "results"

                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                output_file = os.path.join(results_dir, output_filename)

                try:
                    with open(output_file, 'w') as f:
                        json.dump(save_obj, f, indent=4)
                    print(f"\nBest results saved to {output_file}")
                except Exception as e:
                    print(f"Error saving results for {dataset_name}: {e}")

                summary_results[dataset_name] = save_obj
            else:
                print(f"{dataset_name}: No valid results obtained.")

        return summary_results

    # 单个数据集实验
    selected_dataset = None
    try:
        if args.dataset is not None:
            if args.dataset in datasets:
                selected_dataset = args.dataset
            else:
                print("Invalid dataset selection!")
                return None
        else:
            try:
                user_input = input(f"\nSelect dataset (1-{len(datasets)}): ")
                dataset_choice = int(user_input)
                if 1 <= dataset_choice <= len(datasets):
                    selected_dataset = datasets[dataset_choice - 1]
                else:
                    print("Invalid dataset selection!")
                    return None
            except ValueError:
                print("Invalid input!")
                return None
    except Exception as e:
        print(f"Dataset selection error: {e}")
        print("Using NUSWIDE as default dataset.")
        selected_dataset = 'NUSWIDE'

    if selected_dataset:
        print(f"\nRunning experiments on {selected_dataset} (10 runs, reporting max)...")
        all_results = []
        n_runs = 10

        for i in range(n_runs):
            print(f"\n{'=' * 60}")
            print(f"Run {i + 1}/{n_runs}")
            print('=' * 60)

            current_seed = 42 + i
            try:
                results = run_experiment(selected_dataset, seed=current_seed)
                if results:
                    print(f"\nRun {i + 1} Results:")
                    print(f"ACC: {results['accuracy']:.3f}")
                    print(f"NMI: {results['nmi']:.3f}")
                    print(f"ARI: {results['ari']:.3f}")
                    print(f"Purity: {results['purity']:.3f}")
                    all_results.append(results)
            except Exception as e:
                print(f"Run {i + 1} error: {e}")
                continue

        if all_results:
            print(f"\n{'=' * 60}")
            print(f"{selected_dataset} - Summary Statistics ({n_runs} runs)")
            print('=' * 60)

            metrics = ['accuracy', 'nmi', 'ari', 'purity']
            final_stats = {}

            for metric in metrics:
                values = [r[metric] for r in all_results]
                max_val = round3(max(values))
                mean_val = round3(np.mean(values))
                std_val = round3(np.std(values))
                final_stats[metric] = {
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val
                }
                print(f"{metric.upper()}: Max={max_val:.3f}, "
                      f"Mean={mean_val:.3f}, Std={std_val:.3f}")

            # 保存结果
            best_run = max(all_results, key=lambda r: r['accuracy'])
            best_run_rounded = {
                'accuracy': round3(best_run['accuracy']),
                'nmi': round3(best_run['nmi']),
                'ari': round3(best_run['ari']),
                'purity': round3(best_run['purity'])
            }

            save_obj = {
                'dataset': selected_dataset,
                'best_result': best_run_rounded,
                'stats': final_stats
            }

            output_file = f'best_result_{selected_dataset}.json'
            try:
                with open(output_file, 'w') as f:
                    json.dump(save_obj, f, indent=4)
                print(f"\nBest results saved to {output_file}")
            except Exception as e:
                print(f"Error saving results: {e}")

            return final_stats
        else:
            print("No valid results obtained.")
            return None

    return None


if __name__ == "__main__":
    results = main()
