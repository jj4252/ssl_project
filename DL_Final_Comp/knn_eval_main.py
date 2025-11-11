import argparse
import yaml
import torch

from knn_eval import knn_evaluate, knn_evaluate_multiple_k
from linear_probe import linear_probe_evaluate


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required=True, 
                       help='Path to features file (contains both train and test)')
    parser.add_argument('--eval_config', type=str, required=True)
    args = parser.parse_args()
    
    eval_cfg = load_config(args.eval_config)
    
    # Load features (single file contains both train and test)
    data = torch.load(args.features)
    
    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    test_labels = data['test_labels']
    
    print(f"Train: {train_features.shape}, Test: {test_features.shape}")
    
    # k-NN evaluation
    print("\n=== k-NN Evaluation ===")
    results = knn_evaluate_multiple_k(
        train_features, train_labels,
        test_features, test_labels,
        k_values=eval_cfg['k_values']
    )
    
    # Linear probe (optional)
    if eval_cfg.get('linear_probe', False):
        print("\n=== Linear Probe Evaluation ===")
        acc, _, _ = linear_probe_evaluate(
            train_features, train_labels,
            test_features, test_labels,
            C=eval_cfg.get('linear_probe_C', 1.0)
        )
        print(f"Linear Probe Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()

