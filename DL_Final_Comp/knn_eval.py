import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def knn_evaluate(train_features, train_labels, 
                test_features, test_labels, k=20):
    """
    k-NN evaluation with cosine similarity
    """
    # Convert to numpy
    if isinstance(train_features, torch.Tensor):
        train_features = train_features.numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.numpy()
    if isinstance(test_features, torch.Tensor):
        test_features = test_features.numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.numpy()
    
    # k-NN with cosine similarity
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_features, train_labels)
    
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    
    return accuracy, predictions


def knn_evaluate_multiple_k(train_features, train_labels,
                           test_features, test_labels, k_values=[10, 20, 50]):
    """Evaluate with multiple k values"""
    results = {}
    for k in k_values:
        acc, _ = knn_evaluate(train_features, train_labels,
                            test_features, test_labels, k=k)
        results[k] = acc
        print(f"k={k}: Accuracy = {acc:.4f}")
    return results

