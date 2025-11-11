import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def linear_probe_evaluate(train_features, train_labels,
                         test_features, test_labels,
                         C=1.0, max_iter=1000):
    """
    Linear probe evaluation
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
    
    # Logistic regression
    clf = LogisticRegression(C=C, max_iter=max_iter, 
                            multi_class='multinomial', 
                            solver='lbfgs', n_jobs=-1)
    clf.fit(train_features, train_labels)
    
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    
    return accuracy, predictions, clf

