#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

class Node:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def predict(self, X):
        if self.value is not None:
            return np.full(X.shape[0], self.value)
        
        preds = np.zeros(X.shape[0])
        mask = X[:, self.feature] <= self.threshold
        
        if np.any(mask):
            preds[mask] = self.left.predict(X[mask])
        if np.any(~mask):
            preds[~mask] = self.right.predict(X[~mask])
            
        return preds

def find_best_split(X, g, h, l2):
    n_samples, n_features = X.shape
    G = np.sum(g)
    H = np.sum(h)
    
    best_score = 0.0
    best_split = None
    
    score_parent = (G**2) / (H + l2)
    
    for feat in range(n_features):
        feature_values = X[:, feat]
        sorted_indices = np.argsort(feature_values)
        
        sorted_values = feature_values[sorted_indices]
        sorted_g = g[sorted_indices]
        sorted_h = h[sorted_indices]
        
        G_L = 0.0
        H_L = 0.0
        
        for i in range(n_samples - 1):
            G_L += sorted_g[i]
            H_L += sorted_h[i]
            
            if sorted_values[i] != sorted_values[i+1]:
                G_R = G - G_L
                H_R = H - H_L
                
                score_L = (G_L**2) / (H_L + l2)
                score_R = (G_R**2) / (H_R + l2)
                
                gain = score_L + score_R - score_parent
                
                if gain > best_score:
                    best_score = gain
                    threshold = (sorted_values[i] + sorted_values[i+1]) / 2.0
                    best_split = (feat, threshold)
                    
    return best_split

def build_tree(X, g, h, depth, max_depth, l2):
    G = np.sum(g)
    H = np.sum(h)
    leaf_weight = -G / (H + l2)
    
    if depth >= max_depth or X.shape[0] <= 1:
        return Node(value=leaf_weight)
    
    split = find_best_split(X, g, h, l2)
    
    if split is None:
        return Node(value=leaf_weight)
        
    feat, threshold = split
    mask = X[:, feat] <= threshold
    
    left_child = build_tree(X[mask], g[mask], h[mask], depth + 1, max_depth, l2)
    right_child = build_tree(X[~mask], g[~mask], h[~mask], depth + 1, max_depth, l2)
    
    return Node(feature=feat, threshold=threshold, left=left_child, right=right_child)

def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    classes = np.max(target) + 1

    train_logits = np.zeros((train_data.shape[0], classes))
    test_logits = np.zeros((test_data.shape[0], classes))
    
    train_accuracies = []
    test_accuracies = []

    for t in range(args.trees):
        max_train = np.max(train_logits, axis=1, keepdims=True)
        exp_train = np.exp(train_logits - max_train)
        probs_train = exp_train / np.sum(exp_train, axis=1, keepdims=True)
        
        current_iter_trees = []
        
        for k in range(classes):
            
            y_k = (train_target == k).astype(float)
            g = probs_train[:, k] - y_k
            h = probs_train[:, k] * (1.0 - probs_train[:, k])
            
            tree = build_tree(train_data, g, h, 0, args.max_depth, args.l2)
            current_iter_trees.append(tree)

        for k in range(classes):
            tree = current_iter_trees[k]
            
            pred_train = tree.predict(train_data)
            pred_test = tree.predict(test_data)
            
            train_logits[:, k] += args.learning_rate * pred_train
            test_logits[:, k] += args.learning_rate * pred_test
            
        acc_train = sklearn.metrics.accuracy_score(train_target, np.argmax(train_logits, axis=1))
        acc_test = sklearn.metrics.accuracy_score(test_target, np.argmax(test_logits, axis=1))
        
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)

    return [100 * acc for acc in train_accuracies], [100 * acc for acc in test_accuracies]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(main_args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, train_accuracy, test_accuracy))