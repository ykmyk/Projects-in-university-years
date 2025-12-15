#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


class DecisionTreeClassifierCustom:
    def __init__(self, criterion="gini", max_depth=None, max_leaves=None, min_to_split=2):
        self.criterion_name = criterion
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_to_split = min_to_split

        self.n_classes = None
        self.n_features = None
        self.root = None
        self.creation_counter = 0

    class Node:
        def __init__(self, *, prediction, indices, depth, criterion_value, created_order):
            self.prediction = prediction
            self.feature_index = None
            self.threshold = None
            self.left = None
            self.right = None
            self.is_leaf = True
            self.indices = indices
            self.depth = depth
            self.criterion_value = criterion_value
            self.created_order = created_order

    def criterion(self, counts: np.ndarray) -> float:
        """Return impurity scaled by node size (|I| * impurity measure)."""
        n = counts.sum()
        if n == 0:
            return 0.0
        p = counts / n

        if self.criterion_name == "gini":
            return float(n * np.sum(p * (1.0 - p)))
        else:
            nz = p > 0
            return float(-n * np.sum(p[nz] * np.log(p[nz])))

    def make_leaf(self, indices: np.ndarray, depth: int, X: np.ndarray, y: np.ndarray):
        counts = np.bincount(y[indices], minlength=self.n_classes)
        prediction = int(np.argmax(counts))
        criterion_value = self.criterion(counts)
        node = self.Node(prediction=prediction, indices=indices, depth=depth,
                         criterion_value=criterion_value, created_order=self.creation_counter)
        self.creation_counter += 1
        return node

    def find_best_split(self, node, X: np.ndarray, y: np.ndarray):
        indices = node.indices
        node_num = len(indices)

        if self.max_depth is not None and node.depth >= self.max_depth:
            return None
        if node_num < self.min_to_split:
            return None
        if node.criterion_value == 0.0:
            return None

        y_node = y[indices]
        cur_crit = node.criterion_value
        n_features = self.n_features
        best_feature = None
        best_threshold = None
        best_delta = 0.0

        for feature in range(n_features):
            values = X[indices, feature]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_y = y_node[order]
            total_counts = np.bincount(sorted_y, minlength=self.n_classes).astype(np.int64)
            left_counts = np.zeros_like(total_counts)

            for i in range(node_num - 1):
                label = sorted_y[i]
                left_counts[label] += 1
                total_counts[label] -= 1

                if sorted_values[i] == sorted_values[i + 1]:
                    continue

                threshold = 0.5 * (sorted_values[i] + sorted_values[i + 1])

                c_left = self.criterion(left_counts)
                c_right = self.criterion(total_counts)
                delta = c_left + c_right - cur_crit

                if best_feature is None or delta < best_delta:
                    best_feature = feature
                    best_threshold = threshold
                    best_delta = delta

        if best_feature is None or best_delta >= 0.0:
            return None

        feature_values = X[indices, best_feature]
        left_mask = feature_values <= best_threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return None

        return best_feature, best_threshold, left_indices, right_indices, best_delta

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)

        self.n_features = X.shape[1]
        self.n_classes = int(np.max(y)) + 1

        n_samples = X.shape[0]
        indices = np.arange(n_samples, dtype=int)
        self.creation_counter = 0

        if self.max_leaves is None:
            self.root = self.build_recursive(indices, depth=0, X=X, y=y)
        else:
            self.root = self.build_best_first(indices, X=X, y=y)

        self.drop_indices(self.root)
        return self

    def build_recursive(self, indices: np.ndarray, depth: int, X: np.ndarray, y: np.ndarray):
        node = self.make_leaf(indices, depth, X, y)
        split = self.find_best_split(node, X, y)

        if split is None:
            return node

        feature, threshold, left_idx, right_idx, _ = split
        node.is_leaf = False
        node.feature_index = feature
        node.threshold = threshold
        node.left = self.build_recursive(left_idx, depth + 1, X, y)
        node.right = self.build_recursive(right_idx, depth + 1, X, y)
        return node

    def build_best_first(self, indices: np.ndarray, X: np.ndarray, y: np.ndarray):
        self.max_leaves = int(self.max_leaves)
        root = self.make_leaf(indices, depth=0, X=X, y=y)
        leaves = [root]
        n_leaves = 1

        while n_leaves < self.max_leaves:
            best_leaf = None
            best_split = None
            best_delta = 0.0

            for leaf in list(leaves):
                split = self.find_best_split(leaf, X, y)
                if split is None:
                    continue
                _, _, _, _, delta = split
                if (best_split is None
                        or delta < best_delta
                        or (np.isclose(delta, best_delta) and leaf.created_order < best_leaf.created_order)):
                    best_leaf = leaf
                    best_split = split
                    best_delta = delta

            if best_split is None:
                break

            feature, threshold, left_idx, right_idx, _ = best_split
            best_leaf.is_leaf = False
            best_leaf.feature_index = feature
            best_leaf.threshold = threshold

            left_child = self.make_leaf(left_idx, best_leaf.depth + 1, X, y)
            right_child = self.make_leaf(right_idx, best_leaf.depth + 1, X, y)
            best_leaf.left = left_child
            best_leaf.right = right_child

            leaves.remove(best_leaf)
            leaves.append(left_child)
            leaves.append(right_child)
            n_leaves += 1

        return root

    def drop_indices(self, node):
        if node is None:
            return
        node.indices = None
        if not node.is_leaf:
            self.drop_indices(node.left)
            self.drop_indices(node.right)

    def predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.array([self.predict_one(row) for row in X], dtype=int)


def main(args: argparse.Namespace) -> tuple[float, float]:
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    tree = DecisionTreeClassifierCustom(
        criterion=args.criterion, max_depth=args.max_depth,
        max_leaves=args.max_leaves, min_to_split=args.min_to_split,
    )

    tree.fit(train_data, train_target)

    train_pred = tree.predict(train_data)
    test_pred = tree.predict(test_data)

    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_pred)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_pred)

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(main_args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
