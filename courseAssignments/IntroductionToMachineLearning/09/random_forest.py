#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1.0, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=73, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    generator_feature_subsampling = np.random.RandomState(args.seed)

    def subsample_features(number_of_features: int) -> np.ndarray:
        size = int(args.feature_subsampling * number_of_features)
        size = max(1, size)
        return np.sort(generator_feature_subsampling.choice(
            number_of_features, size=size, replace=False))

    generator_bootstrapping = np.random.RandomState(args.seed)

    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    class DecisionTree:
        class Node:
            __slots__ = ("is_leaf", "prediction", "feature", "threshold", "left", "right")

            def __init__(self, is_leaf: bool, prediction: int,
                         feature: int | None = None, threshold: float | None = None,
                         left: "DecisionTree.Node | None" = None,
                         right: "DecisionTree.Node | None" = None):
                self.is_leaf = is_leaf
                self.prediction = prediction
                self.feature = feature
                self.threshold = threshold
                self.left = left
                self.right = right

        def __init__(self, max_depth: int | None, n_classes: int):
            self.max_depth = max_depth
            self.n_classes = n_classes
            self.root: DecisionTree.Node | None = None

        @staticmethod
        def _node_entropy_criterion(counts: np.ndarray) -> float:
            n = counts.sum()
            if n == 0:
                return 0.0
            p = counts.astype(np.float64) / n
            mask = p > 0
            return -n * np.sum(p[mask] * np.log(p[mask]))

        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            n_samples, n_features = X.shape

            def build_node(indices: np.ndarray, depth: int) -> "DecisionTree.Node":
                node_y = y[indices]
                counts = np.bincount(node_y, minlength=self.n_classes)
                prediction = int(np.argmax(counts))
                criterion_here = self._node_entropy_criterion(counts)

                if criterion_here == 0.0:
                    return DecisionTree.Node(True, prediction)
                if self.max_depth is not None and depth >= self.max_depth:
                    return DecisionTree.Node(True, prediction)

                n_features_local = X.shape[1]
                feature_indices = subsample_features(n_features_local)

                best_feature = None
                best_threshold = None
                best_crit_value = np.inf
                best_left_indices = None
                best_right_indices = None

                for feature in feature_indices:
                    feature_values = X[indices, feature]
                    sorted_order = np.argsort(feature_values)
                    sorted_values = feature_values[sorted_order]
                    sorted_labels = node_y[sorted_order]

                    right_counts = np.bincount(sorted_labels, minlength=self.n_classes).astype(np.float64)
                    left_counts = np.zeros(self.n_classes, dtype=np.float64)

                    n_node = len(indices)

                    for i in range(n_node - 1):
                        label = sorted_labels[i]
                        left_counts[label] += 1.0
                        right_counts[label] -= 1.0

                        if sorted_values[i] == sorted_values[i + 1]:
                            continue

                        n_left = i + 1
                        n_right = n_node - n_left
                        if n_left == 0 or n_right == 0:
                            continue

                        crit_left = self._node_entropy_criterion(left_counts)
                        crit_right = self._node_entropy_criterion(right_counts)
                        crit_value = crit_left + crit_right

                        if crit_value < best_crit_value:
                            best_crit_value = crit_value
                            best_feature = feature
                            best_threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0

                if best_feature is None or best_threshold is None:
                    return DecisionTree.Node(True, prediction)

                if best_crit_value >= criterion_here:
                    return DecisionTree.Node(True, prediction)

                feature_values_full = X[indices, best_feature]
                left_mask = feature_values_full <= best_threshold
                right_mask = ~left_mask

                left_indices = indices[left_mask]
                right_indices = indices[right_mask]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    return DecisionTree.Node(True, prediction)

                left_child = build_node(left_indices, depth + 1)
                right_child = build_node(right_indices, depth + 1)

                return DecisionTree.Node(False, prediction,
                                         feature=best_feature,
                                         threshold=best_threshold,
                                         left=left_child,
                                         right=right_child)

            all_indices = np.arange(n_samples)
            self.root = build_node(all_indices, depth=0)

        def _predict_one(self, x: np.ndarray) -> int:
            node = self.root
            assert node is not None
            while not node.is_leaf:
                assert node.feature is not None and node.threshold is not None
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
                assert node is not None
            return node.prediction

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.array([self._predict_one(x) for x in X], dtype=int)

    n_classes = int(np.max(target)) + 1

    forest: list[DecisionTree] = []

    for _ in range(args.trees):
        if args.bagging:
            indices = bootstrap_dataset(train_data)
            X_boot = train_data[indices]
            y_boot = train_target[indices]
        else:
            X_boot = train_data
            y_boot = train_target

        tree = DecisionTree(max_depth=args.max_depth, n_classes=n_classes)
        tree.fit(X_boot, y_boot)
        forest.append(tree)

    def forest_predict(X: np.ndarray) -> np.ndarray:
        all_preds = np.array([tree.predict(X) for tree in forest], dtype=int)
        all_preds = all_preds.T
        votes = []
        for row in all_preds:
            counts = np.bincount(row, minlength=n_classes)
            votes.append(int(np.argmax(counts)))
        return np.array(votes, dtype=int)

    train_pred = forest_predict(train_data)
    test_pred = forest_predict(test_data)

    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_pred)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_pred)

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(main_args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
