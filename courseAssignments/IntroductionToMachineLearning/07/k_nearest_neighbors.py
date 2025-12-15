#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
parser.add_argument("--weights", default="uniform", choices=["uniform", "inverse", "softmax"], help="Weighting to use")
# If you add more arguments, ReCodEx will keep them with your default values.


class MNIST:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


def lp_distances(x: np.ndarray, train_data: np.ndarray, p: int) -> np.ndarray:
    """
    Compute L^p distances from a single point x to all rows in train_data.

    x: shape (D,)
    train_data: shape (N_train, D)
    returns: shape (N_train,)
    """
    diff = train_data - x  # broadcasting
    abs_diff_p = np.abs(diff) ** p
    summed = np.sum(abs_diff_p, axis=1)
    distances = summed ** (1.0 / p)
    return distances


def compute_weights(distances: np.ndarray, scheme: str) -> np.ndarray:
    """
    Compute neighbor weights given their distances and weighting scheme.

    distances: shape (k,)
    returns: shape (k,)
    """
    if scheme == "uniform":
        # All neighbors have equal weight
        return np.ones_like(distances, dtype=float)

    if scheme == "inverse":
        # Handle zero distances specially to avoid division by zero.
        # If any distance is zero, give all zero-distance neighbors equal weight
        # and ignore the others.
        zero_mask = (distances == 0)
        if np.any(zero_mask):
            weights = np.zeros_like(distances, dtype=float)
            count_zero = np.sum(zero_mask)
            # Equal weights among zero-distance neighbors
            weights[zero_mask] = 1.0 / count_zero
            return weights
        # Otherwise standard inverse distance
        return 1.0 / distances

    if scheme == "softmax":
        # Softmax(-distances), numerically stable
        scores = -distances
        max_score = np.max(scores)
        shifted = scores - max_score
        exp_scores = np.exp(shifted)
        sum_exp = np.sum(exp_scores)
        # sum_exp should not be zero, but guard just in case
        if sum_exp == 0:
            return np.ones_like(distances, dtype=float)
        return exp_scores / sum_exp

    # Fallback, should not happen because of argparse choices
    return np.ones_like(distances, dtype=float)


def main(args: argparse.Namespace) -> float:
    mnist = MNIST(data_size=args.train_size + args.test_size)
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, test_size=args.test_size, random_state=args.seed)

    num_train = train_data.shape[0]
    num_test = test_data.shape[0]
    num_classes = int(np.max(train_target)) + 1

    k = min(args.k, num_train) 
    p = args.p

    test_predictions = np.empty(num_test, dtype=int)
    test_neighbors = np.empty((num_test, k), dtype=int)

    for i in range(num_test):
        x = test_data[i]  

        distances = lp_distances(x, train_data, p)
        neighbor_indices = np.argsort(distances)[:k]

        test_neighbors[i] = neighbor_indices

        neighbor_labels = train_target[neighbor_indices]
        neighbor_distances = distances[neighbor_indices]

        weights = compute_weights(neighbor_distances, args.weights)

        class_scores = np.zeros(num_classes, dtype=float)
        for label, w in zip(neighbor_labels, weights):
            class_scores[int(label)] += w

        test_predictions[i] = int(np.argmax(class_scores))

    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_target[i] and not examples[test_target[i]]:
                examples[test_target[i]] = [test_data[i], *train_data[test_neighbors[i]]]
        examples = [[img.reshape(28, 28) for img in example] for example in examples if example]
        examples = [[example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1) for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return 100 * accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(main_args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        main_args.k, main_args.p, main_args.weights, accuracy))
