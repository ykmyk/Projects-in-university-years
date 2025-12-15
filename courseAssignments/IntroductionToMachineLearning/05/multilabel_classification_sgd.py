# #!/usr/bin/env python3
# import argparse

# import numpy as np
# import sklearn.datasets
# import sklearn.metrics
# import sklearn.model_selection

# parser = argparse.ArgumentParser()
# # These arguments will be set appropriately by ReCodEx, even if you change them.
# parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
# parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
# parser.add_argument("--data_size", default=200, type=int, help="Data size")
# parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
# parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
# parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
# parser.add_argument("--seed", default=42, type=int, help="Random seed")
# parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# # If you add more arguments, ReCodEx will keep them with your default values.


# def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
#     # Create a random generator with a given seed.
#     generator = np.random.RandomState(args.seed)

#     # Generate an artificial classification dataset.
#     data, target_list = sklearn.datasets.make_multilabel_classification(
#         n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
#         return_indicator=False, random_state=args.seed)

#     # TODO: The `target` is a list of classes for every input example. Convert
#     # it to a dense representation (n-hot encoding) -- for each input example,
#     # the target should be vector of `args.classes` binary indicators.
#     target = ...

#     # Append a constant feature with value 1 to the end of all input data.
#     # Then we do not need to explicitly represent bias - it becomes the last weight.
#     data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

#     # Split the dataset into a train set and a test set.
#     # Use `sklearn.model_selection.train_test_split` method call, passing
#     # arguments `test_size=args.test_size, random_state=args.seed`.
#     train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
#         data, target, test_size=args.test_size, random_state=args.seed)

#     # Generate initial model weights.
#     weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

#     for epoch in range(args.epochs):
#         permutation = generator.permutation(train_data.shape[0])

#         # TODO: Process the data in the order of `permutation`. For every
#         # `args.batch_size` of them, average their gradient, and update the weights.
#         # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
#         ...

#         # TODO: After the SGD epoch, compute the micro-averaged and the
#         # macro-averaged F1-score for both the train test and the test set.
#         # Compute these scores manually, without using `sklearn.metrics`.
#         train_f1_micro, train_f1_macro, test_f1_micro, test_f1_macro = ...

#         print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
#             epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

#     return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


# if __name__ == "__main__":
#     main_args = parser.parse_args([] if "__file__" not in globals() else None)
#     weights, metrics = main(main_args)
#     print("Learned weights:",
#           *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")

#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def _predict_proba(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # X: [N, D], W: [D, C] -> probs: [N, C]
    return _sigmoid(X @ W)


def _predict_labels(X: np.ndarray, W: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (_predict_proba(X, W) >= threshold).astype(np.int32)


def _f1_micro_macro(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    # y_true, y_pred: [N, C] in {0,1}
    tp = np.logical_and(y_true == 1, y_pred == 1).sum(axis=0).astype(np.float64)
    fp = np.logical_and(y_true == 0, y_pred == 1).sum(axis=0).astype(np.float64)
    fn = np.logical_and(y_true == 1, y_pred == 0).sum(axis=0).astype(np.float64)

    # micro
    TPm, FPm, FNm = tp.sum(), fp.sum(), fn.sum()
    denom_micro = 2 * TPm + FPm + FNm
    f1_micro = (2 * TPm / denom_micro) if denom_micro > 0 else 0.0

    # macro (per-class then mean)
    denom = 2 * tp + fp + fn
    f1_per_class = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    f1_macro = float(f1_per_class.mean()) if f1_per_class.size > 0 else 0.0

    return float(f1_micro), float(f1_macro)


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # Convert list-of-labels to dense n-hot matrix [N, C]
    target = np.zeros((len(target_list), args.classes), dtype=np.int32)
    for i, labels in enumerate(target_list):
        target[i, labels] = 1

    # Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1.0)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights [D, C].
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    N = train_data.shape[0]
    B = args.batch_size
    assert N % B == 0, "Batch size must exactly divide the number of training samples."

    for epoch in range(args.epochs):
        permutation = generator.permutation(N)

        for start in range(0, N, B):
            idx = permutation[start:start + B]
            Xb = train_data[idx]
            Yb = train_target[idx].astype(np.float64)

            P = _sigmoid(Xb @ weights)
            G = (Xb.T @ (P - Yb)) / B
            weights -= args.learning_rate * G

        # Evaluate F1 on train and test (manual, no sklearn.metrics)
        train_pred = _predict_labels(train_data, weights, threshold=0.5)
        test_pred = _predict_labels(test_data, weights, threshold=0.5)

        train_f1_micro, train_f1_macro = _f1_micro_macro(train_target, train_pred)
        test_f1_micro, test_f1_macro = _f1_micro_macro(test_target, test_pred)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
