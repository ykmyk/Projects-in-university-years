#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)


    n_train, d = train_data.shape
    n_test = test_data.shape[0]
    K = args.classes

    class_counts = np.bincount(train_target, minlength=K)
    priors = class_counts / n_train
    log_priors = np.log(priors)

    class_indices = [np.where(train_target == k)[0] for k in range(K)]

    if args.naive_bayes_type == "gaussian":
        means = np.zeros((K, d))
        variances = np.zeros((K, d))

        for k in range(K):
            Xk = train_data[class_indices[k]]
            mk = Xk.mean(axis=0)
            means[k] = mk

            diff = Xk - mk
            vk = (diff ** 2).mean(axis=0)
            variances[k] = vk + args.alpha

        log_joint = np.zeros((n_test, K))
        two_pi = 2 * np.pi

        for k in range(K):
            mean_k = means[k]
            var_k = variances[k]
            diff = test_data - mean_k
            log_likelihood = -0.5 * (np.log(two_pi * var_k) + (diff ** 2) / var_k).sum(axis=1)
            log_joint[:, k] = log_priors[k] + log_likelihood


    elif args.naive_bayes_type == "bernoulli":
        train_bin = (train_data >= 8).astype(int)
        test_bin = (test_data >= 8).astype(int)

        p = np.zeros((K, d))
        for k in range(K):
            Xk = train_bin[class_indices[k]]
            Nk = Xk.shape[0]
            count_ones = Xk.sum(axis=0)
            p[k] = (count_ones + args.alpha) / (Nk + 2 * args.alpha)

        log_p = np.log(p)
        log_1mp = np.log(1 - p)

        log_joint = np.zeros((n_test, K))
        X = test_bin

        for k in range(K):
            log_likelihood = (X * log_p[k] + (1 - X) * log_1mp[k]).sum(axis=1)
            log_joint[:, k] = log_priors[k] + log_likelihood

    else:
        p = np.zeros((K, d))
        for k in range(K):
            Xk = train_data[class_indices[k]]
            nd = Xk.sum(axis=0)
            total = nd.sum()
            p[k] = (nd + args.alpha) / (total + args.alpha * d)

        log_p = np.log(p)

        log_joint = np.zeros((n_test, K))
        X = test_data

        for k in range(K):
            log_likelihood = (X * log_p[k]).sum(axis=1)
            log_joint[:, k] = log_priors[k] + log_likelihood

    predictions = np.argmax(log_joint, axis=1)
    test_accuracy = np.mean(predictions == test_target)

    test_log_probability = log_joint[np.arange(n_test), test_target].sum()

    return 100 * test_accuracy, test_log_probability


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(main_args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))
