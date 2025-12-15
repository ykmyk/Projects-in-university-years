#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()
    # print(dataset.data.shape)
    # print(dataset.data[:5])
    # print(dataset.target[:5])
    # print(dataset.feature_names) 

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Append a constant feature with value 1 to the end of all input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    d = dataset.data

    # each column of X represents each feature(one feature of many patient)
    # each row represents one sample(like one patient)
    X = np.c_[d, np.ones(d.shape[0], dtype=int)] # feature matrices = input matrix
    
    # target vector = output vector
    y = dataset.target # target vectors

    # print(X.shape)
    # dataset.data.shape = (# rows, # columns)


    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    # print(f'shape is {X_train.shape} \n Training set feature: {X_train}')
    # print(f'shape is {y_train.shape} \n Training target: {y_train}')

    # print(f'shape is {X_test.shape} \n Test set feature: {X_test}')
    # print(f'shape is {y_test.shape} \n Test target: {y_test}')

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    X_train_trans = np.matrix.transpose(X_train)
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train_trans, X_train)), X_train_trans), y_train)
   
    # TODO: Predict target values on the test set.
    y = np.dot(X_test, w)
    # print(X_test.shape)
    # print(w.shape)

    # TODO: Manually compute root mean square error on the test set predictions.
    rmse = float(np.sqrt(np.mean(abs(y_test - y) ** 2)))
    # print(rmse)
    return rmse


if __name__ == "__main__":

    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(main_args)
    print("{:.2f}".format(rmse))