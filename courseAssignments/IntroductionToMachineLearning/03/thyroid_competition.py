#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

import sklearn.linear_model
import sklearn.model_selection
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X = train.data
        y = train.target
        # train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        #                                         X, y, test_size=100, random_state=args.seed)


        # binary_features = list(range(15))
        # real_features = list(range(15, 21))
        
        # preprocessor = ColumnTransformer([
        # ("real", MinMaxScaler(), real_features),
        # ("bin", "passthrough", binary_features)
        # ])

        # scaler = sklearn.preprocessing.ColumnTransformer
        pipe = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.MinMaxScaler()),
            ("poly", sklearn.preprocessing.PolynomialFeatures(degree=2)),
            ("logireg", sklearn.linear_model.LogisticRegression(random_state=args.seed))
        ])

        skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)

        params = {
            'poly__degree':[1, 2],
            'logireg__C':[0.0001, 0.001, 0.01, 1, 10, 100, 1000],
            'logireg__solver':('lbfgs', 'sag', 'liblinear', 'newton-cg', 'newton-cholesky', 'liblinear'),
            'logireg__penalty':('l1', 'l2'),
            'logireg__max_iter':[1000]
        }
        

        model = sklearn.model_selection.GridSearchCV(estimator=pipe, param_grid=params, cv=skf)
        model.fit(X, y)
        # test_accuracy = model.score(test_data, test_target)

        # print("Test accuracy: {:.2f}%".format(test_accuracy * 100))
        # print(model.best_score_)
        # print(model.cv_results_["std_test_score"][model.best_index_])
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
