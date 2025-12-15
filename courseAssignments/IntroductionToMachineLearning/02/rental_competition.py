#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn.linear_model
import sklearn.model_selection
from sklearn.pipeline import Pipeline

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def findBestAlpha(X_train, X_test, y_train, y_test) -> float:
    import sys
    lambdas = np.geomspace(0.01, 10, num=500)

    best_rmse = sys.maxsize
    best_lambda = 0
    for a in lambdas:
        model = sklearn.linear_model.SGDRegressor(
            penalty = 'l2', alpha = a, eta0=0.001, max_iter=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r = float(np.sqrt(np.mean(abs(y_test - y_pred) ** 2)))
            
        if r < best_rmse:
            best_rmse = r
            best_lambda = a
    return best_lambda

def findBestLRate(X_train, X_test, y_train, y_test, a) -> float:
    import sys
    best_rmse = sys.maxsize
    best_rate = None
    rmse = sys.maxsize
    rate = None

    low, high = 0, 1.0
    mean = (low + high) / 2

    for i in range(100):
        rate_l = (low + mean) / 2
        model_l = sklearn.linear_model.SGDRegressor(
            penalty = 'l2', alpha = a, eta0 = rate_l, max_iter=500)
        model_l.fit(X_train, y_train)
        y_pred_l = model_l.predict(X_test)
        r_l = float(np.sqrt(np.mean(abs(y_test - y_pred_l) ** 2)))

        rate_h = (high + mean) / 2
        model_h = sklearn.linear_model.SGDRegressor(
            penalty = 'l2', alpha = a, eta0 = rate_h, max_iter=500)
        model_h.fit(X_train, y_train)
        y_pred_h = model_h.predict(X_test)
        
        r_h = float(np.sqrt(np.mean(abs(y_test - y_pred_h) ** 2)))

        if(r_l < r_h):
            high = mean
            rate = rate_l
            rmse = r_l
        else:
            low = mean
            rate = rate_h
            rmse = r_h
        mean = (low + high) / 2

        if(rmse < best_rmse):
            best_rmse = rmse
            best_rate = rate
        else:
            return best_rate


def preprocess(X_train, X_test):
    def is_integer_only(col : np.ndarray) -> bool:
        return np.all(col == np.round(col))
    d_size = X_train.shape[1]
    int_ind = [i for i in range(d_size) if is_integer_only(X_train[:, i])]
    rest_ind = [i for i in range(d_size) if i not in int_ind]
    
    preproc = Pipeline([
        ("ct", sklearn.compose.ColumnTransformer(
            [("ints", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", sparse_output=False), int_ind),
            ("rests", sklearn.preprocessing.StandardScaler(), rest_ind)])),
        ("poly", sklearn.preprocessing.PolynomialFeatures(2, include_bias=False))
    ])

    X_train_poly = preproc.fit_transform(X_train)
    X_test_poly = preproc.transform(X_test)

    

    return X_train_poly, X_test_poly, preproc


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        # TODO: Train a model on the given dataset and store it in `model`.
        np.random.seed(args.seed)
        train = Dataset()
        X = train.data
        y = train.target
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                                X, y, test_size=100, random_state=args.seed)
        
        X_train_p, X_test_p, preproc = preprocess(X_train, X_test)
        a = findBestAlpha(X_train_p, X_test_p, y_train, y_test)
        l_rate = findBestLRate(X_train_p, X_test_p, y_train, y_test, a)
        
        model = Pipeline([
            ("preproc", preproc),
            ("sgd", sklearn.linear_model.SGDRegressor(
                penalty = 'l2', alpha = a, eta0=l_rate, max_iter=1000))
        ])
        model.fit(X, y)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        X = test.data

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(X)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
