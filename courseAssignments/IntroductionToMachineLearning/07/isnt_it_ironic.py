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

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import sklearn

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

def has_inner_uppercase(text: str) -> bool:
    """Return True if any token contains an uppercase letter
    after its first character."""
    for token in text.split():
        # Skip empty tokens just in case
        if len(token) <= 1:
            continue
        # Check characters from index 1 onwards
        for ch in token[1:]:
            if ch.isupper():
                return True
    return False

from typing import Iterable, List
def augment_with_caps_feature(texts):
    """Append a special token describing the presence of inner-uppercase letters."""
    result = []
    for t in texts:
        marker = "<INNER_CAP_1>" if has_inner_uppercase(t) else "<INNER_CAP_0>"
        result.append(t + " " + marker)
    return result

def is_shouty(text, min_ratio: float = 0.3):
    """True if uppercase letters make up at least min_ratio of all letters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    upper_count = sum(1 for ch in letters if ch.isupper())
    return (upper_count / len(letters)) >= min_ratio


def many_exclamations(text, threshold: int = 2):
    return text.count("!") >= threshold


def many_questions(text, threshold: int = 2):
    return text.count("?") >= threshold


def augment_texts(texts):
    result = []
    for t in texts:
        markers = []

        # Inner uppercase
        markers.append("<INNER_CAP_1>" if has_inner_uppercase(t) else "<INNER_CAP_0>")

        # Shoutiness
        if is_shouty(t):
            markers.append("<SHOUTY>")

        # Multiple exclamation/question marks
        if many_exclamations(t):
            markers.append("<MANY_EXCL>")
        if many_questions(t):
            markers.append("<MANY_QUEST>")

        if markers:
            t = t + " " + " ".join(markers)

        result.append(t)
    return result

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        model = Pipeline([
            ("count", CountVectorizer(
                tokenizer=str.split,
                preprocessor=None,
                lowercase=False,
                ngram_range=(1, 1),
                min_df=1,
                max_df=0.8,
            )),
            # ("count", TfidfVectorizer(
            #     tokenizer=str.split,
            #     preprocessor=None,
            #     ngram_range=(1, 1), 
            #     lowercase=False, 
            #     min_df=1, 
            #     max_df=0.8)),

            # ("tfidf", TfidfTransformer(
            #     norm="l1",
            #     sublinear_tf=False,
            #     smooth_idf=False,
            #     )),
            # ("classifier", sklearn.svm.LinearSVC(
            #     C=3.0,
            #     class_weight="balanced",
            #     random_state=args.seed,
            # ))
            ("classifier", sklearn.naive_bayes.MultinomialNB())
        ])
        train.data = augment_texts(train.data)
        model.fit(train.data, train.target)
        # X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(
        #     train.data, train.target, test_size=300, random_state=args.seed)
        # X_train = augment_texts(X_train)
        # model.fit(X_train, y_train)
        # X_dev = augment_texts(X_dev)
        # pred_dev = model.predict(X_dev)
        # print("F1 on dev split:", f1_score(y_dev, pred_dev))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        test.data = augment_texts(test.data)
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

