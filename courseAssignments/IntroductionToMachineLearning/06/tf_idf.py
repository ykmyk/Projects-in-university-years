#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import re
import sys
import urllib.request
import warnings

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

# Deliberately ignore the liblinear-is-deprecated-for-multiclass-classification warning.
warnings.filterwarnings("ignore", "Using the 'liblinear' solver.*is deprecated.")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=79, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


# Regular expression for terms: maximal sequences of at least one word character \w
_TERM_RE = re.compile(r"\w+")


def tokenize(text):
    # Do NOT lowercase, to match the course reference implementation.
    return [m.group(0) for m in _TERM_RE.finditer(text)]


def build_vocabulary(train_texts):
    from collections import Counter

    total_counts = Counter()
    for doc in train_texts:
        tokens = tokenize(doc)
        total_counts.update(tokens)

    # Keep only terms with corpus frequency >= 2
    terms = [t for t, c in total_counts.items() if c >= 2]

    # Sort to have deterministic indices
    terms.sort()
    vocab = {term: idx for idx, term in enumerate(terms)}
    return vocab


def compute_features(texts, vocab, use_tf, idf=None):
    n_docs = len(texts)
    n_terms = len(vocab)
    X = np.zeros((n_docs, n_terms), dtype=np.float64)

    # For computing TF, we first fill in raw counts; later normalize.
    for i, doc in enumerate(texts):
        tokens = tokenize(doc)
        if not tokens:
            continue
        # Count per-document
        from collections import Counter
        counts = Counter(tokens)
        if use_tf:
            # Store counts; we normalize after.
            for term, cnt in counts.items():
                idx = vocab.get(term)
                if idx is not None:
                    X[i, idx] = float(cnt)
        else:
            # Binary: just mark presence
            for term in counts.keys():
                idx = vocab.get(term)
                if idx is not None:
                    X[i, idx] = 1.0

    if use_tf:
        row_sums = X.sum(axis=1, keepdims=True)  # shape (n_docs, 1)

        # Avoid division by zero: for empty documents, keep the row as all zeros.
        row_sums[row_sums == 0] = 1.0

        X /= row_sums

    if idf is not None:
        # Multiply each feature (column) by IDF.
        X *= idf

    return X


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    vocab = build_vocabulary(train_data)
    print("Number of unique terms with at least two occurrences: {}".format(len(vocab)))

    n_train = len(train_data)
    n_terms = len(vocab)
    df = np.zeros(n_terms, dtype=np.int32)

    for doc in train_data:
        tokens = tokenize(doc)
        if not tokens:
            continue
        # unique terms in this document
        seen = set()
        for term in tokens:
            idx = vocab.get(term)
            if idx is not None:
                seen.add(idx)
        for idx in seen:
            df[idx] += 1

    idf = None
    if args.idf:
        idf = np.log(n_train / (df.astype(np.float64) + 1.0))

    X_train = compute_features(train_data, vocab, use_tf=args.tf, idf=idf)
    X_test = compute_features(test_data, vocab, use_tf=args.tf, idf=idf)

    clf = sklearn.linear_model.LogisticRegression(
        solver="liblinear",
        C=10_000,
        random_state=args.seed,
        max_iter=1000,
    )
    clf.fit(X_train, train_target)

    test_pred = clf.predict(X_test)
    f1_score = sklearn.metrics.f1_score(test_target, test_pred, average="macro")

    return 100 * f1_score


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(main_args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(main_args.tf, main_args.idf, f1_score))
