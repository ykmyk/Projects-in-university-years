#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X = []
        y = []
        window = 4

        for i in range(window, len(train.data) - window):
            if train.data[i].lower() in Dataset.LETTERS_NODIA:
                context = train.data[i - window : i + window + 1]
                X.append(context)
                y.append(train.target[i])

        y = np.array(y)
        # colver=saga, ngram_range=(1,5), max_feature = 40000, Accuracy: 87.56 => oversize 15MiB
        #                          (1,4).               60000            85    => oversize 12MiB
        #                          (1,4)                30000            85    => oversize 12MiB
        #                          (1,3)               100000            79.6  => accuracy in Recodex 82.5%   
        # lbfgs, l2 => 79.10
        # saga, l2 => 79.10
        # saga, l1 => 
        # saga, elasticnet =>
        #                          (1,5)                 5000            82.09  => accuracy in Recodex 83.78%, 2MiB
        #                          (1,5)                10000            83.58  => accuracy in Recodex 84.71%, 3.97MiB
        #                          (1,5)                15000            83.58  => accuracy in Recodex 85.31%, 6.2M
        #                          (1,5)                20000            85.57  => accuracy in Recodex 85.50%, 7.88MiB
        # lbfgs => 85.07
        #                          (1,5)                25000            84.58  => accuracy in Recodex 85.62%, 9.80MiB
        # saga, n_jobs=None => 85.57%
        # lbfgs, C=1.5 => 86.07%
        # lbfgs, C=2.0 => 86.07%, accuracy in Recodex 85.65%
        # saga, C=0.5 => 85.07%
        # saga, C=1.5 => 85.07%
        # saga, C=1.8 => 85.07%, accuracy in Recodex 85.80%
        # saga, C=1.8, maxiter=600 => 85.57
        # saga, C=1.8, maxiter=600, (2, 5) => 85.07
        # saga, C=2.0 => 85.57%, accuracy in Recodex 85.82%
        # saga, C=2.0, l2 => 85.57%, accuracy in Recodex 85.82%
        # - // - PLUS lexicon dictionary => 
        # saga, C=2.0, maxiter=400 (1,5)                27000            86.07  => accuracy in Recodex %, 10.6MiB
        #                          (1,6)                 5000            82.59
        #                          (1,6)                10000            84.08  
        #                          (1,6)                15000            85.07
        #                          (1,6)                20000            85.07 => Recodex 85.60%
        #                          (1,7)                20000            85.07 => Recodex 85.60%
        #                          (1,7)                25000            85.57 => Recodex over capacity
        #                          (1,8)                20000            85.57
        # saga, C=1.0 => 85.57 
        #                          (1,8)                25000            85.57 => Recodex over capacity
        
        # introduced new test data(bigger) + lexicon dict
        # +lexicon dict            (1,5)                20000            66.58  => Recodex: passed, 8.03MiB
        # +lower() dict, window 2  (1,5)                20000            82.06  => Recodex: passed, 8.03MiB, 1m 13.7s
        # +lower() dict, window 3  (1,5)                20000            84.03  
        # +lower() dict, window 4  (1,5)                20000            84.65 
        #                                               30000            85.66
        #                                               35000            84.03
        #                                               40000            84.69
        #                                               50000            84.01
        #                                               55000            84.34
        # +lower() dict, window 5  (1,5)                20000            79.89

 
        model = sklearn.pipeline.Pipeline([
            ("vec", CountVectorizer(analyzer="char", ngram_range=(1, 5), max_features=45000)),
            ("clf", sklearn.linear_model.LogisticRegression(multi_class="multinomial", solver="saga",
                                    C=2.0, max_iter=500, n_jobs=-1)),
        ])

        
        model.fit(X, y)
        # print(model)

        word_map = defaultdict(Counter)
        for wd, wn in zip(train.target.split(), train.data.split()):
            wd = wd.lower()
            wn = wn.lower()    
            word_map[wn][wd] += 1
        lexicon = {wn: max(counter, key=counter.get) for wn, counter in word_map.items()}
        model_bundle = {"model": model, "lexicon": lexicon}

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model_bundle, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        test.data
        test.target
        window = 4

        with lzma.open(args.model_path, "rb") as model_file:
            bundle = pickle.load(model_file)
        model = bundle["model"]
        lexicon = bundle.get("lexicon", {})

        # classes = all dia_able letters 
        classes = model.named_steps["clf"].classes_  # array of strings, each a single char

        # get map of character ch of non-dia letter
        def to_base(ch): return ch.translate(Dataset.DIA_TO_NODIA)

        # get the array of all base letters of dia-able letters in the same order as in the classes
        class_to_base = np.array([to_base(c) for c in classes])

        words = test.data.split()
        prediction = []
        for w in words:
            w_lower = w.translate(Dataset.DIA_TO_NODIA).lower()
            if w_lower in lexicon:
                pred = lexicon[w_lower]
                if w.istitle(): pred = pred.capitalize()
                elif w.isupper(): pred = pred.upper()
                # pred = ""
                # for i, c in enumerate(w):
                #     if c.isupper():
                #         pred += candidate[i].upper()
                #     else:
                #         pred += candidate[i]
                prediction.append(pred)
            else:
                pred_chars = []
                for i, ch in enumerate(w):
                    if ch.lower() in Dataset.LETTERS_NODIA:
                        left = max(0, i - window)
                        right = min(len(test.data), i + window + 1)
                        context = w[left:right]

                        # Get probabilities over all classes(predict_proba is sklearn built in function)
                        # returns 2D shape (n_sample, n_classes) so extract the first "sample" part
                        # looks like [0.1, 0.8, 0.1] with indexes [a, b, c] 
                        # meaning 10% to be "a" or "c" and 80% to be "b"
                        proba = model.predict_proba([context])[0]   

                        base = ch.lower()

                        # list of boolean showing 
                        # if the given letter is dia-able letter of the pairwise letter in the classes
                        # True if the given base is dia-able
                        # False if not(to be skipped)
                        # ex). 
                        # if classes = ['a', 'á', 'b', 'c', 'č'], base = 'a'
                        # then mask = [T, T, F, F, F]
                        mask = (class_to_base == base)

                        # if it is dia-able letter
                        if mask.any():
                            # take the highest probability index that is pairwise dia-able letter 
                            # in filtered array by mask
                            # np.argmax is the func. to return index
                            idx = np.argmax(proba[mask])

                            # get the index in original(non-filtered array) of the max prob.
                            # flatnonzero works as follows
                            # x = array([-2, -1,  0,  1,  2])
                            # np.flatnonzero(x) = array([0, 1, 3, 4]) <= index 0 is skipped since the value is 0
                            # make np array of indexes that the original value is nonzero

                            # [idx] indexes => pick the corresponding value 
                            abs_idx = np.flatnonzero(mask)[idx]
                            pred_letter = classes[abs_idx]
                        else:
                            # Fallback: keep the original character
                            pred_letter = ch

                        # Preserve case
                        if ch.isupper():
                            pred_letter = pred_letter.upper()

                        pred_chars.append(pred_letter)
                    else:
                        pred_chars.append(ch)
                
                prediction.append("".join(pred_chars))
        result = " ".join(prediction)
        print(result, end="")
        return result

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
