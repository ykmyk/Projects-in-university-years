#!/usr/bin/env python3
import os
import sys
import urllib.request
#!/usr/bin/env python3
import argparse
import lzma
import pickle
from typing import Optional

import numpy as np
import sklearn.linear_model
import sklearn.pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(
        LETTERS_DIA + LETTERS_DIA.upper(),
        LETTERS_NODIA + LETTERS_NODIA.upper()
    )

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


def should_check(ch: str) -> bool:
    return ch.lower() in Dataset.LETTERS_NODIA


def diacritize_sentence(
    model,
    dictionary: Dictionary,
    train_singletons: dict,      # word -> unique dia form (for words NOT in dictionary)
    train_variants: dict,        # word -> set of dia forms seen in training
    nd_sentence: str,
    window: int
) -> str:
    """
    Fast per-word diacritization:
      - For each word:
        * If in dictionary with a single variant: use that variant, no model calls.
        * Else if word not in dictionary but in train_singletons: use that training variant.
        * If in dictionary with multiple variants: filter variants to those
          seen in training (if available) and choose best by predict_proba.
        * If not in dictionary and not in train_singletons: char-wise fallback
          using predict_proba with base-letter restriction.
    """
    clf = model.named_steps["clf"]
    classes = clf.classes_                 # array of characters
    class_index = {c: i for i, c in enumerate(classes)}

    def to_base(ch: str) -> str:
        # Map a diacritized letter to its base; non-diacritics stay the same.
        return ch.translate(Dataset.DIA_TO_NODIA)

    # For base-letter restriction in fallback mode
    class_to_base = np.array([to_base(c) for c in classes])

    result_chars = list(nd_sentence)
    n = len(nd_sentence)
    i = 0

    while i < n:
        if not nd_sentence[i].isalpha():
            i += 1
            continue

        # find the word [start:end)
        start = i
        while i < n and nd_sentence[i].isalpha():
            i += 1
        end = i

        word = nd_sentence[start:end]
        key = word.lower()

        # --- Step 1: dictionary lookup ---
        variants = dictionary.variants.get(key)

        # If multiple dictionary variants, try to restrict to those seen in training
        if variants is not None and len(variants) > 1:
            tv = train_variants.get(key)
            if tv is not None:
                filtered = [v for v in variants if v in tv]
                if filtered:
                    variants = filtered

        # Case 1: dictionary has exactly one (possibly filtered) variant
        if variants is not None and len(variants) == 1:
            chosen = variants[0]
            # apply casing per character
            for offset, c in enumerate(chosen[:len(word)]):
                src = word[offset]
                if src.isupper():
                    c = c.upper()
                elif src.islower():
                    c = c.lower()
                result_chars[start + offset] = c
            continue

        # --- Step 2: training singleton lexicon for words NOT in dictionary ---
        if variants is None:
            ts = train_singletons.get(key)
            if ts is not None:
                chosen = ts
                for offset, c in enumerate(chosen[:len(word)]):
                    src = word[offset]
                    if src.isupper():
                        c = c.upper()
                    elif src.islower():
                        c = c.lower()
                    result_chars[start + offset] = c
                continue
        # -------------------------------------------------------------

        # For multi-variant or unknown words, we may need model predictions.
        # Collect positions of dia-able letters in this word.
        offsets = [pos for pos, ch in enumerate(word) if should_check(ch)]

        # If there is nothing to diacritize, either pick first variant or leave as is.
        if not offsets:
            if variants:
                chosen = variants[0]
                for offset, c in enumerate(chosen[:len(word)]):
                    src = word[offset]
                    if src.isupper():
                        c = c.upper()
                    elif src.islower():
                        c = c.lower()
                    result_chars[start + offset] = c
            # else: keep original word as it is
            continue

        # Build contexts and call predict_proba once per relevant character.
        contexts = []
        for off in offsets:
            idx = start + off
            l = max(0, idx - window)
            r = min(n, idx + window + 1)
            context = nd_sentence[l:r]
            contexts.append(context)

        proba_list = model.predict_proba(contexts)  # shape: (len(offsets), n_classes)

        # Case 2: word is in dictionary with (possibly filtered) multiple variants → choose best variant
        if variants:
            best_score = -np.inf
            best_variant = None

            for var in variants:
                if len(var) != len(word):
                    continue
                score = 0.0
                for prob, off in zip(proba_list, offsets):
                    target_char = var[off]
                    idx_class = class_index.get(target_char)
                    if idx_class is None:
                        p = 1e-12
                    else:
                        p = prob[idx_class]
                    score += np.log(p + 1e-12)
                if score > best_score:
                    best_score = score
                    best_variant = var

            chosen = best_variant if best_variant is not None else variants[0]

            # apply casing per character
            for offset, c in enumerate(chosen[:len(word)]):
                src = word[offset]
                if src.isupper():
                    c = c.upper()
                elif src.islower():
                    c = c.lower()
                result_chars[start + offset] = c

        else:
            # Case 3: word is not in dictionary and not in train_singletons → char-wise fallback
            for prob, off in zip(proba_list, offsets):
                ch = word[off]
                base = ch.lower()
                mask = (class_to_base == base)

                if mask.any():
                    masked_probs = prob[mask]
                    idx_in_mask = np.argmax(masked_probs)
                    abs_idx = np.flatnonzero(mask)[idx_in_mask]
                    pred_char = classes[abs_idx]
                else:
                    pred_char = ch

                if ch.isupper():
                    pred_char = pred_char.upper()
                elif ch.islower():
                    pred_char = pred_char.lower()

                result_chars[start + off] = pred_char

        # next word

    return "".join(result_chars)


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        window = 4

        dictionary = Dictionary()

        # --- Build train_singletons for words NOT in dictionary ---
        singleton_tmp = {}
        # --- Build train_variants for ALL words (including those in dictionary) ---
        variants_tmp = {}

        train_nodia_words = train.data.split()
        train_dia_words = train.target.split()

        for nd, dia in zip(train_nodia_words, train_dia_words):
            key = nd.lower()

            # For train_variants: collect all dia-forms seen in training
            if key not in variants_tmp:
                variants_tmp[key] = {dia}
            else:
                variants_tmp[key].add(dia)

            # For train_singletons: only words NOT in dictionary
            if key in dictionary.variants:
                continue
            if key not in singleton_tmp:
                singleton_tmp[key] = dia
            else:
                # if we ever see a different variant, mark as ambiguous
                if singleton_tmp[key] is not None and singleton_tmp[key] != dia:
                    singleton_tmp[key] = None

        # keep only truly unambiguous words
        train_singletons = {k: v for k, v in singleton_tmp.items() if v is not None}
        # freeze variant sets
        train_variants = {k: frozenset(v) for k, v in variants_tmp.items()}

        X = []
        y = []

        for i in range(window, len(train.data) - window):
            if train.data[i].lower() in Dataset.LETTERS_NODIA:
                context = train.data[i - window: i + window + 1]
                X.append(context)
                y.append(train.target[i])

        y = np.array(y)

        model = sklearn.pipeline.Pipeline([
            ("vec", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), max_features=60000)),
            ("clf", sklearn.linear_model.LogisticRegression(
                multi_class="multinomial",
                solver="saga",
                C=5.0,
                max_iter=500,
                n_jobs=-1
            )),
        ])

        model.fit(X, y)

        clf = model.named_steps["clf"]
        clf.coef_ = clf.coef_.astype(np.float16)
        clf.intercept_ = clf.intercept_.astype(np.float16)

        # save bundle (model + train_singletons + train_variants)
        bundle = {
            "model": model,
            "train_singletons": train_singletons,
            "train_variants": train_variants,
        }
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(bundle, model_file)

    else:
        # Use the model and return test set predictions.
        window = 4
        dictionary = Dictionary()

        with lzma.open(args.model_path, "rb") as model_file:
            bundle = pickle.load(model_file)
        model = bundle["model"]
        train_singletons = bundle["train_singletons"]
        train_variants = bundle["train_variants"]

        pred_lines = []
        with open(args.predict, "r", encoding="utf-8") as f:
            for line in f:
                nd_sentence = line.rstrip("\n")
                if not nd_sentence:
                    pred_lines.append("")
                    continue

                sentence = diacritize_sentence(
                    model,
                    dictionary,
                    train_singletons,
                    train_variants,
                    nd_sentence,
                    window
                )
                pred_lines.append(sentence)

        prediction = "\n".join(pred_lines)
        print(prediction, end="")
        return prediction


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
