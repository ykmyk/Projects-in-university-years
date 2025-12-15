#!/usr/bin/env python3
import argparse
import dataclasses

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=100, type=int, help="Bootstrap samples")
parser.add_argument("--data_size", default=1000, type=int, help="Data set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.


class ArtificialData:
    @dataclasses.dataclass
    class Sentence:
        """ Information about a single dataset sentence."""
        gold_edits: int  # Number of required edits to be performed (for example in grammar error correction).
        predicted_edits: int  # Number of edits predicted by a model.
        predicted_correct: int  # Number of correct edits predicted by a model.
        human_rating: int  # Human rating of the model prediction.

    def __init__(self, args: argparse.Namespace):
        generator = np.random.RandomState(args.seed)

        self.sentences = []
        for _ in range(args.data_size):
            gold = generator.poisson(2)
            correct = generator.randint(gold + 1)
            predicted = correct + generator.poisson(0.5)
            human_rating = max(0, int(100 - generator.uniform(5, 8) * (gold - correct)
                                      - generator.uniform(8, 13) * (predicted - correct)))
            self.sentences.append(self.Sentence(gold, predicted, correct, human_rating))


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Create the artificial data.
    data = ArtificialData(args)

    # Create `args.bootstrap_samples` bootstrapped samples of the dataset by
    # sampling sentences of the original dataset, and for each compute
    # - average of human ratings,
    # - TP, FP, FN counts of the predicted edits.
    human_ratings, predictions = [], []
    generator = np.random.RandomState(args.seed)
    for _ in range(args.bootstrap_samples):
        # Bootstrap sample of the dataset.
        sentences = generator.choice(data.sentences, size=len(data.sentences), replace=True)

        ratings = [s.human_rating for s in sentences]
        ave_rating = np.mean(ratings)
        human_ratings.append(ave_rating)


        # TODO: Compute TP, FP, FN counts of predicted edits in `sentences`
        # and append them to `predictions`.
        tp = 0.0
        fp = 0.0
        fn = 0.0
        for s in sentences:
            tp += s.predicted_correct
            fp += s.predicted_edits - s.predicted_correct
            fn += s.gold_edits - s.predicted_correct
        predictions.append((tp, fp, fn))

    human_ratings_arr = np.array(human_ratings, dtype=float)
    # Compute Pearson correlation between F_beta score and human ratings
    # for betas between 0 and 2.
    betas, correlations = [], []
    for beta in np.linspace(0, 2, 201):
        betas.append(beta)

        # TODO: For each bootstrap dataset, compute the F_beta score using
        # the counts in `predictions` and then manually compute the Pearson
        # correlation between the computed scores and `human_ratings`. Append
        # the result to `correlations`.
        beta2 = beta * beta
        f_betas = []
        for tp, fp, fn in predictions:
            tp_f = float(tp)
            fp_f = float(fp)
            fn_f = float(fn)
            f_denom = (1.0 + beta2) * tp_f + beta2 * fn_f + fp_f
            if f_denom == 0:
                F_beta = 0.0
            else:
                F_beta = (1.0 + beta2) * tp_f / f_denom

            
            f_betas.append(F_beta)

        f_betas_arr = np.array(f_betas, dtype=float)

        x = f_betas_arr
        y = human_ratings_arr

        x_mean = x.mean()
        y_mean = y.mean()

        x_centered = x - x_mean
        y_centered = y - y_mean

        numerator = np.sum(x_centered * y_centered)
        denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))

        if denom == 0.0:
            corr = 0.0
        else:
            corr = numerator / denom


        correlations.append(corr)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(betas, correlations)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"Pearson correlation of $F_\beta$-score and human ratings")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    # TODO: Assign the highest correlation to `best_correlation` and
    # store corresponding beta to `best_beta`.
    correlations_arr = np.array(correlations, dtype=float)
    best_index = int(np.argmax(correlations_arr))
    best_beta = float(betas[best_index])
    best_correlation = float(correlations_arr[best_index])

    return best_beta, best_correlation


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    best_beta, best_correlation = main(main_args)

    print("Best correlation of {:.3f} was found for beta {:.2f}".format(
        best_correlation, best_beta))
