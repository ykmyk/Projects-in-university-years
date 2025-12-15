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
import sklearn
import sklearn.neural_network
import sklearn.model_selection
import scipy.ndimage as ndi

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="miniaturization.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


# The following class modifies `MLPClassifier` to support full categorical distributions
# on input, i.e., each label should be a distribution over the predicted classes.
# During prediction, the most likely class is returned, but similarly to `MLPClassifier`,
# the `predict_proba` method returns the full distribution.
# Note that because we overwrite a private method, it is guaranteed to work only with
# scikit-learn 1.7.2, but it will most likely work with any 1.7.*.
class MLPFullDistributionClassifier(sklearn.neural_network.MLPClassifier):
    class FullDistributionLabels:
        y_type_ = "multiclass"

        def fit(self, y):
            return self

        def transform(self, y):
            return y

        def inverse_transform(self, y):
            return np.argmax(y, axis=-1)

    def _validate_input(self, X, y, incremental, reset):
        X, y = sklearn.utils.validation.validate_data(
            self, X, y, multi_output=True, dtype=(np.float64, np.float32), reset=reset)
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = self.FullDistributionLabels()
            self.classes_ = np.arange(y.shape[1])
        return X, y
    
def augment_images(X: np.ndarray, y: np.ndarray, factor: int = 1,
                   # degrees, max absolute rotation
                   rotate_range: float = 10.0,
                   # pixels, max absolute shift in x/y
                   shift_range: float = 2.0,
                   # gaussian noise std (in same scale as X, i.e. 0..1)
                   noise_std: float = 0.0,
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    
    rng = np.random.RandomState(seed)

    if X.ndim == 2 and X.shape[1] == 784:
        X_imgs = X.reshape((-1, 28, 28)).astype(np.float32)
    elif X.ndim == 3 and X.shape[1:] == (28, 28):
        X_imgs = X.astype(np.float32)
    else:
        raise ValueError("augment_images expects X of shape (N,784) or (N,28,28)")

    N = X_imgs.shape[0]
    if factor <= 0:
        return X.reshape((N, 784)).astype(np.float32), y.copy()

    aug_list = [X] 

    for k in range(factor):
        Xk = np.empty_like(X_imgs)
        angles = rng.uniform(-rotate_range, rotate_range, size=N)
        shifts_x = rng.uniform(-shift_range, shift_range, size=N)
        shifts_y = rng.uniform(-shift_range, shift_range, size=N)
        scales = rng.uniform(0.85, 1.15, size=N)
        shears = rng.uniform(-10, 10, size=N)

        for i in range(N):
            angle_rad = np.deg2rad(angles[i])
            shear_rad = np.deg2rad(shears[i])
            scale = scales[i]

            c, s = np.cos(angle_rad), np.sin(angle_rad)
            rot_matrix = np.array([[c, -s], [s, c]])
            scale_matrix = np.array([[scale, 0], [0, scale]])
            shear_matrix = np.array([[1, -np.sin(shear_rad)], [0, np.cos(shear_rad)]])
            matrix = rot_matrix @ scale_matrix @ shear_matrix

            center = 13.5
            offset = center - matrix @ [center, center] + [shifts_y[i], shifts_x[i]]
            img = X_imgs[i]
            Xk[i] = ndi.affine_transform(img, matrix, offset=offset, order=1, mode='constant', cval=0.0)
        aug_list.append(Xk.reshape(N, 784))

    X_all = np.concatenate(aug_list, axis=0)
    y_all = np.tile(y, (factor + 1, ))

    return X_all, y_all




def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()
        # train = Dataset("local_train.npz")

        X = train.data.astype(np.float32) / 255.0
        y = train.target

        X_aug, y_aug = augment_images(X, y, factor = 30, rotate_range= 10.0, shift_range= 2.0, noise_std=0.02,
                                      seed= args.seed)
        
        perm = np.random.RandomState(args.seed).permutation(X_aug.shape[0])
        X_aug = X_aug[perm]
        y_aug = y_aug[perm]

        num_teachers = 7
        teacher_preds = []

        for i in range(num_teachers):
            t_seed = args.seed + i
            teacher = sklearn.neural_network.MLPClassifier(
                hidden_layer_sizes=(300, 300),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=25,
                random_state=t_seed,
                verbose=False,
                early_stopping=True,
                n_iter_no_change=5,
                validation_fraction=0.1,
            )

            teacher.fit(X_aug, y_aug)

            teacher_probs = teacher.predict_proba(X_aug).astype(np.float32)
            teacher_preds.append(teacher_probs)

        avg_teacher_probs = np.mean(teacher_preds, axis=0).astype(np.float32)
        student = MLPFullDistributionClassifier(
                hidden_layer_sizes=(200,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=50,
                random_state=args.seed,
                verbose=False,
                early_stopping=True,
                n_iter_no_change=5,
                validation_fraction=0.01,
            )
        num_classes = 10
        one_hot = np.eye(num_classes, dtype=np.float32)[y_aug]

        lambda_soft = 0.7
        soft_labels = (1 - lambda_soft) * one_hot + lambda_soft * avg_teacher_probs

        student.fit(X_aug, soft_labels)

        mlp = student
        mlp._optimizer = None
        mlp._best_coefs = None
        mlp._best_intercepts = None

        for i in range(len(mlp.coefs_)):
            mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)):
            mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        model = mlp

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        X_test = test.data.astype(np.float32) / 255.0
        predictions = model.predict(X_test)

        return predictions
    
if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)