#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def ReLU(x):
        return np.maximum(x, 0.0)

    def ReLU_g(x):
        return (x > 0).astype(x.dtype)

    def softMax(x):
        if x.ndim == 1:
            exp_ = np.exp(x - np.max(x))
            return exp_ / np.sum(exp_)
        else:
            exp_ = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_ / np.sum(exp_, axis=1, keepdims=True)

    
    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where $ReLU(x) = max(x, 0)$, and an output layer with softmax
        # activation.
            
        # The value of the hidden layer is computed as `ReLU(inputs @ weights[0] + biases[0])`.
        h_in = inputs @ weights[0] + biases[0]
        h_out = ReLU(h_in)
        # The value of the output layer is computed as `softmax(hidden_layer @ weights[1] + biases[1])`.
        o_in = h_out @ weights[1] + biases[1]
        output = softMax(o_in)
        
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        # raise NotImplementedError()

        return h_in, h_out, o_in, output


    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        b_size = args.batch_size
        for i in range(0, train_data.shape[0], b_size):
            indices= permutation[i : i + b_size]
            X_batch = train_data[indices]
            t_batch = train_target[indices]
            h_in, h_out, o_in, output = forward(X_batch)
            
            # The gradient used in SGD has now four parts, gradient of `weights[0]` and `weights[1]`
            # and gradient of `biases[0]` and `biases[1]`.
            #
            # You can either compute the gradient directly from the neural network formula,
            # i.e., as a gradient of $-log P(target | data)$, or you can compute
            # it step by step using the chain rule of derivatives, in the following order:
            # - compute the derivative of the loss with respect to *inputs* of the
            #   softmax on the last layer,
            b = t_batch.shape[0]
            g = output.copy()
            g[np.arange(b), t_batch] -= 1
            g = g / b
            
            # - compute the derivative with respect to `weights[1]` and `biases[1]`,
            gw1 = h_out.T @ g
            gb1 = np.sum(g, axis=0)
            
            # - compute the derivative with respect to the hidden layer output,
            deriv_out = g @ weights[1].T
                
            # - compute the derivative with respect to the hidden layer input,
            deriv_in = deriv_out * ReLU_g(h_in)
                
            # - compute the derivative with respect to `weights[0]` and `biases[0]`.
            gw0 = X_batch.T @ deriv_in
            gb0 = np.sum(deriv_in, axis=0)

            weights[0] -= args.learning_rate * gw0
            weights[1] -= args.learning_rate * gw1
            biases[0] -= args.learning_rate * gb0
            biases[1] -= args.learning_rate * gb1
            
        _, _, _, train_prob = forward(train_data)
        train_pred = np.argmax(train_prob, axis=1)
        train_accuracy = np.mean(train_pred == train_target)

        _, _, _, test_probs = forward(test_data)
        test_pred = np.argmax(test_probs, axis=1)
        test_accuracy = np.mean(test_pred == test_target)

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(main_args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")
