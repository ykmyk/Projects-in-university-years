#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", choices=["random", "kmeans++"], help="Initialization")
parser.add_argument("--iterations", default=20, type=int, help="Number of kmeans iterations to perform")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def plot(args: argparse.Namespace, iteration: int,
         data: np.ndarray, centers: np.ndarray, clusters: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if args.plot is not True:
        plt.gcf().get_axes() or plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Iteration {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")


def main(args: argparse.Namespace) -> np.ndarray:
    generator = np.random.RandomState(args.seed)

    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    centers = np.zeros((args.clusters, data.shape[1]))

    if args.init == "random":
        indices = generator.choice(len(data), size=args.clusters, replace=False)
        centers = data[indices]

    elif args.init == "kmeans++":
        first_index = generator.randint(len(data))
        centers[0] = data[first_index]
        chosen_indices = [first_index]

        for i in range(1, args.clusters):
            current_centers = centers[:i]
            
            dists = np.linalg.norm(data[:, np.newaxis] - current_centers, axis=2)
            
            min_dists = np.min(dists, axis=1)
            square_distances_all = min_dists ** 2

            all_indices = np.arange(len(data))
            mask = np.isin(all_indices, chosen_indices, invert=True)
            unused_points_indices = all_indices[mask]
            
            square_distances = square_distances_all[mask]

            next_idx = generator.choice(
                unused_points_indices, 
                p=square_distances / np.sum(square_distances)
            )
            
            centers[i] = data[next_idx]
            chosen_indices.append(next_idx)

    if args.plot:
        plot(args, 0, data, centers, clusters=None)

    clusters = np.zeros(len(data), dtype=int)

    for iteration in range(args.iterations):
        dists = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        clusters = np.argmin(dists, axis=1)

        new_centers = np.zeros_like(centers)
        for k in range(args.clusters):
            points_in_cluster = data[clusters == k]
            
            if len(points_in_cluster) > 0:
                new_centers[k] = np.mean(points_in_cluster, axis=0)
            else:
                new_centers[k] = centers[k]
        
        centers = new_centers

        if args.plot:
            plot(args, 1 + iteration, data, centers, clusters)

    return clusters


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    clusters = main(main_args)
    print("Cluster assignments:", clusters, sep="\n")