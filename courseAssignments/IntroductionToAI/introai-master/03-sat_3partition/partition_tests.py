#!/usr/bin/env python3

import sys
import random
sys.path.append("..")
# import check_versions
from prettytable import PrettyTable
from time import time
from partition_sat import solve_3partition

def verify_3partition(sorted, n, total, partition):
    if not isinstance(partition, list) or len(partition) != n:
        return (False, f"Partition must be a list of length {n}.")
    
    for t in partition:
        if len(t) != 3:
            return (False, f"Every triplet must contains exactly three elements")
        if sum(t) != total:
            return (False, f"Incorrect sum in a triplet.")
        
    used = [ x for t in partition for x in t ]
    used.sort()
    if used != sorted:
        return (False, f"Every integer must occur in exactly one triplet.")

    return (True, "Correct")

def partition_test(n, total, seed):
    rng = random.Random(seed)
    min = int(total/4)
    max = int(total/2)
    numbers = set()
    while len(numbers) < 3*n:
        a = rng.randrange(min, max)
        b = rng.randrange(min, max)
        c = total - a - b
        if min <= c and c <= max and a != b and a != c and b != c and not a in numbers and not b in numbers and not c in numbers:
            numbers.update([a,b,c])

    sorted = list(numbers)
    sorted.sort()
    partition = solve_3partition(sorted)
    return verify_3partition(sorted, n, total, partition)

def partition_dataset(dataset):
    for d in dataset:
        result = partition_test(*d)
        if not result[0]:
            return result
    return (True, "Correct")

def main():
    dataset_one = [
        (4, 100, 3),
        (5, 100, 10),
        (6, 110, 7),
        (7, 1000000, 8),
    ]
    dataset_two = [
        (10, 200, 1),
        (12, 250, 10),
        (14, 300, 8),
        (15, 1000000, 7),
    ]
    dataset_three = [
        (20, 450, 45),
        (30, 600, 43),
        (40, 850, 11),
        (50, 1000000, 8),
    ]
    dataset_four = [
        (100, 2100, 41),
        (150, 3100, 77),
        (200, 4100, 12),
    ]
    dataset_five = [
        (250, 5000, 45),
        (1000, 10000000, 7),
        (1000, 5000000, 9),
        (3000, 1000000000, 18),
        (3000, 50000000, 15),
    ]

    tests = {
            "one": dataset_one,
            "two": dataset_two,
            "three": dataset_three,
            "four": dataset_four,
            "five": dataset_five,
    }

    if len(sys.argv) == 1:
        results = PrettyTable(["Test name", "Points", "Your time [s]", "Time limit on recodex [s]", "Evaluation"])
        for name in tests:
            print("Running test", name)
            start_time = time()
            status, msg = partition_dataset(tests[name])
            running_time = time() - start_time
            print(msg)
            print()
            results.add_row([name, 3, running_time, 60, msg])
        print(results)
    else:
        name = sys.argv[1]
        if name in tests:
            status, msg = partition_dataset(tests[name])
            print(msg)
        else:
            print("Unknown test", name)

"""
To run all tests, run the command
$ python3 partition_tests.py

To run a test NAME, run the command
$ python3 partition_tests.py NAME
"""
if __name__ == "__main__":
    main()
