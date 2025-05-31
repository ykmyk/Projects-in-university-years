# version 2
# solved but may be too late?
# five   |   3    |   123.89233112335205  |             60            | Partition must be a list of length 250.

# Install package python-sat !!!
from pysat.solvers import Minisat22
from pysat.card import CardEnc
# like iterator in Java!! useful!
from itertools import combinations
from time import time

def solve_3partition(numbers):
    n = len(numbers) // 3                            
    target_sum = sum(numbers) // n                   

    numbers = sorted(numbers)
    number_set = set(numbers)
    triplets = []

    # minimize the number of the triplets to generate
    # Try all pairs (a, b) and compute c = target_sum - a - b
    # If c exists in the list and is different from a and b, we accept it
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            a = numbers[i]
            b = numbers[j]
            c = target_sum - a - b
            if c in number_set and c != a and c != b:
                # Make sure triplet is unique and in sorted order
                t = tuple(sorted((a, b, c)))
                if t[0] in (a, b) and t[1] in (a, b):  # Ensure a & b are original ones
                    triplets.append(t)


    # Create SAT variables for each triplet
    # map each triplet to a variable: var_id = i + 1 (1-based indexing for SAT solvers)
    var_for_triplet = {}
    for i in range(len(triplets)):
        var_for_triplet[i] = i + 1

    solver = Minisat22()

    # For each number, ensure it appears in exactly one selected triplet
    for num in numbers:
        # store all triplets that has 
        vars_with_num = []
        for i in range(len(triplets)):
            triplet = triplets[i]
            if num in triplet:
                vars_with_num.append(var_for_triplet[i])

        # Add a cardinality constraint!!
        # exactly one of these variables must be True
        card = CardEnc.equals(lits=vars_with_num, bound=1, encoding=1)
        for clause in card.clauses:
            solver.add_clause(clause)

    # Ensure exactly n triplets are selected
    all_triplet_vars = list(var_for_triplet.values())
    card = CardEnc.equals(lits=all_triplet_vars, bound=n, encoding=1)
    for clause in card.clauses:
        solver.add_clause(clause)

    # Solve the SAT problem
    if not solver.solve():
        return []  # No solution found

    # else, SAT solver will return a list of literals(solution for the task)
    model = solver.get_model()                        

    # Extract the selected triplets from the model
    solution = [
        list(triplets[i])
        for i, var in var_for_triplet.items()
        if model[var - 1] > 0                         # SAT variables are 1-based
    ]

    return solution

