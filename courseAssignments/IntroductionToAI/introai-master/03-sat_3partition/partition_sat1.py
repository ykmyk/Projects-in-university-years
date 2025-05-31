# version 1
# couldn't solve test 5 on time

# Install package python-sat !!!
from pysat.solvers import Minisat22
from pysat.card import CardEnc
# like iterator in Java!! useful!
from itertools import combinations
from time import time

def solve_3partition(numbers):
    n = len(numbers) // 3                            
    target_sum = sum(numbers) // n                   

    # Generate all valid triplets
    # keep the one only matches with target_sum
    # using itertools => new but it is very useful!
    triplets = []
    for triplet in combinations(numbers, 3):
        if sum(triplet) == target_sum:
            triplets.append(triplet)

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

