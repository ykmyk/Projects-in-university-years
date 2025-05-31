# version 3
# tried to optimize more with unit clause
# it worked!!!(probably)

# Install package python-sat !!!
from pysat.solvers import Minisat22
from pysat.card import CardEnc
# # like iterator in Java!! useful!
# from itertools import combinations
from time import time

def solve_3partition(numbers):
    n = len(numbers) // 3                            
    target_sum = sum(numbers) // n                   

    numbers = sorted(numbers)
    number_set = set(numbers)
    triplets = []

    # minimize the number of the triplets to generate
    MAX_CANDIDATES = n * 50
    triplets = []
    seen = set()

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            a = numbers[i]
            b = numbers[j]
            c = target_sum - a - b
            if c in number_set and c != a and c != b:
                t = tuple(sorted((a, b, c)))
                if t not in seen:
                    triplets.append(t)
                    seen.add(t)
                    if len(triplets) >= MAX_CANDIDATES:
                        break
        if len(triplets) >= MAX_CANDIDATES:
            break
    

    # Create SAT variables for each triplet
    # map each triplet to a variable: var_id = i + 1 (1-based indexing for SAT solvers)
    var_for_triplet = {}
    for i in range(len(triplets)):
        var_for_triplet[i] = i + 1

    solver = Minisat22()

    # For each number, ensure it appears in exactly one selected triplet
    for num in numbers:
        vars_with_num = []
        for i in range(len(triplets)):
            triplet = triplets[i]
            if num in triplet:
                vars_with_num.append(var_for_triplet[i])
        
        if len(vars_with_num) == 0:
            continue

        # Optimize with unit clause
        if len(vars_with_num) == 1:
            solver.add_clause([vars_with_num[0]])  # unit clause
        else:
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

    model = solver.get_model()
    selected_vars = set(lit for lit in model if lit > 0)

    solution = [
        list(triplets[i])
        for i, var in var_for_triplet.items()
        if var in selected_vars
    ]
    
    if len(solution) != n:
        return []
        
    return solution