from pysat.formula import CNF
from pysat.solvers import Solver
from collections import defaultdict

def solve_3partition(numbers):
    n = len(numbers) // 3
    target_sum = sum(numbers) // n

    triplet_vars = {}  # var_id -> (a, b, c)
    involved = defaultdict(list)
    var_id = 1

    # 1. Generate valid triplets and assign SAT variables
    num_len = len(numbers)
    for i in range(num_len):
        a = numbers[i]
        for j in range(i + 1, num_len):
            b = numbers[j]
            for k in range(j + 1, num_len):
                c = numbers[k]
                if a + b + c == target_sum:
                    triplet_vars[var_id] = (a, b, c)
                    involved[a].append(var_id)
                    involved[b].append(var_id)
                    involved[c].append(var_id)
                    var_id += 1

    cnf = CNF()

    # 2. Each number appears in exactly one triplet
    for num in numbers:
        vars_with_num = involved[num]

        # At least one
        if vars_with_num:
            cnf.append(vars_with_num)

        # Unit clause optimization
        if len(vars_with_num) == 1:
            cnf.append([vars_with_num[0]])
        else:
            for i in range(len(vars_with_num)):
                for j in range(i + 1, len(vars_with_num)):
                    cnf.append([-vars_with_num[i], -vars_with_num[j]])

    # 3. Solve
    with Solver(bootstrap_with=cnf.clauses) as solver:
        if solver.solve():
            model = set(solver.get_model())

            used = set()
            result = []

            for var in model:
                if var > 0 and var in triplet_vars:
                    triplet = triplet_vars[var]
                    if all(num not in used for num in triplet):
                        result.append(list(triplet))
                        used.update(triplet)
                        if len(result) == n:
                            break
            return result
        else:
            return []
