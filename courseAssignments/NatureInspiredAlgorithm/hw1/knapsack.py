import random
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--mut_prob", help="Mutation Probability", type=float, default=0.6)
parser.add_argument("--input_file", help="Input File", type=str, required=True)
args = parser.parse_args()

def load_input(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    n, capacity = map(int, lines[0].split())
    items = [tuple(map(int, line.split())) for line in lines[1:]]
    return n, capacity, items

n, W, ITEMS = load_input(args.input_file)
MAX_GEN = 1000
POP_SIZE = 50
IND_LEN = n
CX_PROB = 0.8
MUT_PROB = args.mut_prob 
MUT_FLIP_PROB = 1/IND_LEN


# onemax problem
def fitness(ind):
    total_weight = total_value = 0
    for gene, (value, weight) in zip(ind, ITEMS):
        if gene:
            total_value += value
            total_weight += weight
    return total_value if total_weight <= W else 0

def create_random_population():
    population = []

    # Add (POP_SIZE - 1) random individuals
    for _ in range(POP_SIZE - 1):
        individual = []
        for _ in range(IND_LEN):
            gene = 1 if random.random() < 0.003 else 0  # very low chance
            individual.append(gene)
        population.append(individual)

    # Add one valid individual manually: empty knapsack (always valid, fitness 0)
    population.append([0] * IND_LEN)

    return population



def select(pop, fits):
    return random.choices(pop, weights=fits, k=POP_SIZE)

def crossover(pop):
    off = []
    for p1, p2 in zip(pop[::2], pop[1::2]):
        if random.random() < CX_PROB:
            point = random.randrange(0, IND_LEN)
            o1 = p1[:point] + p2[point:]
            o2 = p2[:point] + p1[point:]
            off.append(o1)
            off.append(o2)
        else:
            off.append(p1[:])
            off.append(p2[:])

    return off

def mutation(pop):
    off = []
    for p in pop:
        if random.random() < MUT_PROB:
            o = [1-i if random.random() < MUT_FLIP_PROB else i for i in p]
            off.append(o)
        else:
            off.append(p[:])
    return off

def evolution():
    log = []
    pop = create_random_population()
    for gen in range(MAX_GEN):
        fits = [fitness(ind) for ind in pop]
        valid_fits = [f for f in fits if f > 0]
        print(f"Generation {gen}: valid={len(valid_fits)} max={max(fits)}")

        # Fix: if all individuals are invalid (fitness = 0), set uniform weights
        if sum(fits) == 0:
            fits = [1 for _ in pop]

        log.append(max(fits))
        mating_pool = select(pop, fits)
        off = crossover(mating_pool)
        off = mutation(off)
        off[0] = max(pop, key=fitness)  # Elitism
        pop = off[:]
    
    return pop, log



def save_best_solution(best_ind, filename="best_solution.txt"):
    total_value = total_weight = 0
    selected_items = []

    for i, gene in enumerate(best_ind):
        if gene:
            value, weight = ITEMS[i]
            total_value += value
            total_weight += weight
            selected_items.append((i, value, weight))

    print(f"Selected {len(selected_items)} items. Total weight = {total_weight}, total value = {total_value}")

pop, log = evolution()
best_ind = max(pop, key=fitness)
print("Best fitness:", fitness(best_ind))
save_best_solution(best_ind)

print("First 5 items:", ITEMS[:5])



plt.plot(log)
plt.title("Fitness over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid()
plt.show()

# MAX_GEN = 2000, POP_SIZE = 100
# Best fitness: 50760
# Knapsack capacity: 5002
# First 5 items: [(94, 485), (506, 326), (416, 248), (992, 421), (649, 322)]

# MAX_GEN = 1000, POP_SIZE = 50
# Best fitness: 44826
# Knapsack capacity: 5002
# First 5 items: [(94, 485), (506, 326), (416, 248), (992, 421), (649, 322)]
