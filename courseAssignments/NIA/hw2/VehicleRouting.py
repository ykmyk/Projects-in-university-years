from collections import namedtuple
import math
import functools
import numpy as np

Vertex = namedtuple('Vertex', ['name', 'x', 'y', 'demand'])

# main ACO function
def ant_solver(vertices, distance, vehicle_capacity, ants=10, max_iter=3000, alpha=1, beta=3, Q=100, rho=0.8):
    P = initialize_pheromone(len(vertices))
    history = []
    best_fit = float('inf')
    best_sol = None
    for it in range(max_iter):
        sols = list(generate_solutions(vertices, P, distance, vehicle_capacity, ants, alpha=alpha, beta=beta))
        fits = list(map(lambda x: fitness(vertices, distance, x, vehicle_capacity), sols))

        for s, f in zip(sols, fits):
            if f < best_fit:
                best_fit = f
                best_sol = s

        P = update_pheromone(P, sols, fits, Q=Q, rho=rho)
        history.append(best_fit)  
    return best_sol, P, history

# compute distance
@functools.lru_cache(maxsize=None)
def distance(v1, v2):
    return math.sqrt((v1.x - v2.x)**2+(v1.y - v2.y)**2)

# compute fitness
def fitness(vertices, dist, sol, vehicle_capacity):
    sd = 0
    for x, y in zip(sol, sol[1:]):
        sd += dist(vertices[x], vertices[y])
    sd += dist(vertices[sol[-1]], vertices[sol[0]])
    return sd

# pheromone initizalization
def initialize_pheromone(N):
    return 0.01*np.ones(shape=(N,N))

# generate solution
def generate_solutions(vertices, P, dist, vehicle_capacity, N, alpha=1, beta=3):
    
    # probability of selecting and edge (without scaling)
    def compute_prob(v1, v2):
        if v1 == v2:
            return 0.000001
        nu = 1/dist(vertices[v1], vertices[v2])
        tau = P[v1, v2]
        ret = pow(tau, alpha) * pow(nu,beta)
        return ret if ret > 0.000001 else 0.000001

    V = P.shape[0]
    for i in range(N):
        available = list(range(V))
        sol = [np.random.randint(0, V)]
        available.remove(sol[0])
        remaining_capaicty = vehicle_capacity
        while available:
            probs = np.array(list(map(lambda x: compute_prob(sol[-1], x), available)))
            selected = np.random.choice(available, p=probs/sum(probs)) # edge selection
            if vertices[selected].demand <= remaining_capaicty:
                sol.append(selected)
                available.remove(selected)
                remaining_capaicty -= vertices[selected].demand
            else:
                sol.append(0)
                remaining_capaicty = vehicle_capacity
                if 0 in available:
                    available.remove(0)
        yield sol

def update_pheromone(P, sols, fits, Q=100, rho=0.6):
    ph_update = np.zeros(shape=P.shape)
    for s, f in zip(sols, fits):
        for x, y in zip(s, s[1:]):
            ph_update[x][y] += Q/f
        ph_update[s[-1]][s[0]] += Q/f
    
    return (1-rho)*P + ph_update        
