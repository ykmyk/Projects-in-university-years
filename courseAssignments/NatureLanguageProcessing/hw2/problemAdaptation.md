# adaptation of the source code to solve this problem
I reffered the code in this Github directory: https://github.com/gabikadlecova/inspinature/blob/main/en/10-ants_n_birbs/ants_n_birbs.ipynb

## Extention of the data set
I extended the data set from 
```
Vertex = namedtuple('Vertex', ['name', 'x', 'y'])
```
to
```
Vertex = namedtuple('Vertex', ['name', 'x', 'y', 'demand'])
```
to consider available shipment size on each node.

## Vehicle capacity
For overall codes, I let them manage the vehicle capacity that change of code can be found in many places, for instance
```
def ant_solver(vertices, distance, vehicle_capacity, ants=10, max_iter=3000, alpha=1, beta=3, Q=100, rho=0.8):
```

## Capacity tracking
when it generate the solution(in def generate_solution), it will keep tracking remaining capacity so that if next demand fits, append it but if not, return to the deposit(central hub) 0 and reset the capacity.
Also, by removing 0 from "available" when it returned, it will prevent looping on the deposit itself.


## Fitness
in original fittness was just sum of edge lengths over 
one cycle, however, I updated it to be able to manage multiple route.
```
def update_pheromone(P, sols, fits, Q=100, rho=0.6):
    ph_update = np.zeros(shape=P.shape)
    for s, f in zip(sols, fits):
        for x, y in zip(s, s[1:]):
            ph_update[x][y] += Q/f
        ph_update[s[-1]][s[0]] += Q/f
```