from parseXML import parse_xml
from VehicleRouting import ant_solver, distance, Vertex
import matplotlib.pyplot as plt
import os

def main():
    parsed_data = parse_xml('data_422.xml')

    # right after parsing:
    vertices = [
        Vertex(name=node["id"], x=node["cx"], y=node["cy"], demand=0)
        for node in parsed_data["nodes"]
    ]
    id2idx = { v.name: i for i, v in enumerate(vertices) }

    # then for each request:
    for req in parsed_data["requests"]:
        idx = id2idx[req["node"]]
        v = vertices[idx]
        vertices[idx] = v._replace(demand=req["quantity"])

    vehicle_capacity = parsed_data["fleet"][0]["capacity"]  
    ants = 10
    max_iter = 3000
    alpha = 1
    beta = 3
    Q = 100
    rho = 0.8

    best_solution, pheromone_matrix, hist = ant_solver(vertices, distance, vehicle_capacity, ants,  max_iter, alpha, beta, Q, rho)
    draw(hist, "data_422.xml")
    best_solution = [int(x) for x in best_solution]
    print("Best solution:", best_solution)

    return best_solution

def draw(data, filename):
    plt.plot(data)
    plt.title(filename)
    plt.xlabel("Iterations")
    plt.ylabel("Objective Value")
    plt.show()

    
if __name__ == "__main__":
    main()
