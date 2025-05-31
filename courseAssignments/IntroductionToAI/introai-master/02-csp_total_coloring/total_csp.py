# Install package python-constraint, not constraint !!!
import constraint
import networkx as nx

def total_coloring(graph):
    #{"node" : degree}
    max_degree = max(dict(graph.degree()).values())

    # make the list of nodes and edges
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    # to let readable for the python constrain 
    node_vars = {node: f"n_{node}" for node in nodes}
    edge_vars = {tuple(sorted(edge)): f"e_{min(edge)}_{max(edge)}" for edge in edges}


    # since the upper bound is max_degree + 2(from wikipedia resource)
    for color_count in range(max_degree + 1, max_degree + 3):
        problem = constraint.Problem()

        # assigning the variable which set the color
        for var in node_vars.values():
            problem.addVariable(var, range(color_count))
        for var in edge_vars.values():
            problem.addVariable(var, range(color_count))
        
        # adding the constraints
        # constraint 1: Adjacent vertices must have different colors
        for u, v in graph.edges():
            problem.addConstraint(constraint.AllDifferentConstraint(), [node_vars[u], node_vars[v]])

        # constraint 2: Adjacent edges must have different colors
        for node in graph.nodes():
            incident_edges = list(graph.edges(node))
            for i in range(len(incident_edges)):
                for j in range(i + 1, len(incident_edges)):
                    e1 = incident_edges[i]
                    e2 = incident_edges[j]
                    key1 = edge_vars[tuple(sorted(e1))]
                    key2 = edge_vars[tuple(sorted(e2))]
                    problem.addConstraint(constraint.AllDifferentConstraint(), [key1, key2])

        # constraint 3 : Each edge must differ from both of its endpoints
        for u, v in graph.edges():
            edge_var = edge_vars[tuple(sorted((u, v)))]
            problem.addConstraint(constraint.AllDifferentConstraint(), [edge_var, node_vars[u]])
            problem.addConstraint(constraint.AllDifferentConstraint(), [edge_var, node_vars[v]])

        # get the solution with the constrains above
        solution = problem.getSolution()
        
        # if any solution is found, assing the color to each nodes and edges, 
            # and return the color count
        if solution:
            for node in graph.nodes():
                graph.nodes[node]["color"] = solution[node_vars[node]]
            for u, v in graph.edges():
                graph.edges[u, v]["color"] = solution[edge_vars[min(u, v), max(u, v)]]
            return color_count



    colors = 0
    for u in graph.nodes():
        graph.nodes[u]["color"] = colors
        colors += 1
    for u,v in graph.edges():
        graph.edges[u,v]["color"] = colors
        colors += 1
    return colors

