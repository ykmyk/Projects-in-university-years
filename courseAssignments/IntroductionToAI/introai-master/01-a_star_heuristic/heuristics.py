# TODO: Implement more efficient monotonic heuristic
#
# Every function receive coordinates of two grid points returns estimated distance between them.
# Each argument is a tuple of two or three integer coordinates.
# See file task.md for description of all grids.

import math
from graphs import Grid2D, GridDiagonal2D, GridGreatKing2D, GridRook2D, GridJumper2D, Grid3D, GridFaceDiagonal3D, GridAllDiagonal3D

# For two points a and b in the n-dimensional space, return the d-dimensional point r such that r_i = | a_i - b_i | for i = 1...d
def distance_in_each_coordinate(x, y):
    return [ abs(a-b) for (a,b) in zip(x, y) ]

def grid_2D_heuristic(current, destination):
    x = distance_in_each_coordinate(current, destination)
    return x[0] + x[1]

def grid_diagonal_2D_heuristic(current, destination):
    x = distance_in_each_coordinate(current, destination)
    return max(x[0], x[1])

def grid_3D_heuristic(current, destination):
    x = distance_in_each_coordinate(current, destination)
    return x[0] + x[1] + x[2]

def grid_face_diagonal_3D_heuristic(current, destination):
    dist = distance_in_each_coordinate(current, destination)
    return math.ceil(max(sum(dist) / 2, max(dist)))


def grid_all_diagonal_3D_heuristic(current, destination):
    x = distance_in_each_coordinate(current, destination)
    return max(x[0], x[1], x[2])

def grid_great_king_2D_heuristic(current, destination):
    chebyshev = grid_diagonal_2D_heuristic(current, destination)
    return math.ceil(chebyshev / 8)

def grid_rook_2D_heuristic(current, destination):
    dist = distance_in_each_coordinate(current, destination)
    x = math.ceil(dist[0] / 8)
    y = math.ceil(dist[1] / 8)
    return x + y

# def grid_jumper_2D_heuristic(current, destination):
#     dist = distance_in_each_coordinate(current, destination)
#     x = dist[0]
#     y = dist[1]
#     move32 = max(math.ceil(x / 3), math.ceil(y / 2))
#     move23 = max(math.ceil(x / 2), math.ceil(y / 3))
#     return math.ceil((move23 + move32) / 2)

def grid_jumper_2D_heuristic(current, destination):
    dist = distance_in_each_coordinate(current, destination)
    return math.ceil(max(dist[0] / 3, dist[1] / 2))

# if __name__ == "__main__":
#     current = (0, 0, 0)
#     destination = (4, 9, 22)
#     print("heuristic distance", grid_face_diagonal_3D_heuristic(current, destination))