**Autonomous Robot Navigation in the Vicinity of Saturn's Rings**

Following our successful colonization of Mars, we are now faced with the challenge of exploring the Rings of Saturn. 

The initial station has been constructed; however, it is not yet self-sustaining and thus requires regular shipments of supplies. Due to navigational difficulties, rockets do not land precisely at the station but rather in its vicinity, often resulting in crashes. Consequently, supplies must be transported from the crash site to the station by an autonomous robot. Our objective is to develop a program that effectively controls this robot.

For the sake of simplicity, the vicinity of the station is divided into cells that form a toroidal map. While finding the shortest path on a torus may seem straightforward, several significant challenges must be addressed.

1. The robot accepts commands such as NORTH, SOUTH, WEST, and EAST, which are intended to move the robot one cell in the specified direction. However, due to damage sustained during the landing, the robot's movement is imperfect. Following an initial diagnosis, the robot has estimated probabilities p_forward, p_backward, p_left, and p_right that collectively sum to one. These probabilities represent the likelihood of the robot's actual movement relative to the commanded direction. For instance, when commanded to move EAST, the robot will move eastward with probability p_forward, southward with probability p_right, westward with probability p_backward, and northward with probability p_left.

2. The environment surrounding the Rings of Saturn is fraught with obstacles. Previous explorations have provided estimates of the energy required to traverse from one cell (A) to an adjacent cell (B). The robot possesses a map that indicates the location of the base and includes a matrix M that quantifies the energy expenditure necessary to reach each cell. Thus, the energy required to travel from cell A to cell B is represented by M[B].

Fortunately, the robot's localization system functions flawlessly, allowing it to ascertain its position after each landing and movement.

Our task is to develop a program that minimizes the total energy expenditure required for the robot to reach the station, as the station has substantial energy demands. The file `robot_control.py` currently contains a rudimentary control algorithm, and it is your responsibility to enhance it. You are permitted to modify this file as needed, but it is essential to maintain the interface utilized by the file `robot_test.py`. Only the file `robot_control.py` is expected to be submitted.

You are encouraged to leverage all knowledge acquired from our course, including informed search, logic, and probabilistic reasoning. In particular, Chapters 17.2 and 17.3 of the book *Artificial Intelligence: A Modern Approach* (3rd edition) may prove beneficial. However, you are also expected to provide a clear explanation of your approach, so please include comments in your code. Inadequate clarity in your code may result in a reduction of points.

**Hints:**
- Do not interpret the Bellman equation from the lecture too rigidly; it has numerous variants, and the objective of this assignment is to adapt it appropriately for the given problem.
- The energy matrix also defines the coordinate system. Therefore, moving south increases the row index, while moving east increases the column index.
- The structure of the Rings of Saturn is toroidal. Thus, moving north from the first row leads the robot to the last row, and moving east from the last column brings it back to the first column. Consequently, the modulus operator (%) may be useful for calculating the neighbors of each cell.
- Each test involves multiple landings in the vicinity of a single station, meaning that energy consumption remains constant. It may be counterintuitive, but the distribution of actual movements p_forward, p_backward, p_left, and p_right remains unchanged after each landing. Therefore, it is advisable to pre-compute commands for every cell during the initialization of each test to enhance the efficiency of your program.
- Mathematically, the problem is stationary; that is, the expected energy required to reach the station from a given cell does not depend on the past (i.e., the manner in which the cell was reached, assuming successful arrival). Thus, it is recommended to calculate the minimal expected energy required to reach the station from every cell (which can be interpreted as the expected distance from a cell to the station) and to determine the optimal policy (direction) for each cell. The calculation of these distances and policies can be verified using the script `probability_test.py`. Since the evaluation of these tests on Recodex yields zero points, their execution is voluntary and may provide valuable insights for completing this assignment.

- The robot has a limited energy capacity; however, exceeding this limit indicates that the robot is being directed in entirely incorrect directions (for example, confusing west with east, or miscalculating coordinates on the torus).
  
- It may be beneficial to initially implement the value update method; however, this approach often leads to a lower score due to its inherent inefficiency. To improve computational speed, it is advisable to focus on implementing the policy update method.
Policy updates can be tested using the script `policy_iteration_test.py`. This script expects the implementation of the functions `get_policy_from_distance` and `get_distance_from_policy`, as discussed during the lecture and tutorial sessions.
By adhering to these guidelines and ensuring that the required functions are correctly implemented, you will facilitate a more efficient navigation strategy for the robot, ultimately contributing to the successful completion of the assignment.

- The Python package `scipy` may be instrumental in solving this task. In particular, the use of a [sparse matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html) and a [linear system solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html) could be particularly beneficial.

