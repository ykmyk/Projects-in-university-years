---
title: Untitled

---

## Description of the algorithm (encoding of individuals, genetic operators, chosen selection method, etc.)

### encoding of individuals
1 represents the item at that position is selected
0 represents the item at that position is not selected

### generic operators
Crossover
I used single point crossover with 80% probability

Mutation
I used bit-flip mutation as we learned in the class

### chosen selection method
I used Roulette wheel selection

## Algorithm code + instructions for running it
The algorithm code is in the file knapsack.py

just run like
python3 knapsack.py --input_file texts/input_100.txt

Sorry, I didn't get what you meant, was this what you meant?

## Graph showing how fitness changed over generations
it is attached togather with the name 
FitnessOverGeneration10.png
FitnessOverGeneration20.png
FitnessOverGeneration100.png
FitnessOverGeneration1000.png

## The best solution you discovered (especially its price)
### for the debug_10.txt
Best fitness: 295
Selected 6 items. Total weight = 269, total value = 295
First 5 items: [(55, 95), (10, 4), (47, 60), (5, 32), (4, 23)]

### for the debug_20.txt
Best fitness: 1024
Selected 17 items. Total weight = 871, total value = 1024
First 5 items: [(44, 92), (46, 4), (90, 43), (72, 83), (91, 84)]

### for the input_100.txt
Best fitness: 9147
Selected 12 items. Total weight = 985, total value = 9147
First 5 items: [(94, 485), (506, 326), (416, 248), (992, 421), (649, 322)]

### for the input_1000.txt
Best fitness: 40582
Selected 59 items. Total weight = 4981, total value = 40582
First 5 items: [(94, 485), (506, 326), (416, 248), (992, 421), (649, 322)]
