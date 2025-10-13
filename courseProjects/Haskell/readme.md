# Multiple knapsack Problem solving
I implement the code to solve multiple knapsack problem with Haskell


## about multiple knapsack problem and my approach

### definition of multiple knapsack problem
This time, I supposed to be given following inputs in this multiple knapsack problem
- a list of item sizes
- a list of bin capacities

And the goal of this problem is packing the items into bin as efficient as possible, that is, 
- pack the maximum possible total size of the items in all bins
- but no bin is overfilled


### my approach
I tried to solve this problem by implementing simple genetic algorithm.
I learned this algorithm in another course called Nature Inspired Algorithm 
[Link](https://ktiml.mff.cuni.cz/~pilat/en/nature-inspired-algorithms/evolutionary-algorithms-introduction/)
I also referred the code we learned for knapsack problem in that tutorial (in python) 
[Link](https://ktiml.mff.cuni.cz/~pilat/media/materials/pia/sga.py)
The detail of the algorithm(fitness function, mutation function, etc.) 
is written in the latter section with actual code.

At the same time, I tried to implement the simple recursive algorithm 
to find the best solution only for small input.
I implemented it to compare the output between two approach.

### what we can see on this project
We can implement the comparison on multiple knapsack problem solving.
I prepared 2 versions for the evolutionary algorithm(simple genetic algorithm), 
the difference is specified in the latter section *version 1 vs version 2*.
I also prepared simple recursive solution for small input, 
so we can also see the difference between simple recursion and evolutionary algorithm.

As for the evolutionary algorithm implementation, I used randomness on some values, 
so there is a small range of the error.

## how to run the code
1. store the knapsack.hs and suitable input file in the same directory.
2. open the terminal and go to that directory
3. enter the command `runhaskell knapsack.hs input_filename.txt`
4. you will see the output something like 
```
Fitness over generations:
173
...
175
...
178
..
180
...

Best individual found:
[-1,11,14,-1,7,-1,10,2,-1,11,-1,-1,14,-1,0,-1,-1,-1,12,7,-1,1,4,-1,-1,9,10,6,8,-1,2,12,1,-1,0,-1,12,-1,6,3,13,5,8,5,4,9,13,8,5,3]
Total packed size: 180 out of 293
```

## input and output

### input
The code loads the .txt file so we need to store the suitable input in it.
The first line of the input.txt file should represent the sizes of each items 
and second line should represents the capacities of each bins, 
both lines are separated by the whitespace.

ex.) 
```
5 4 3
5 5
```

This input means that we have 3 items that is

```
item 0 = size 5
item 1 = size 4
item 2 = size 3
and 2 bins that is
bin 0 = capacity 5
bin 1 = capacity 5
```

### output
It shows the fitness over generations, best individual and fitness.

ex.) (the output is not related to the input mentioned above)
```
Fitness over generations:
4
...
5
...
6
...
7
...

Best individual found:
[-1, 2, 0, 1, 3]
Total packed size: 7 out of 21
Recursively computed solution:
As individual: 
[3, 2, 1, -1, 0]
Total packed size: 16 out of 21
```

This input says that it computed with the simple genetic algorithm that

The change of the fitness over generation is 
```
4
...
5 (found the better fitness at this generation)
...
7
```
And each item assigned as follows (= individual)
```
item 0 -> unassigned
item 1 -> bin 2
item 2 -> bin 0
item 3 -> bin 1
item 4 -> bin 3
```

The fitness, the total sizes of the item packed is 7 out of 21.

while the recursively computed solution is 
```
item 0 -> bin 3
item 1 -> bin 2
item 2 -> bin 1
item 3 -> unassigned
item 4 -> bin 0
```

with the total sizes of the item packed is 16 out of 21.



## code explanation - data types

### type ItemSize 
type Int value. This represents the size of one item.


### type BinCapacity
type Int value. This represents the capacity of one bin.


### type Individual
type [Int] value.  
Each value at each index represents the item with the index number goes to the bin 
with that value at each index.
```
Individual = [1, 0]
```
means 
```
item 0 -> bin 1
item 1 -> bin 0.
```

## code explanation - global parameters

### maxGen :: Int
Defining the maximum number of generation. The default value is 1000.


### popSize :: Int
Defining the number of individuals per generation. the default value is 50.


### cxProb :: Double
Defining the probability to apply the crossover. The default value is 0.8.


### mutProb :: Double
Defining the probability to apply the mutation overall. The default value is 0.6.


### flipProb :: Double
Defining the probability to apply the flipping (mutating) in each gene. The default value is 0.2.



## code explanation - function
Basically, I added the explanation of each function more in detail 
as the comment(how each line works etc.) 
I will just sum up the purpose of each function here.

### loadInput :: FilePath -> IO ([ItemSize], [BinCapacity])
Load the input file with item sizes on first line and bin capacities on the second line.


### fitness :: [BinCapacity] -> [ItemSize] -> Individual -> Int
This fitness function evaluates the solution by 
- counting how much sizes of the item in total we could have fit
- returns -1000 as a penalty if any bin is overfilled.


**★Note for the change from the original suggestion**
I think I suggested to set fitness function to count the number of non used bins 
and you approved it. 
However, I found it inefficient so I changed to the current one to count 
how much sizes of the items are packed and trying to maximize 
that number as far as no items are duplicated and no bins are overfilled.

### createPopulation :: Int -> Int -> Int -> IO [Individual]
This function generates the population of popSize 
by using the helper function *createRandomIndividual*.

### createRandomIndividual :: Int -> Int -> IO Individual
This is a helper function for *createPopulation* function. 
This returns a list of n random integers (between 0 to numBins - 1) 
that represents the random individual assigned each item to a random bin.

### crossover :: [Int] -> [Int] -> IO ([Int], [Int])
This implement crossover with *cxProb*.
I implemented single-point crossover which splits two parents 
and generating children by swapping a part of one parent to another parent. 
If cxProb is not big enough, returns original two parents.

### mutation :: Int -> [Int] -> [Int] -> [Int] -> IO [Int]
This implements the mutation with flipProb.
It iterates over all items in all bins. 
In each item, we get some random probability r and we reassign the item to a new random bin 
if r is larger than small probability defined as flipProb (default is 0.05).

After randomly assigning to new bin, we accept this change 
only if this change didn't make any overfilling bin. 
If this change made a overfilling bin, we ignore this change and keep the old bin to be assign. 
We check this by using helper function *isValidAssignment*.


**★Note for the change from original suggestion*
I think we have talked with slightly different algorithm 
that picking one random item out of all items and reassigning to new random bin. 
However, I decided to change the approach to current one 
since I already tried to implement first one in the Nature Inspired Algorithm HW 
and I wanted to try something new and interesting. I hope it was okay to do this change.


### isValidAssignment :: [ItemSize] -> Individual -> [BinCapacity] -> Bool
This is an helper function for *mutation* to check if no bin has overfilled items after mutated.

### crossoverPopulation :: [[Int]] -> IO [[Int]]
This is helper function for *evoluteGeneric*. This applies the crossover to all individuals.

### validateIndividual :: [Int] -> [Int] -> [Int] -> Bool -> IO [Int]
This is helper function for *evolute*. This function firstly checks if no bin is over capacity.

After that, if the given Boolean argument *reassign* is True, then it reassign the unassigned items to valid random bins that can fit them.

If the item has nowhere to be assigned, left as unassigned (= -1)


### buildBinsList :: [Int] -> Int -> [[Int]]
This is helper function of *validateIndividual*.
This builds the list of bins and assign items to bins.


### trimBin :: [Int] -> [Int] -> Int -> ([Int], [Int])
This is helper function of *validateIndividual*.
This trims the items from the bin to let them fit within the capacity. 
It is returned as a list of pair represents (item to keep, item to remove).


### assignItem :: [Int] -> [Int] -> [[Int]] -> Int -> IO [[Int]]
This is helper function of *validateIndividual*. This reassign the item to a random bin 
that still have some space to fit. If there is none of such bin, item is left as unassigned.(= -1)

### binHas :: [Int] -> [Int] -> Int
This is helper function of *assignItem*. This compute how much size the bins currently have in total.

### evolveGeneric :: Int -> [Individual] -> [ItemSize] -> [BinCapacity] -> 
###                  Bool > IO ([Individual], [Int])
This is the function implements the main loop for evolution.
This loop recursively calls itself maxGen times and track best fitness per generation.
The mutation is applied with mutProb.

### evolve :: FilePath -> String -> Bool -> IO()
This function works to get best individual over N *maxGen* generations. 

This fist set up the *itemSizes* and *binCaps* based on the given *filePath*, and initialize the first population. Then run evolutionary algorithm loop for N *maxGen* generations.



### main :: IO()
This is main entry point of this codes. This reads the input file, initializes the population 
and runs simple genetic algorithm.
After running the algorithm, print the output.
There is three options for the algorithm.
Without any option, we can run version 2 algorithm.
With `-r` option, we can run version 1 algorithm.
And with `-e` option, we can run the recursive algorithm for comparison.


## for comparison
I wrote the function recursively calculate the best packing

### findBestPartialPacking :: [Int] -> [Int] -> [[Int]]
Compute the optimal packing.
This tries to assign each item to every bin and returns the best packing.


### binsToIndividual :: Int -> [[Int]] -> [Int]
This converts the style to more readable individual-style representation 
so that it becomes easier to compare the results.



## ver.1 vs ver.2
The key change is the way to deal with the removed item due to the duplication and overfilling 
after crossover and mutation. 
I was just curious to see the difference of the outcome 
between the difference of this part of approach.
It is managed in the function *validateIndividual* which is called in *createNextGeneration*

### version 1
In *validateIndividual_1* function, we check all bins if any bin is overfilled 
after removing the duplicated item(keep in one random bin).
If there is any bin that is overfilled, we removed the items in that bin randomly 
until it is not overfilled.
Then, we assign -1 to mark "unassigned" to all removed items when we rebuild the individual.


### version 2
I tried to another way for this part and tried to reassign those removed items in this version 2.
The helper functions *removeDuplicates* and *trimBin* for the function *validateIndividual* 
work similarly to the strategy of version 1 that manages the duplication and overfilling.
After managing those two cases, we reassign removed items to new bins with making sure 
no bins are overfilled and no item is duplicated.


## test cases

### test 1 - 1 (small input that all of them can fits)
```
input:
3 2 4 1 5 2 3 1 4 2
10 8 9 7 6
```

**output of version 1**
```
Best individual found: 
[0,3,4,2,2,3,0,4,0,3]
Total packed size: 27 out of 27
```

**output of version 2**
```
Best individual found: 
[0,2,0,1,2,4,1,2,4,0]
Total packed size: 27 out of 27
```

**output of recursion**
```
Recursively computed solution: 
As individual: 
[1,1,2,3,2,3,3,3,4,4]
Total packed size: 27 out of 27
```

### test 1 - 2 (small input2 that not all of them can fits)
```
input:
6 5 4 3 7 2 8 2 3 5
5 5 10 6 7
```

**output of version 1**
```
Best individual found: 
[3,-1,-1,-1,4,2,2,1,1,0]
Total packed size: 33 out of 45
```

**output of version 2**
```
Best individual found: 
[3,2,-1,1,4,1,-1,2,2,0]
Total packed size: 33 out of 45
```

**output of recursion**
```
As individual: 
[3,0,-1,1,4,1,2,2,-1,-1]
Total packed size: 33 out of 45
```


### test 1 - 3 (small input2 that none of them can fit)
```
input:
11 12 13 14 15 16 17 18 19 20
5 5 5 5 5
```
**output of version 1**
```
Best individual found: 
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
Total packed size: 0 out of 155
```

**output of version 2**
```
Best individual found: 
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
Total packed size: 0 out of 155
```

**output of recursion**
```
As individual: 
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
Total packed size: 0 out of 155
```

### test 2 - 1 (mid input that all of them can fit)
```
input:
4 2 3 1 7 5 2 6 3 4 8 1 2 5 6 4 3 2 1 7 6 2 5 3 4 2 1 3 4 2
15 12 18 10 17 20 13 16 15 19
```

**output of version 1**
```
Best individual found: 
[2,3,2,8,5,6,9,8,0,1,5,7,0,9,0,6,2,7,3,7,4,3,7,2,0,4,9,2,3,9]
Total packed size: 108 out of 108
```

**output of version 2**
```
Best individual found: 
[8,4,6,1,3,5,2,5,5,9,0,9,5,6,4,2,2,5,0,2,4,1,8,7,1,1,1,7,0,7]
Total packed size: 108 out of 108
```


### test 2 - 2 (mid input that all of them can't fit)
```
input:
5 8 6 9 3 7 5 4 8 6 9 5 3 7 6 8 5 4 6 7 9 3 4 8 6 5 7 3 4 6
10 12 8 9 10 11 10 12 9 9
```

**example output of version 1**
```
Best individual found: 
[-1,7,-1,-1,1,-1,-1,7,1,-1,5,-1,-1,2,-1,-1,-1,4,-1,-1,6,-1,-1,0,9,8,3,9,8,4]
Total packed size: 91 out of 176
```

**example output of version 2**
```
Best individual found: 
[5,-1,-1,-1,7,-1,7,8,-1,-1,-1,-1,3,2,-1,-1,8,0,5,-1,6,3,4,9,4,1,1,3,7,0]
Total packed size: 97 out of 176
```

**change of the fitness after running 10 times**
```
- version 1
    91, 98, 93, 92, 96, 97, 97, 98, 96, 97
    average: 95.5
    
- version 2
    97, 99, 100, 100, 99, 98, 97, 99, 97, 98
    average: 98.4
```

### test 3 - 1(large input that all of them can fit)
```
input:
3 5 7 2 6 1 8 4 3 5 9 2 4 6 3 1 2 7 4 8 6 5 3 2 7 9 4 1 3 5 2 8 6 7 5 4 3 6 1 9 2 5 8 6 4 3 2 7 5 6
20 15 25 22 18 30 16 24 19 21 26 28
```

**output of version 1**
```
Best individual found:
[7,11,8,8,10,6,5,3,2,6,9,6,9,5,0,11,8,5,10,7,2,9,0,5,1,7,10,8,4,3,9,0,11,3,10,3,10,2,2,11,11,4,2,4,4,10,11,8,6,1]
Total packed size: 234 out of 234
```

**output of version 2**
```
Best individual found: 
[6,8,2,6,1,5,10,9,2,1,4,0,11,11,5,8,1,9,3,5,2,10,10,5,8,3,8,0,7,10,9,7,5,0,11,5,6,9,11,7,11,4,0,11,5,11,2,2,3,6]
Total packed size: 234 out of 234
```

### test 3 - 2 (large input that all of them can not fit)
```
input:
4 7 6 5 8 9 6 5 7 3 4 6 7 8 3 9 6 7 5 4 8 6 5 7 9 3 6 5 4 7 8 3 6 7 9 4 5 6 8 7 6 4 5 3 7 8 6 5 4 3
12 14 13 10 12 11 13 12 14 11 12 10 13 12 13
```


**example output of version 1**
```
Best individual found: 
[0,-1,3,-1,-1,-1,7,4,-1,-1,7,-1,4,-1,12,-1,-1,-1,14,2,-1,-1,13,-1,2,1,-1,11,11,6,14,1,6,13,8,5,10,5,-1,-1,9,3,0,12,12,1,10,9,8,0]
Total packed size: 176 out of 293
```

**example output of version 2**  
```
Best individual found: 
[11,-1,12,0,-1,-1,-1,14,-1,9,10,8,-1,9,-1,-1,-1,-1,-1,8,10,4,6,-1,-1,7,13,2,8,-1,6,13,0,14,-1,7,3,1,5,12,4,2,7,13,-1,1,11,3,2,5]
Total packed size: 180 out of 293
```

**change of the fitness value**
```
- version 1
    175, 176, 177, 173, 173, 177, 178, 175, 174, 178
    average: 175.6
    
- version 2
    180, 177, 179, 180, 180, 180, 180, 181, 180, 181
    average: 179.8
```

## observation of the sample test results
As for the comparison between evolutionary algorithm vs simple recursion, 
both produced the same results in all 3 samples.
The process speed was much faster with evolutionary algorithm than simple recursion.

As for the comparison between version 1 and 2, there were no difference for the small input 
and both could get the optimal solution.
Both of the could have gotten the optimal solution for the larger input when all items fit to bins.
It showed the difference for the larger input that all items can not fits.
In both tests mid size input and large size input, 
the version 2 brought more efficient solution than version 1.

In the end, the modification for the reassignment of the removed items was meaningful 
to improve the outcome of the algorithm.