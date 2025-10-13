{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use uncurry" #-}
{-# HLINT ignore "Eta reduce" #-}
{-# HLINT ignore "Redundant return" #-}
{-# HLINT ignore "Use notElem" #-}
{-# HLINT ignore "Use infix" #-}
{-# HLINT ignore "Replace case with maybe" #-}
{-# HLINT ignore "Redundant bracket" #-}
{-# HLINT ignore "Use list comprehension" #-}
{-# HLINT ignore "Use if" #-}
import System.Environment
import System.IO
import Control.Monad
import Data.List
import System.Random
import Data.Ord

-- Define the data type for ItemSize and BinCapacity
type ItemSize = Int
type BinCapacity = Int
type Individual = [Int]

maxGen :: Int
maxGen = 100

popSize :: Int
popSize = 50

cxProb :: Double
cxProb = 0.8

-- default was set as 0.6 in reference code
mutProb :: Double
mutProb = 0.6

flipProb :: Double
flipProb = 0.2

-- Read and parse the input file
-- we suppose that the file contains 
-- 5 30 10 23 7 ... item sizes
-- 9 20 40 15 5 ... bin capacities

loadInput :: FilePath -> IO ([ItemSize], [BinCapacity])
loadInput path = do
    content <- readFile path
    let ls = lines content
    let itemSizes = map read (words (head ls)) :: [ItemSize]
    let binCaps = map read (words (ls !! 1)) :: [BinCapacity]
    return (itemSizes, binCaps)

fitness :: [BinCapacity] -> [ItemSize] -> Individual -> Int
fitness binCaps itemSizes individual =
    -- zip item with its bin assignments
    -- skip that has bin -1 
    -- => it is the removed item due to duplication/overfilling in past generation
    let assigned = [ (bin, size) | (bin, size) <- zip individual itemSizes, bin >= 0 ]

    -- group items into bins 
    -- itemsInBins :: [[ItemSize]] => [[items sizes in bin 1], [items sizes in bin 2], ...]
        binCount = length binCaps
        itemsInBins = map (\b -> [size | (bin, size) <- assigned, bin == b]) [0 .. binCount - 1]

    -- total size of the items in each bins
        totalItemSizes = map sum itemsInBins

    -- check if all bins are valid
    -- lambda function: for a pair (cur, cap) return True if cur <= cap
        valid = all (\(cur, cap) -> cur <= cap) (zip totalItemSizes binCaps)

    in if valid
        then sum [size | (_, size) <- assigned] -- total size packed
        else -1000

-- create random individuals
createPopulation :: Int -> Int -> Int -> IO [Individual]
createPopulation popSize numItems numBins =
    replicateM popSize (createRandomIndividual numItems numBins)
-- helper function
-- returns a list of n random integers (between 0 to numBins - 1)
-- [2, 0, 1, 1, 2] means item 0 goes to bin 2, item 1 goes bin 0, ...
createRandomIndividual :: Int -> Int -> IO Individual
createRandomIndividual n numBins = replicateM n (randomRIO (0, numBins - 1))

crossover :: [Int] -> [Int] -> IO ([Int], [Int])
crossover parent1 parent2 = do
-- depends on the random probability r, crossover might be skipped
    r <- randomRIO (0.0, 1.0)
    if r < cxProb
        then do
            let len = length parent1
        -- pick the random point to implement crossover
            point <- randomRIO (0, len - 1)
            let (p1L, p1R) = splitAt point parent1
            let (p2L, p2R) = splitAt point parent2
            let child1 = p1L ++ p2R
            let child2 = p2L ++ p1R
            return (child1, child2)
        else
            return (parent1, parent2)


-- implement mutation if random probability r is less than flipProb
mutation :: Int -> [Int] -> [Int] -> [Int] -> IO [Int]
mutation numBins itemSizes binCaps individual =
    foldM mutateGene [] (zip [0..] individual)
  where
    mutateGene acc (i, oldBin) = do
        r <- randomRIO (0.0, 1.0)
        -- only if the random probability r is less than flipProb
        if r < flipProb
            then do
                -- pick random bin
                newBin <- randomRIO (0, numBins - 1)
                let candidate = acc ++ [newBin] ++ drop (i + 1) individual
                -- consider this mutation only if all bins are valid after mutating
                -- if isValidAssignment itemSizes candidate binCaps
                --     then return (acc ++ [newBin]) else
                return (acc ++ [newBin])  
            else return (acc ++ [oldBin])

-- helper function to check if it is valid assignment
isValidAssignment :: [ItemSize] -> Individual -> [BinCapacity] -> Bool
isValidAssignment itemSizes individual binCaps =
    let itemAssignment = zip individual itemSizes
        binCount = length binCaps
        itemsInBins = map (\b -> [size | (bin, size) 
                                <- itemAssignment, bin == b]) [0 .. binCount - 1]
        totalItemSizes = map sum itemsInBins
    in all (\(cur, cap) -> cur <= cap) (zip totalItemSizes binCaps)


-- helper function to define pairwise crossover of the entire population recursively
-- input : a population of individuals (each [Int] is the individual, representing a bin assignment)
-- output : new list of individuals(offspring)
crossoverPopulation :: [[Int]] -> IO [[Int]]
crossoverPopulation [] = return []
crossoverPopulation [p] = return [p]
crossoverPopulation (p1 : p2 : rest) = do
    (c1, c2) <- crossover p1 p2
    restChildren <- crossoverPopulation rest
    return (c1 : c2: restChildren)

-- validating the individual or fix it to be validated if it's invalidate by
-- 1. checking no bin is over capacity
-- 2. (if reassign = True) re-assigning unassigned items to valid bins that can fit them
-- input : list of item sizes, current assignment of items to bins(might be invalid)
--         and capacity of each bin
validateIndividual :: [Int] -> [Int] -> [Int] -> Bool -> IO [Int]
validateIndividual itemSizes individual binCaps reassign = do
    let numBins = length binCaps
        numItems = length itemSizes
        bins = buildBinsList individual numBins

    -- trim overfilled bins
    let trimAll [] [] = ([], [])
        trimAll (bin : bs) (cap : cs) =
            let (kept, removed) = trimBin itemSizes bin cap
                (restKept, restRemoved) = trimAll bs cs
            in (kept : restKept, removed ++ restRemoved)
    let (trimmedBins, removed1) = trimAll bins binCaps
    
    -- reassign removed items
    finalBins <- case reassign of
        True -> foldM (assignItem itemSizes binCaps) trimmedBins removed1
        False -> return trimmedBins
    -- rebuild individual
    -- assign -1 if the item is not assigned anywhere
    let findBin _ [] _ = -1
        findBin i (bin:rest) b = if i `elem` bin then b else findBin i rest (b + 1)
        rebuilt = [findBin i finalBins 0 | i <- [0 .. numItems - 1]]

    return rebuilt

-- helper function to assign items to bins
-- ex. [2, 0, 2, 1] 3 returns [[1], [3], [0, 2]]
buildBinsList :: [Int] -> Int -> [[Int]]
buildBinsList individual numBins =
    let indexed = zip [0..] individual
        addToBins bins (item, bin)
            -- skip unassigned or invalid bin
            | bin < 0 || bin >= numBins = bins
            | otherwise = take bin bins ++ [item : (bins !! bin)] ++ drop (bin + 1) bins
    in foldl addToBins (replicate numBins []) indexed

-- helper function
-- returns the list of pair (kept, removed)
-- ex. itemSizes = [1, 2, 3, 4], binItems = [0, 1, 2], binCap 5 then
-- item 0 + item 1 = 3 (still fit) but item 0 + 1 + 2 = 6 (overflow)
-- so the trimBin [1, 2, 3, 4] [0, 1, 2] 5 returns ([0, 1], [2])
trimBin :: [Int] -> [Int] -> Int -> ([Int], [Int])
trimBin itemSizes binItems cap = trim binItems [] 0
    where
        trim [] kept _ = (reverse kept, [])
        trim (i : is) kept cur =
            let size = itemSizes !! i in
                if cur + size <= cap
                    then trim is (i : kept) (cur + size) -- keep item i
                    else let (k, r) = trim is kept cur -- skip item i
                        in (k, i : r)

-- helper function to re-assigning the removed item
assignItem :: [Int] -> [Int] -> [[Int]] -> Int -> IO [[Int]]
assignItem itemSizes binCaps bins i = do
    let size = itemSizes !! i
        numBins = length binCaps
        tryBins = [b | b <- [0 .. numBins - 1], 
                                binHas itemSizes (bins !! b) + size <= binCaps !! b]
    if not (null tryBins)
        then do
            b <- randomRIO (0, length tryBins - 1)
            let chosenBin = tryBins !! b
            return $ take chosenBin bins
                  ++ [(bins !! chosenBin) ++ [i]]
                  ++ drop (chosenBin + 1) bins
        else
            -- No bin fits → leave item unassigned
            return bins

-- helper function to compute how much size the bin currently have in total
binHas :: [Int] -> [Int] -> Int
binHas itemSizes itemIDs = sum [itemSizes !! i | i <- itemIDs]

-- implemented Roulette-wheel selection
selection :: [(Individual, Int)] -> Int -> IO [Individual]
selection popWithFit k = do
    let totalFit = sum (map snd popWithFit)
        cumulative = scanl1 (+) (map snd popWithFit)
        individuals = map fst popWithFit
    replicateM k $ do
        r <- randomRIO (1, totalFit)
        let idx = min (length individuals - 1) (length (takeWhile (< r) cumulative))
        return $ individuals !! idx
        

evolveGeneric :: Int -> [Individual] -> [ItemSize] -> [BinCapacity] -> 
                    Bool -> IO ([Individual], [Int])
evolveGeneric 0 pop itemSizes binCaps validateF = return (pop, [])
evolveGeneric gen pop itemSizes binCaps reassign = do
    let fits = map (fitness binCaps itemSizes) pop
    selected <- selection (zip pop fits) (length pop)
    offspring <- crossoverPopulation selected
    mutated <- mapM (\ind -> do
        r <- randomRIO (0.0, 1.0)
        if r < mutProb
            then mutation (length binCaps) itemSizes binCaps ind
            else return ind)
        offspring
    validated <- mapM (\ind -> validateIndividual itemSizes ind binCaps reassign) mutated

    let best = maximumBy (comparing (fitness binCaps itemSizes)) pop
        finalPop = best : take (length validated - 1) validated
        bestFitness = maximum (map (fitness binCaps itemSizes) finalPop)
        maxPossibleFitness = sum itemSizes

    if bestFitness == maxPossibleFitness
        then return (finalPop, [bestFitness])
    else do
        (nextPop, log) <- evolveGeneric (gen - 1) finalPop itemSizes binCaps reassign
        return (nextPop, bestFitness : log)


-- for the comparison only for small input when length itemSizes <= 10 && length binCaps <= 5
-- find best packing by simple recursive function

-- Find the best packing of any subset of items maximizing total size packed
findBestPartialPacking :: [Int] -> [Int] -> [[Int]]
findBestPartialPacking itemSizes binCaps =
    let itemIndices = [0 .. length itemSizes - 1]
        allSubsets = subsequences itemIndices  -- try larger subsets first
        allAssignments = [ (subset, bins)
                         | subset <- allSubsets,
                         bins <- assignItems subset]
    in snd $ maximumBy comparePackedSize allAssignments
        where
        -- Try to assign items to bins without overfilling
        assignItems :: [Int] -> [[[Int]]]
        assignItems [] = [replicate (length binCaps) []]
        assignItems (i:is) = do
            rest <- assignItems is
            tryAssign i rest

        tryAssign :: Int -> [[Int]] -> [[[Int]]]
        tryAssign i bins =
            let size = itemSizes !! i
                numBins = length binCaps
                tryBin j =
                    if (j < numBins) && (binSum (bins !! j) + size <= binCaps !! j)
                        then [take j bins ++ [(bins !! j) ++ [i]] ++ drop (j + 1) bins]
                        else []
            in concatMap tryBin [0 .. numBins - 1]

        binSum bin = sum [itemSizes !! i | i <- bin]

        comparePackedSize (s1, _) (s2, _) = compare (binSum s1) (binSum s2)

-- convert bin-based solution to individual-style representation
binsToIndividual :: Int -> [[Int]] -> [Int]
binsToIndividual numItems bins =
    let itemAssignments = concat [ [(i, b)] | (b, bin) <- zip [0..] bins, i <- bin ]
        lookupMap = [ case lookup i itemAssignments of
                        Just b -> b
                        Nothing -> -1
                    | i <- [0 .. numItems - 1]]
    in lookupMap

evolve :: FilePath -> String -> Bool -> IO()
evolve filePath version reassign = do
            (itemSizes, binCaps) <- loadInput filePath
            let numItems = length itemSizes
            let numBins = length binCaps

            -- initialize population
            initPop <- createPopulation popSize numItems numBins

            -- run loop for N maxGen generations
            (finalPop, fitnessLog) <- evolveGeneric maxGen initPop itemSizes binCaps reassign
            -- print the best fitness per generation
            putStrLn "\nFitness over generations:"
            mapM_ print fitnessLog 

            -- get the best individual
            let best = maximumBy (comparing (fitness binCaps itemSizes)) finalPop
            putStrLn $ "Best individual found with version " ++ version ++ " algorithm :" 
            print best

            let assigned = [size | (bin, size) <- zip best itemSizes, bin >= 0]
            let totalPackedSize = sum assigned
            putStrLn $ "Total packed size: " ++ show totalPackedSize 
                                ++ " out of " ++ show (sum itemSizes)


-- enter "runhaskell binPacking.hs input.txt" from the terminal
main :: IO()
main = do
    args <- getArgs
    case args of
        ["-e", filePath] -> do
            (itemSizes, binCaps) <- loadInput filePath
            do
                putStrLn "Running exact search for comparison..."
                case findBestPartialPacking itemSizes binCaps of
                    [] -> putStrLn "No items could be packed."
                    bins -> do
                        putStrLn "Recursively computed solution: "
                        let bestExactIndividual = binsToIndividual (length itemSizes) bins
                        putStrLn "As individual: "
                        print bestExactIndividual
                        let packed = sum [itemSizes !! i | bin <- bins, i <- bin]
                        putStrLn $ "Total packed size: " ++ show packed 
                                    ++ " out of " ++ show (sum itemSizes)
                        
        ["-r", filePath] -> do
            evolve filePath "1" False

        [filePath] -> do
            evolve filePath "2" True


        -- if we run "guci binPacking.hs input.txt" and enter just "main"
        -- it will show this useage     
        _ -> putStrLn "Usage: runhaskell knapsack.hs [-r] [-e] <filename> \n\
                            \ -r: run version 1 algorithm \n\
                            \ -e: get exact solution with recursive approach."

        --     putStrLn "Item sizes: "
        --     print itemSizes
        --     putStrLn "Bin capacities: "
        --     print binCaps


        -- -- Assign each item to bin
        --     let testIndividual = [0,1,2,3,4]
        --     putStrLn "Test fitness on individual: "
        --     print testIndividual

        --     let fit = fitness binCaps itemSizes testIndividual
        --     putStrLn "Fitness: "
        --     print fit

        --     let numItems = length itemSizes
        --     let numBins = length binCaps
        --     population <- createPopulation popSize numItems numBins
        --     putStrLn "Random Population: "
        --     mapM_ print population

        --     putStrLn "--- Testing validateIndividual on one random individual ---"
        --     validated <- validateIndividual itemSizes (head population) binCaps
        --     putStrLn "Validated individual: "
        --     print validated
        --     putStrLn $ "Fitness: " ++ show (fitness binCaps itemSizes validated)

        --     putStrLn "--- Testing crossover + validation ---"
        --     let [p1, p2] = take 2 population
        --     (c1, c2) <- crossover p1 p2
        --     c1' <- validateIndividual itemSizes c1 binCaps
        --     c2' <- validateIndividual itemSizes c2 binCaps
        --     putStrLn "Parent 1: "
        --     print p1
        --     putStrLn "Parent 2: "
        --     print p2
        --     putStrLn "Child 1 (validated): "
        --     print c1'
        --     putStrLn "Child 2 (validated): "
        --     print c2'

        --     putStrLn "--- Testing mutation ---"
        --     mutated <- mutation numBins (head population)
        --     putStrLn "Original individual: "
        --     print (head population)
        --     putStrLn "Mutated individual: "
        --     print mutated

        --     putStrLn "--- Testing createNextGeneration ---"
        --     nextGen <- createNextGeneration population itemSizes binCaps
        --     putStrLn "First 3 individuals of next generation: "
        --     mapM_ print (take 3 nextGen)


