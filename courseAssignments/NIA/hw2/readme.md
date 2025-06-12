# HW2 Yuka Miyake
## Content of the hw2 directory
### Codes for the task
- Main.py
Run VehivleRouting.py and generate graph, output the best_solution
- VehicleRouting.py
Main algorithm code
- parseXML.py
Parsing the XML to be used in VehicleRouting.py

### Given dataset
- data_32.xml
- data_72.xml
- data_422.xml

### Additional python file
- solutionFormatting.py

### Outcomes
- bestSolution.txt
You can fine the best_solution for each dataset with my algorithm
- convergenceGraph directory
It has .png files of the convergenceGraph for the three given dataset

### Other
- problemAdaptation.md
explaining how I adapted the tutorial code to solve this exercise.

## Running instruction
1. Store .xml file that you want to test in the same directory as Main.py, parseXML.py and VehicleRouting.py
2. Open the Main.py file and change following to your filename 
```
line 7, parsed_data = parse_xml('change_to_your_filename')
line 31, draw(hist, "change_to_your_filename")
```
3. Run following command in the terminal
```
python3 Main.py
```
4. It will shows the convergence graph of the given dataset
5. Once you close the tab of the graph, it will provide you the best_solution in the form of
```
[2, 3, 23, 28, 4, 0, 3, ...]
```
6. Copy that output.
7. Open solutionFormatting.py and paste the output to 
```
best_solution = paste_your_solution
```

8. Save your solutionFormatting.py
9. Go back to the terminal and run following command
```
python3 solutionFormatting.py
```
10. You will get a best_solution with nicer looking.