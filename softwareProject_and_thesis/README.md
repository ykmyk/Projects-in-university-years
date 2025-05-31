# Individual Software Project and Thesis plan
Here is an overall plan for Individual Software Project(we shall call it ISP in the following) and its extended bachelor thesis.

## Overall plan
I am planning to work on on the meal planner, that suggests you the meal plan for 1 week with its recipe based on ingredients set as the default  (always available), and typed as currently available/preferred. 

In the ISP, I will implement WebApplication for this meal planner that has basic features as a web app and suggest you 1 week meal plan randomly. 
Based on the feature implemented on ISP, I will extend my topic to the theses with several optimization for generating 1 week meal plan. 
Detail specification on ISP and thesis can be found in the following. 


##  ISP detail
In ISP, I plan to construct basic web application with (LANGUAGE) 
The goal of the ISP is implementing the web application that has basic feature as and app and generate meal plan for a week.
### Expected feature of meal plan app
- The user can set default ingredients 
- The user can set available ingredients each time
- The app shows the random 1 week meal plan that can be done with the entered ingredients
- The user can swap the meal plan within the week(s.t. Monday menu to Tuesday menu)
- If the user didn't like the meal plan, then they can ask for the regeneration
- The user can partially change/delete the meal plan of the week

### meal plan generation algorithm
In the ISP version, we don't plan to implement "clever" algorithm on it.
It will generate just random meal plan depending on the selected ingreditens

## The addtional feature planned after first version

### Additional feature
- Generate the shopping list to cook the givem meal plan
- Increase the variety of the recipe(database)
- Parsing to the current user security(creating the user account etc.)

## Thesis extension detail
The main goal of the thesis is optimize the meal plan generation and compare the result depending on the algorithm we used.

### Optimization idea (not yet decided)
- Optimization on the food which they don't eat
- Optimization of the nurtrition 
- Optimization of the user's preference(include more that user picked previously, less that user deleted/changed etc.
