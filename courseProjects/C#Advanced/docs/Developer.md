# FocusTube - Developer Guide
FocusTube is a Web Application that is developed for the student/working people who is struggling 
to manage the use of the Youtube with distractive contents while they still want to use it to learn something.
For detail specification and objective of this project, please take a look at spec/README.md file.

## How the Application is Designed
FocusTube is a Web API built with C# and ASP.NET Core. 
The app is split into distinct layers(front-end and back-end) and the projects accordingt to the role.
For the front end, we use HTML, CSS and JavaScript, and the C# for the back-end.

Right now, we do not use a database. All the user's watch history and limit settings are stored in-memory 
(in lists and dictionaries) inside the WatchGuardService. 
This means if you restart the server, the data resets.

## Project Layout (The Solution)
The solution is divided projects to keep things organized.
We mainly have 4 core projects, **FocusTube.API, FocusTube.Core, FocusTube.Service, and FocusTube.Tests**.

### FocusTube.API:
This project mainly has the files related to front-end layer, simple HTML, CSS and JavaScript files 
in internal wwwroot folder, json files for API, and also has ```Controllers(VideoController and SettingsController)```
as a ccak-end written in C#, which recieved the request from the user.
We also have a main Program.cs file to launch entire Web application, in this API folder.

### FocusTube.Core:
This project holds our main domain model, which is the ```TimeQuata``` class.
It has no dependencies on other parts of the app.

### FocusTube.Service:
This project has the main "Brain" part of the application. 
This contains the ```WatchGuardService```, which handles all the limit computation and tracking and 
the```YouTubeIntegrationService```, which manage the communication with the YouTube's API.

### FocusTube.Tests
This project consist of the xUnit tests that verify the functionality of the code.

### Documentations
Apart from the main folders for the project, we have this ```docs/Developer.md``` as a developer documentation,
```spec/README.md``` as a specification and ```/README.md``` as a user documentation.

## How User Input is Handled
The user input is first handled by the project ```FocusTube.API``` then passed to the further projects.

When a user interacts with the app (like pasting a link or changing a limit), 
our application is handling the input data as follows:

1. The user clicks a button on the HTML page (e.g., ```Watch Video``` button or ```Save all change``` button, etc.).

2. JavaScript codes in ```FocusTube.API/wwwroot``` catches this click and sends a fetch() request to our C# API.3

3. The corresponding Controller (like ```SettingsController.cs```) receives this data using 
the object ```TimeSetting``` to safely read the JSON content.

4. The Controller passes the data to the ```WatchGuardService``` to actually process the data and save the changes.

5. The Controller sends a success or error message back to the JavaScript, which updates the screen for the user.

## Key Algorithms
All the algorithm involved in this app is quite simple, and I mention one logic 
to consider Hierarchical Limits of the time spend.

Inside AuditVideo, the code doesn't just check one limit. 
It checks all of the Total limit, then the Category limit, then the Channel limit. 
If any of them are over the limit, it blocks the video and builds a custom error(corresponding to the main reason of the limit exceed) 
string explaining exactly which rules were broken.


## Key features from C# that you need to know

### API:
API is the very fundamental feature on this project, which you need to know how to deal with.
### Custom Operators: 
In the TimeQuota.cs class, we overloaded the < and - operators. 
This allows the rest of the code to just do math on the objects more naturally, 
like _totalQuota - duration or if (_totalQuota < duration).

### LINQ for Analytics: 
To figure out how much time a user spent on specific categories or channels, 
we use LINQ in GetDailyUsageStats(). 
It uses .GroupBy() to group the video history by category, and .Sum() to add up all the durations.

### Delegates and Events: 
If a user breaks a rule/limit, the WatchGuardService triggers a custom event delegate called OnLimitExceeded.

## How to Extend the Application
If you want to add new features to FocusTube, here is what I would suggest:

### Add a Real Database: 
Right now, the _history is just a List<VideoLog> and just hold the temporary memory, 
which will be lost once the user relaunch the app. 
You could replace this with Entity Framework Core and an SQLite database (or any other database)
so the user's history is saved permanently even if the server turns off.


### User Accounts: 
You could add authentication (like ASP.NET Identity) so multiple students can use 
the same server with their own personal limits.