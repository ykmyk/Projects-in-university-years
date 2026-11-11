# Specification of the final project for relevant C# courses
(When modifying this document, please maintain the layout and structure and follow the inline instructions.)

## C# Courses selection
(Change `[ ]` to `[x]` for the courses you plan to use this final project for.)

- [x] NPRG035 (Programming in C# language | Programování v jazyce C#)
- [x] NPRG038 (Advanced C# Programming | Pokročilé programování v jazyce C#)
- [ ] NPRG057 (Advanced .NET Programming II | Pokročilé programování pro .NET II)
- [ ] NPRG064 (Programming user interfaces in .NET | Programování uživatelských rozhraní v .NET)

## Specification

### FocusTube: A policy-based content filter and time management engine for YouTube 
---
 - Motivation

    Motivation came from my own experience. 
    I like watching YouTube but often waste too much time on it that could have used to do some other more meaningful thing.
    However, I don't want to completely block my youtube website since I also use Youtube for learning/studying.
    So this time, I want to solve this dilemma by impelmenting FocusTube, and enable to manage and use time more efficiently.
    It allows users to access selected field of chanels only while restricting the use of other "noisy" field (Gaming, Entertainment, etc), heping user control the useage without losing access to study tools.

 - Use case scenarios: 
 
    - The focused Student: 
    
        A user need to watch a "C# generic interface tutorial". FocusTube permits this but automatically blocks if the user tries to move to irrerevant video and watch more than the daily limit(set by user).


    - For the daily use:

        The user can set the daily limit for each field of videos, while they set unlimited for music, they can limit shorts or entertaiment for 30 mins.
        This allows user to use Youtube for back ground music to motivate their daily mood, but prevent wasting their time on unwanted shorts/video unconsiously.

 - Main Features:

    For this project, I will forcus on implementing the base logic as written above and showing it by using API. 

    I expect user to paste a link to a box, and the application accumilates the info of video(category/duration) and shows "blocking" sign after exceeding limit.

    As I aggreed with Mr. Filip Kliber, I will add some features if the volume of the file won't exceed 45kB, otherwise I will stay on the basic fetures for this time.
 
    - Video Metadata Auditing

        The application accepts a YouTube URL by communitating with the YouTube Data API v3, and retrives the video's category and channel information.

    - Time management

        The application will track "Time spent" on each categories, and total.

    - Personal and dynamic setting

        User can define their personal preffered setting for each category and also some channel specific. They can also set "Free" category that they allow themsleves to watch unlimitedly, by excluding from daily limit. (ex. Education, Music)
    - Rule Enforcement

        The application has the logic that comparing current session data to the user-defined limits(both Global limits and Category limit).

        
 - UI/UX:

    For this version, the application will be a WebAPI.
    I plan to use the built-in Swagger UI for user to interacts with application.  I may update if I find better option.

 - Persistence: 
 
    Data will be stored using JSON File persistence.

 - Libraries/Technologies:
    - Google.Apis.YouTube.v3: Official NuGet package for YouTube data retrieval.
    - xUnit / FluentAssertions: For a comprehensive testing suite.
    Following might change, but most likely using them.
    - ASP.NET Core WebAPI: The primary framework.
    - Newtonsoft.Json / System.Text.Json: For data persistence.

 - Testing: 
    - Unit Tests: To verify basic functionality of management logic.
    - Integration Tests: I may add for the API functionality, if the volume of the code is not enough.

I did the brainstorming with Gemini to decide my project topic, the conversation link is the following.
https://gemini.google.com/share/da58a26cbea2