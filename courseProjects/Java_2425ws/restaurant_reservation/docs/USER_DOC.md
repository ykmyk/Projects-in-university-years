# Restaurant reservation system

A simple java web application for the restaurant reservation system.
At this point, I expect restaurant owner to enter the reservation, so the reservation creating page and reservation management page(changing the reservation status) is at the same end.

The user has two option of interface
- Command line mode
- Web Application mode


## What the user can do

### Make a reservation
The user can make the reservation with the customer name, email. phone number, then the reservation date, time and size of party.

### Change the reservation status
The initial reservation state is "PENDING".
After the PENDING reservation is created, the user can "CONFIRM" or "CANCEL" the reservation.

### List and filter the reservation
Once the reservation is made, all the reservation is listed.
The user can filter the reservation with the some original key, like name or email.


## How to launch

### Prerequisites
- **Java 23 JDK** (tested with JDK 23+37)
- **Maven 3.8+**
- A standards-compliant servlet container (Jetty, Tomcat, WildFly…), or use the built-in Jetty plugin.

### Pre-running
1. Go to the directory 'restaurant_reservation'
2. run `mvn clean package compile`

### For command line mode(after Pre-running)
1. Run `mvn exec:java`
2. Then it will show
   "Welcome! Enter commands: (to see possible command, type 'help'"


### For web application mode(after Pre-running)
1. Run `mvn jetty:run`
2. The browser is available at `http://localhost:8080/reservations`


## How to test
Just run `mvn clean test` at 'restaurant_reservation' directory.

