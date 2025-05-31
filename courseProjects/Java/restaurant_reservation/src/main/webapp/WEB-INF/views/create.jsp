<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head><title>New Reservation</title></head>
<body>
<h1>Create Reservation</h1>
<form action="${pageContext.request.contextPath}/reservations/create"
      method="post">
    Name: <input type="text" name="name" required/><br/>
    Email: <input type="email" name="email" required/><br/>
    Phone: <input type="text" name="phone" required/><br/>
    Date: <input type="date" name="date" required/><br/>
    Time: <input type="time" name="time" required/><br/>
    Party Size: <input type="number" name="partySize" min="1" required/><br/>
    <button type="submit">Submit</button>
</form>
<p><a href="${pageContext.request.contextPath}/reservations">Back to list</a></p>
</body>
</html>
