<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="java.util.List" %>
<%@ page import="cz.cuni.mff.miyakey.restaurant.model.Reservation" %>
<%@ page import="cz.cuni.mff.miyakey.restaurant.model.ReservationManager" %>
<%@ page import="cz.cuni.mff.miyakey.restaurant.model.Table" %>
<html>
<head>
    <title>All Reservations</title>
</head>
<body>
<%-- pull our two “filter” values out of the request --%>
<%
    String date = (String)request.getAttribute("date");
    String key  = (String)request.getAttribute("key");
    if (date == null) date = "";
    if (key  == null) key  = "";
%>

<h1>
    All Reservations
    <% if (!date.isBlank()) { %>
    on <%= date %>
    <% } %>
</h1>

<p>
    <a href="<%=request.getContextPath()%>/reservations/create">
        New Reservation
    </a>
    &nbsp;|&nbsp;
    <a href="<%=request.getContextPath()%>/reservations?date=<%=date%><%= key.isBlank() ? "" : "&key="+key %>">
        Show by Date
    </a>
    &nbsp;|&nbsp;
    <a href="<%=request.getContextPath()%>/reservations">
        Show All
    </a>
</p>

<%-- combined date + text filter form --%>
<form action="<%=request.getContextPath()%>/reservations" method="get">
    <label for="date">Date:</label>
    <input type="date" id="date" name="date" value="<%=date%>"/>

    <label for="key">Search:</label>
    <input
            type="text"
            id="key"
            name="key"
            placeholder="Name or email…"
            value="<%=key%>"
    />

    <button type="submit">Apply</button>
</form>

<%-- fetch the manager + the already-filtered list from the request --%>
<%
    ReservationManager manager =
            (ReservationManager)application.getAttribute("reservationManager");
    List<Reservation> reservations =
            (List<Reservation>)request.getAttribute("reservations");
%>

<table border="1" cellpadding="4">
    <tr>
        <th>ID</th>
        <th>Date</th>
        <th>Customer</th>
        <th>Time</th>
        <th>Party</th>
        <th>Deposit</th>
        <th>Status</th>
        <th>Table</th>
        <th>Actions</th>
    </tr>
    <% for (Reservation r : reservations) {
        Table t = manager.getAssignedTable(r.getResId()).orElse(null);
    %>
    <tr>
        <td><%= r.getResId() %></td>
        <td><%= r.getDate() %></td>
        <td><%= r.getCustomer().getName() %>
            (<%= r.getCustomer().getEmail() %>)</td>
        <td><%= r.getTime() %></td>
        <td><%= r.getPartySize() %></td>
        <td><%= r.getDeposit() %></td>
        <td><%= r.getStatus() %></td>
        <td><%= t != null ? t.getTableNumber() : "N/A" %></td>
        <td>
            <% if (r.getStatus() == Reservation.ReservationStatus.PENDING) { %>
            <form
                    action="<%=request.getContextPath()%>/reservations/confirm"
                    method="post"
                    style="display:inline"
            >
                <input type="hidden" name="id"   value="<%=r.getResId()%>"/>
                <input type="hidden" name="date" value="<%=date%>"/>
                <input type="hidden" name="key"  value="<%=key%>"/>
                <button type="submit">Confirm</button>
            </form>
            <% } %>

            <% if (r.getStatus() != Reservation.ReservationStatus.CANCELLED) { %>
            <form
                    action="<%=request.getContextPath()%>/reservations/cancel"
                    method="post"
                    style="display:inline"
            >
                <input type="hidden" name="id"   value="<%=r.getResId()%>"/>
                <input type="hidden" name="date" value="<%=date%>"/>
                <input type="hidden" name="key"  value="<%=key%>"/>
                <button type="submit">Cancel</button>
            </form>
            <% } %>
        </td>
    </tr>
    <% } %>
</table>
</body>
</html>
