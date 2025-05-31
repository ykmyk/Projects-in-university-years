package cz.cuni.mff.miyakey.restaurant.web;


import cz.cuni.mff.miyakey.restaurant.model.Customer;
import cz.cuni.mff.miyakey.restaurant.model.Reservation;
import cz.cuni.mff.miyakey.restaurant.model.ReservationManager;
import cz.cuni.mff.miyakey.restaurant.persistence.JsonReservation;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.*;
import java.io.IOException;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.UUID;


/**
 * HTTP endpoint for creating a new reservation.
 *
 *  GET requests forward to the JSP form at /WEB-INF/views/create.jsp
 *
 *  POST requests read parameters
 *  (name, email, phone, date, time, partySize),
 *  build a {@link Customer},
 *  compute the deposit
 *  (5.00 per guest for now but will be updated on @version 2.0 to
 *  the minimum deposit amount that each restaurant define),
 *  call {@link ReservationManager#createReservation},
 *  then persist and redirect back to the listing for that date.
 *
 * URL mapping: {@code /reservations/create}
 *
 * @author Yuka Miyake
 * @version 1.0
 */

@WebServlet("/reservations/create")
public class CreateReservationServlet extends HttpServlet {
    /**
     * Serves the booking form.
     */
    @Override
    protected void doGet(HttpServletRequest req,
                         HttpServletResponse resp)
            throws ServletException, IOException {
        // show form
        req.getRequestDispatcher("/WEB-INF/views/create.jsp")
                .forward(req, resp);
    }


    /**
     * Processes form submissions to create a reservation.
     *
     * @param req  must contain form fields:
     *             (name, email, phone, date, time, partySize)
     * @param resp used to redirect to the list page
     * @throws ServletException on servlet failure
     * @throws IOException      on I/O or redirect failure
     */
    @Override
    protected void doPost(HttpServletRequest req,
                          HttpServletResponse resp)
            throws ServletException, IOException {
        var ctx = getServletContext();
        var manager = (ReservationManager) ctx.getAttribute("reservationManager");
        var repo    = (JsonReservation) ctx.getAttribute("reservationRepo");

        // read parameters
        String name  = req.getParameter("name");
        String email = req.getParameter("email");
        String phone = req.getParameter("phone");
        LocalDate date = LocalDate.parse(req.getParameter("date"));
        LocalTime time = LocalTime.parse(req.getParameter("time"));
        int partySize    = Integer.parseInt(req.getParameter("partySize"));
        BigDecimal deposit = BigDecimal.valueOf(partySize).multiply(BigDecimal.valueOf(5));

        // create customer + reservation
        Customer customer = new Customer(UUID.randomUUID(), name, email, phone);
        Reservation res = manager.createReservation(
                customer, date, time, partySize, deposit
        );

        // persist change
        repo.saveAll(manager.getAllReservations());

        // redirect to list for that date
        resp.sendRedirect(req.getContextPath()
                + "/reservations?date=" + date);
    }
}