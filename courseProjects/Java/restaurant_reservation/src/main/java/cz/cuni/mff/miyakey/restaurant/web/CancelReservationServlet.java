package cz.cuni.mff.miyakey.restaurant.web;

import cz.cuni.mff.miyakey.restaurant.model.ReservationManager;
import cz.cuni.mff.miyakey.restaurant.persistence.JsonReservation;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.*;
import java.io.IOException;
import java.util.UUID;

/**
 * HTTP endpoint to cancel an existing (either PENDING/CONFIRMED) reservation.
 * This expects a POST with parameters
 *
 * - id ... the UUID of the reservation to cancel
 * - date ... the date filter to return to
 *
 * After successful cancelling, persists changes and redirects back to
 * '/reservation?date=...'
 *
 * URL mapping: {@code /reservation/cancel}
 *
 * @author Yuka Miyake
 * @version 1.0
 */

@WebServlet("/reservations/cancel")
public class CancelReservationServlet extends HttpServlet {

    /**
     * Cancels the reservation whose UUID is in the request,
     * then redirects back to the list page for the same date.
     *
     * @param req  the HttpServletRequest, must contain “id” & “date”
     * @param resp the HttpServletResponse, used for redirect
     * @throws ServletException on servlet errors
     * @throws IOException      on I/O errors during redirect
     */
    @Override
    protected void doPost(HttpServletRequest req,
                          HttpServletResponse resp)
            throws ServletException, IOException {
        var ctx     = getServletContext();
        var manager = (ReservationManager) ctx.getAttribute("reservationManager");
        var repo    = (JsonReservation) ctx.getAttribute("reservationRepo");

        UUID id = UUID.fromString(req.getParameter("id"));
        manager.cancelReservation(id);
        repo.saveAll(manager.getAllReservations());

        String date = req.getParameter("date");
        resp.sendRedirect(req.getContextPath()
                + "/reservations?date=" + date);
    }
}
