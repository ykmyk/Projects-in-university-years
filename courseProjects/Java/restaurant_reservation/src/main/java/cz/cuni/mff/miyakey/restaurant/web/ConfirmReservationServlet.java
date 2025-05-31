package cz.cuni.mff.miyakey.restaurant.web;
import cz.cuni.mff.miyakey.restaurant.model.ReservationManager;
import cz.cuni.mff.miyakey.restaurant.persistence.JsonReservation;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.*;
import java.io.IOException;
import java.util.UUID;


/**
 * HTTP endpoint to confirm an existing PENDING status reservation.
 * This expects a POST with parameters
 *
 * - id ... the UUID of the reservation to confirm
 * - date ... the date filter to return to
 *
 * After successful confirmation, persists changes and redirects back to
 * '/reservation?date=...'
 *
 * URL mapping: {@code /reservation/cancel}
 *
 * @author Yuka Miyake
 * @version 1.0
 */

@WebServlet("/reservations/confirm")
public class ConfirmReservationServlet extends HttpServlet {



    @Override
    protected void doPost(HttpServletRequest req,
                          HttpServletResponse resp)
            throws ServletException, IOException {
        var ctx     = getServletContext();
        var manager = (ReservationManager) ctx.getAttribute("reservationManager");
        var repo    = (JsonReservation) ctx.getAttribute("reservationRepo");

        UUID id = UUID.fromString(req.getParameter("id"));
        manager.confirmReservation(id);
        repo.saveAll(manager.getAllReservations());

        // keep same date filter on return
        String date = req.getParameter("date");
        resp.sendRedirect(req.getContextPath()
                + "/reservations?date=" + date);
    }
}