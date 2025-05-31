package cz.cuni.mff.miyakey.restaurant.web;
import cz.cuni.mff.miyakey.restaurant.model.Reservation;
import cz.cuni.mff.miyakey.restaurant.model.ReservationManager;
import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.*;
import java.io.IOException;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * HTTP endpoint to list all reservation for a given date.
 * And optionally filter reservations for a given date.
 *
 * Supports two query parameters
 * - 'date=YYYY-MM-DD' ... which day to list up / filter
 *                          if the parameter is missing,
 *                          uses today's date as a default
 * - 'key='            ... key text to search for
 *                          in customer name or email address
 *
 * Fetches {@link ReservationManager} from the servlet context,
 * and retrieves the list.
 *
 * After applying date + filter, sets request attributes:
 * - 'reservations' -> List of Reservation
 * - 'date'         -> String of the date
 * - 'key'          -> the original filter text (possibly null)
 *
 * and 'date' then forwards to '/WEB-INF/views/list.jsp' for rendering.
 *
 * @author Yuka Miyake
 * @version 1.1
 */
@WebServlet("/reservations")
public class ListReservationsServlet extends HttpServlet {

    /**
     * RHandles GET /reservations requests by loading, filtering, and forwarding.
     *
     * @param req  may contain 'date' parameter
     * @param resp used to forward to JSP
     * @throws ServletException on failure loading JSP
     * @throws IOException      on I/O error
     */
    @Override
    protected void doGet(HttpServletRequest req,
                         HttpServletResponse resp)
            throws ServletException, IOException {

        ReservationManager manager =
                (ReservationManager) getServletContext().getAttribute("reservationManager");

        // parse 'all' flag
        boolean showAll = Boolean.parseBoolean(req.getParameter("all"));

        // pull dateParam; if it's null, empty, or the literal "null", treat as missing
        String dateParam = req.getParameter("date");
        boolean hasDateParam = dateParam != null
                && !dateParam.isBlank()
                && !"null".equalsIgnoreCase(dateParam);

        // if we're not in "show all" mode but the dateParam is missing, default to today
        LocalDate date = hasDateParam
                ? LocalDate.parse(dateParam)
                : LocalDate.now();

        // choose data
        List<Reservation> list = showAll
                // flat-map all reservations
                ? new ArrayList<>(manager.getAllReservations().values())
                // filter by date
                : manager.listReservations(date);

        // optional text filter
        String key = req.getParameter("key");
        if (key != null && !key.isBlank()) {
            String lc = key.toLowerCase(Locale.ROOT);
            list = list.stream()
                    .filter(r ->
                            r.getCustomer().getName().toLowerCase(Locale.ROOT).contains(lc)
                                    || r.getCustomer().getEmail().toLowerCase(Locale.ROOT).contains(lc))
                    .collect(Collectors.toList());
        }

        // stash into request
        req.setAttribute("showAll", showAll);
        req.setAttribute("date", date.toString());
        req.setAttribute("key", key);
        req.setAttribute("reservations", list);

        // forward
        req.getRequestDispatcher("/WEB-INF/views/list.jsp")
                .forward(req, resp);
    }
}