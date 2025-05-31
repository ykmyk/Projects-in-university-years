package cz.cuni.mff.miyakey.restaurant.web;

import cz.cuni.mff.miyakey.restaurant.model.Reservation;
import cz.cuni.mff.miyakey.restaurant.model.ReservationManager;

import jakarta.servlet.ServletContext;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.*;
import java.time.LocalDate;
import java.util.List;
import java.util.UUID;

/**
 * JAX-RS resource providing a JSON API for reservations.
 *
 * Base URL: /api/reservations
 *   GET '/api/reservations?date=YYYY-MM-DD' → list
 *   GET '/api/reservations/{id}'           → single
 *
 * Responses are in application/json via Jackson.
 *
 * @author Yuka Miyake
 * @version 1.1
 */
@Path("/reservations")
@Produces(MediaType.APPLICATION_JSON)
public class ReservationResource {

    /** Injected servlet context to access the ReservationManager. */
    @Context
    private ServletContext servletContext;

    /**
     * Lists all reservations for the given date
     * (if missing, the default value is today).
     *
     * @param dateParam optional date in YYYY-MM-DD
     * @return HTTP 200 with JSON array of Reservation objects
     */
    @GET
    public Response getAll(@QueryParam("date") String dateParam) {
        LocalDate date = (dateParam != null && !dateParam.isBlank())
                ? LocalDate.parse(dateParam)
                : LocalDate.now();

        ReservationManager mgr = (ReservationManager)
                servletContext.getAttribute("reservationManager");
        List<Reservation> list = mgr.listReservations(date);

        return Response.ok(list).build();
    }

    /**
     * Fetches a single reservation by UUID.
     *
     * @param id the reservation UUID
     * @return 200+JSON if found, 404 otherwise
     */
    @GET
    @Path("{id}")
    public Response getOne(@PathParam("id") String id) {
        ReservationManager mgr = (ReservationManager)
                servletContext.getAttribute("reservationManager");
        UUID uuid = UUID.fromString(id);
        return mgr.getAllReservations().containsKey(uuid)
                ? Response.ok(mgr.getAllReservations().get(uuid)).build()
                : Response.status(Response.Status.NOT_FOUND).build();
    }
}
