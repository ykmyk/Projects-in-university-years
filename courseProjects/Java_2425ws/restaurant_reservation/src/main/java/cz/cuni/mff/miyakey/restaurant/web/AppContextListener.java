package cz.cuni.mff.miyakey.restaurant.web;

import cz.cuni.mff.miyakey.restaurant.model.ReservationManager;
import cz.cuni.mff.miyakey.restaurant.persistence.JsonReservation;
import cz.cuni.mff.miyakey.restaurant.model.Table;

import jakarta.servlet.ServletContext;
import jakarta.servlet.ServletContextEvent;
import jakarta.servlet.ServletContextListener;
import jakarta.servlet.annotation.WebListener;

import java.io.File;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;


/**
 * Application context listener.
 * When the webapp starts/be launched
 * this will initialize the restaurant’s {@link ReservationManager}
 * and JSON‐backed {@link JsonReservation}.
 * Also, this will persist data back when it shuts down.
 *
 * On startup it:
 *
 *   Creates /WEB-INF/views directory if missing
 *   Instantiates JsonReservation on reservations.json
 *   Loads existing reservations into a new ReservationManager
 *   Saves both objects into the ServletContext under
 *       “reservationManager” and “reservationRepo”.
 * On shutdown it reads them back and calls {@code saveAll(...)}.
 *
 * Note: uses Servlet 4.0 annotations; no web.xml required.
 *
 * @author Yuka Miyake
 * @version 1.0
 */

@WebListener
public class AppContextListener implements ServletContextListener {
    /**
     * Called when the servlet context is initialized
     * (when the webapp is launched)
     * Sets up persistence and manager singletons.
     *
     * @param sce ServletContextEvent that will be initialized
     */
    @Override
    public void contextInitialized(ServletContextEvent sce) {
        ServletContext servletContext = sce.getServletContext();

        // define the tables that restaurant has
        // now this is the fixed number since we suppose to manage one specific restaurant.
        // better to update in @version 2.0 to be more flexible
        // when we start managing more than one restaurant
        List<Table> tables = Arrays.asList(
                new Table(1, 2),
                new Table(2, 4),
                new Table(3, 4),
                new Table(4, 6),
                new Table(5, 8)
        );

        // ensure data directory exists under WEB-INF/views
        String dataDir = servletContext.getRealPath("/WEB-INF/views");
        new File(dataDir).mkdirs();
        File dataFile = Paths.get(dataDir, "reservations.json").toFile();

        // create & load JsonReservation: repo
        JsonReservation repo = new JsonReservation(dataFile);
        Map<UUID, ?> loaded = repo.loadAll();

        // create the reservation manager (we ignore the loaded for now)
        ReservationManager manager = new ReservationManager(tables);

        // store both in application scope for use by servlets
        servletContext.setAttribute("reservationManager", manager);
        servletContext.setAttribute("reservationRepo", repo);

    }

    /**
     * Called when the servlet context is about to be closed.
     * (when the webapp is shut down)
     * Persists all in-memory reservations back to disk.
     *
     * @param sce ServletContextEvent that was closed.
     */
    @Override
    public void contextDestroyed(ServletContextEvent sce){
        ServletContext servletContext = sce.getServletContext();
        var manager = (ReservationManager) servletContext.getAttribute("reservationManager");
        var repo = (JsonReservation) servletContext.getAttribute("reservationRepo");
        repo.saveAll(manager.getAllReservations());
    }
}
