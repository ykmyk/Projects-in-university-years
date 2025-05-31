package cz.cuni.mff.miyakey.restaurant.web;

import jakarta.ws.rs.ApplicationPath;
import org.glassfish.jersey.server.ResourceConfig;

/**
 * JAX-RS application entry point.
 * All classes in this package annotated with @Path
 * will be published under the "/api" root.
 *
 * @author Yuka Miyake
 * @version 1.1
 */
@ApplicationPath("/api")
public class AppConfig extends ResourceConfig {
    public AppConfig() {
        // scan this package for @Path / @Provider classes
        packages("cz.cuni.mff.miyakey.restaurant.web");
    }
}
