package cz.cuni.mff.miyakey.restaurant.persistence;

import cz.cuni.mff.miyakey.restaurant.model.Reservation;
import java.util.*;

/**
 * Generic persistence interface to store and retrieve
 * a collection of {@link Reservation} instances with key; {@link UUID}.
 *
 * Implementations may choose any backing store (JSON, CSV, database, etc.),
 *
 * @author You
 * @version 1.0
 */

public interface Repository {
    /**
     * Loads all reservations from the backing store.
     *
     * @return a Map of reservation ID(UUID) to Reservation, never null
     */
    Map<UUID, Reservation> loadAll();


    /**
     * Persists the given collection of reservations.
     *
     * @param reservations the created reservations to store; never null
     */
    void saveAll(Map<UUID, Reservation> reservations);
}
