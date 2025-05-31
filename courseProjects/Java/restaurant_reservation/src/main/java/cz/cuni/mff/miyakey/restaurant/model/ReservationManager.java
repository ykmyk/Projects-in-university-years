package cz.cuni.mff.miyakey.restaurant.model;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.*;

/**
 * Manage the reservation request by assigning the table, updating the status
 * to CONFIRMED or CANCELLED
 *
 * @author Yuka Miyake
 * @version 1.0
 */


public class ReservationManager {

    /** list of all tables in the restaurant
     *  with each table number and capacity
     */
    private final List<Table> tables;
    private final Map<UUID, Reservation> reservations = new HashMap<>();
    private final Map<UUID, Table> assignedTables = new HashMap<>();

    /**
     * Constructs a reservation manager of given tables
     *
     * @param tables list of table available for booking
     * @throws IllegalArgumentException if the list, tables is empty or null
     */
    public ReservationManager(List<Table> tables) {
        if (tables == null || tables.isEmpty()) {
            throw new IllegalArgumentException("Tables must not be null or empty");
        }
        this.tables = new ArrayList<>(tables);
        this.tables.sort(Comparator.comparingInt(Table::getCapacity));

    }
    /**
     * Create a new reservation in PENDING status,
     * assigning the smallest table more than the partySize
     * at the given date and time
     *
     * @param date          reservation date
     * @param time          reservation time
     * @param partySize     the number of guests
     * @param deposit       the amount of deposit
     * @param customer      the person who made a reservation
     * @return the created Reservation (status == PENDING)
     * @throws IllegalArgumentException if partySize is less than 1 or deposit is less than 0
     * @throws IllegalStateException if no table is available
     * */
    public Reservation createReservation(Customer customer,
                                         LocalDate date,
                                         LocalTime time,
                                         int partySize,
                                         BigDecimal deposit) {
        if(partySize < 1) {
            throw new IllegalArgumentException("Party size must be greater than 0");
        }
        if(deposit.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Deposit must be greater than 0");
        }
        if(customer == null) {
            throw new IllegalArgumentException("Customer must not be null");
        }

        for(Table table : tables) {
            if(table.getCapacity() >= partySize && isTableFree(table, date, time)){
                UUID id = UUID.randomUUID();
                Reservation reservation = new Reservation(
                        id, customer, date, time, partySize, deposit,
                        Reservation.ReservationStatus.PENDING
                );
                reservations.put(id, reservation);
                assignedTables.put(id, table);
                return reservation;
            }
        }

        throw new IllegalArgumentException("No such table for" + partySize +
                "people on" + date + "at" + time);
    }


    /**
     * Checks if a table is free (no PENDING or CONFIRMED reservation) at date/time.
     *
     * @param table the table to check
     * @param date the reservation date
     * @param time the reservation time
     * @return true if table is unassigned at that slot
     */
    public boolean isTableFree(Table table, LocalDate date, LocalTime time) {
        for(Map.Entry<UUID, Reservation> entry : reservations.entrySet()) {
            Reservation res = entry.getValue();
            if(!res.getStatus().equals(Reservation.ReservationStatus.CANCELLED)
            && res.getDate().equals(date) && res.getTime().equals(time)
            && table.equals(assignedTables.get(res.getResId())))
                return false;
        }
        return true;
    }

    /**
     * Confirms a PENDING reservation, and update the status to CONFIRMED
     *
     * @param id the UUID of the wired reservation to update status
     * @return true if the status was PENDING and updated to CONFIRMED
     *         false if not.
     */
    public boolean confirmReservation(UUID id) {
        Reservation res = reservations.get(id);
        if(res != null && res.getStatus() == Reservation.ReservationStatus.PENDING) {
            res.setStatus(Reservation.ReservationStatus.CONFIRMED);
            return true;
        }
        return false;
    }

    /**
     * Cancels a PENDING or CONFIRMED reservation, and update the status to CANCELLED
     * Removing the id from the assigned table
     *
     * @param id the UUID of the wired reservation to update status
     * @return true if the status was PENDING  or CONFIRMED and updated to CANCELLED
     *         false if not.
     */

    public boolean cancelReservation(UUID id) {
        Reservation res = reservations.get(id);
        if(res != null && res.getStatus() != Reservation.ReservationStatus.CANCELLED) {
            res.setStatus(Reservation.ReservationStatus.CANCELLED);
            assignedTables.remove(id);
            return true;
        }
        return false;
    }
    /**
     * Lists all reservations (of any status) on a given date.
     *
     * @param date the date to filter by
     * @return unmodifiable list of matching reservations
     */
    public List<Reservation> listReservations(LocalDate date){
        List<Reservation> res = new ArrayList<>();
        for(Reservation reservation : reservations.values()) {
            if(reservation.getDate().equals(date)) {
                res.add(reservation);
            }
        }
        return Collections.unmodifiableList(res);
    }
    /**
     * Returns the table assigned to a given reservation, if any.
     *
     * @param resId the UUID of the reservation
     * @return Optional of the assigned Table, or empty if not found/cancelled
     */

    public Optional<Table> getAssignedTable(UUID resId) {
        return Optional.ofNullable(assignedTables.get(resId));
    }

    /**
     * Extract all existing reservation
     *
     * @return unmodifiable map of ID -> Reservation
     */

    public Map<UUID, Reservation> getAllReservations() {
        return Collections.unmodifiableMap(reservations);
    }



}
