package cz.cuni.mff.miyakey.restaurant.model;


import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.UUID;

/**
 *  Contains information about the reservation date, time,
 *  the size of party(the number of people), deposit amount,
 *  and current status
 *
 *  This class is JSOn-serializable via Jackson
 *
 * @author Yuka Miyake
 * @version 1.0
 */

public class Reservation {
    private final UUID resId;
    private final Customer customer;
    private final LocalDate date;
    private final LocalTime time;
    private final int partySize;
    private final BigDecimal deposit;
    private ReservationStatus status;

    /**
     * Construct a new Reservation
     * @param resId     unique identifier
     * @param customer  who made a reservation
     * @param date      reservation date
     * @param time      reservation time
     * @param partySize number of guests
     * @param deposit   deposit amount
     * @param status    reservation status
     */
    @JsonCreator
    public Reservation(@JsonProperty("resId") UUID resId,
                       @JsonProperty("customer") Customer customer,
                       @JsonProperty("date") LocalDate date,
                       @JsonProperty("time") LocalTime time,
                       @JsonProperty("partySize") int partySize,
                       @JsonProperty("deposit") BigDecimal deposit,
                       @JsonProperty("status") ReservationStatus status) {
        this.resId = resId;
        this.customer = customer;
        this.date = date;
        this.time = time;
        this.partySize = partySize;
        this.deposit = deposit;
        this.status = status;
    }

    /** @return identify the reservation */
    public UUID getResId() {
        return resId;
    }

    /** @return reservation date*/
    public LocalDate getDate() {
        return date;
    }

    /** @return reservation time*/
    public LocalTime getTime() {
        return time;
    }

    /** @return the customer who made this reservation*/
    public Customer getCustomer() {
        return customer;
    }

    /** @return number of guests*/
    public int getPartySize() {
        return partySize;
    }

    /** @return the amount of deposit*/
    public BigDecimal getDeposit() {
        return deposit;
    }

    /** @return current status of the reservation*/
    public ReservationStatus getStatus() {
        return status;
    }

    /** @return update the status of reservation
     *
     * @param newStatus the new status to apply
     */
    public void setStatus(ReservationStatus newStatus){
        this.status = newStatus;
    }

    @Override
    public String toString() {
        return "Reservation{" +
                "id=" + resId +
                ", date" + date +
                ", time" + time +
                ", partySize=" + partySize +
                ", deposit=" + deposit +
                ", status=" + status +
                '}';
    }

    /**
     * the option for the reservation status
     */
    public enum ReservationStatus {
        /** the reservation request is submitted,
          * but neither confirmed nor cancelled */
        PENDING,
        /** the restaurant has confirmed the reservation */
        CONFIRMED,
        /** the reservation has been cancelled
         *  (by the customer or restaurant) */
        CANCELLED
    }
}
