package cz.cuni.mff.miyakey.restaurant.model;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class ReservationManagerTest {

    private ReservationManager mgr;
    private List<Table> tables;
    private Customer customer;

    @BeforeEach
    void setUp() {
        tables = List.of(new Table(1, 2), new Table(2,4));
        mgr = new ReservationManager(tables);
        customer = new Customer(
                UUID.randomUUID(), "Name", "test@test.com", "000"
        );
    }

    @Test
    void createReservation() {
        Reservation r1 = mgr.createReservation(
                customer, LocalDate.now(), LocalTime.NOON, 2, BigDecimal.ZERO);
        Optional<Table> optT1 = mgr.getAssignedTable(r1.getResId());
        assertEquals(optT1.isPresent(), true);
        Table t1 = optT1.get();
        assertEquals(1, t1.getTableNumber());

        Reservation r2 = mgr.createReservation(
                customer, LocalDate.now(), LocalTime.NOON, 3, BigDecimal.ZERO);
        Optional<Table> optT2 = mgr.getAssignedTable(r2.getResId());
        assertEquals(optT2.isPresent(), true);
        Table t2 = optT2.get();
        assertEquals(2, t2.getTableNumber());

    }

    @Test
    void twoBookingsOnDifferentTables() {
        Reservation r1 = mgr.createReservation(customer, LocalDate.now(), LocalTime.NOON, 2, BigDecimal.ZERO);
        Reservation r2 = mgr.createReservation(customer, LocalDate.now(), LocalTime.NOON, 2, BigDecimal.ZERO);
        assertEquals(2,
                mgr.getAssignedTable(r2.getResId()).get().getTableNumber()
        );
    }

    @Test
    void confirmAndCancelReservation() {
        Reservation r1 = mgr.createReservation(
                customer, LocalDate.now(), LocalTime.NOON, 2, BigDecimal.ZERO);
        UUID id = r1.getResId();
        assertTrue(mgr.confirmReservation(id));
        assertEquals(Reservation.ReservationStatus.CONFIRMED, r1.getStatus());

        assertTrue(mgr.cancelReservation(id));
        assertEquals(Reservation.ReservationStatus.CANCELLED, r1.getStatus());

        assertDoesNotThrow(() -> mgr.createReservation(
                customer, LocalDate.now(), LocalTime.NOON, 2, BigDecimal.ZERO));
    }

    @Test
    void listReservations() {
        LocalDate d1 = LocalDate.now();
        LocalDate d2 = d1.plusDays(1);

        Reservation res = mgr.createReservation(customer, d1, LocalTime.NOON, 2, BigDecimal.ZERO);
        mgr.createReservation(customer, d2, LocalTime.NOON, 1, BigDecimal.ZERO);

        List<Reservation> list = mgr.listReservations(d1);
        assertEquals(1, list.size());
        assertTrue(list.contains(res));
    }

}
