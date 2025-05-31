package cz.cuni.mff.miyakey.restaurant.persitent;

import cz.cuni.mff.miyakey.restaurant.model.Customer;
import cz.cuni.mff.miyakey.restaurant.model.Reservation;
import cz.cuni.mff.miyakey.restaurant.persistence.JsonReservation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

public class JsonReservationTest {
    private File tempFile;
    private JsonReservation repo;

    @BeforeEach
    void setUp() throws Exception {
        tempFile = File.createTempFile("test", ".json");
        tempFile.deleteOnExit();
        repo = new JsonReservation(tempFile);
    }

    @Test
    void saveAndLoad() {
        Reservation r1 = new Reservation(
                UUID.randomUUID(),
                new Customer(UUID.randomUUID(), "sample", "sample@s.com", "000"),
                LocalDate.now(), LocalTime.NOON, 2, BigDecimal.valueOf(20),
                Reservation.ReservationStatus.PENDING
        );
        Map<UUID, Reservation> toSave = Map.of(r1.getResId(), r1);

        repo.saveAll(toSave);
        Map<UUID, Reservation> loaded = repo.loadAll();

        assertEquals(1, loaded.size());
        Reservation r2 = loaded.get(r1.getResId());
        assertEquals(r1.getPartySize(), r2.getPartySize());
        assertEquals(r1.getCustomer().getEmail(), r2.getCustomer().getEmail());
    }
}
