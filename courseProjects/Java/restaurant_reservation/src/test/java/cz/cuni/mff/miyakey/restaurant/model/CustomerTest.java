package cz.cuni.mff.miyakey.restaurant.model;

import org.junit.jupiter.api.Test;
import java.util.UUID;
import static org.junit.jupiter.api.Assertions.*;

public class CustomerTest {

    @Test
    void constructor_setFields(){
        UUID id = UUID.randomUUID();
        Customer c = new Customer(id, "Adam", "a@example.com", "000-0000");
        assertEquals(id, c.getCusId());
        assertEquals("Adam", c.getName());
        assertEquals("a@example.com", c.getEmail());
        assertEquals("000-0000", c.getPhone());
    }

    @Test
    void constructor_setNullId(){
        UUID id = UUID.randomUUID();
        assertThrows(IllegalArgumentException.class,
                () -> new Customer(null, "name", "e@example", "123"));
        assertThrows(IllegalArgumentException.class,
                () -> new Customer(id, null, "e@example", "123"));
        assertThrows(IllegalArgumentException.class,
                () -> new Customer(id, "name", null, "123"));
    }

    @Test
    void equalsAndHashCode(){
        UUID id = UUID.randomUUID();
        Customer c1 = new Customer(id, "name", "e@example", "123");
        Customer c2 = new Customer(id, "name", "e@example", "123");
        Customer c3 = new Customer(id, "name", "a@example.com", "000-0000");
        assertEquals(c1, c2);
        assertEquals(c1.hashCode(), c2.hashCode());
        assertNotEquals(c1, c3);
    }

    @Test
    void toString_(){
        UUID id = UUID.randomUUID();
        Customer c1 = new Customer(id,
                "name", "e@example", "123");
        String s = c1.toString();
        assertTrue(s.contains(id.toString()));
        assertTrue(s.contains("name"));
        assertTrue(s.contains("e@example"));
    }
}
