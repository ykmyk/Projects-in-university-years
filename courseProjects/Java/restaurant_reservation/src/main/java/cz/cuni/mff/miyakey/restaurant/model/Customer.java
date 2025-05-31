package cz.cuni.mff.miyakey.restaurant.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Objects;
import java.util.UUID;

/**
 * Represents a customer making a reservation
 *
 * @author Yuka Miyake
 * @version 1.0
 */

public class Customer {
    private final UUID cusId;
    private final String name;
    private final String email;
    private final String phone;


    /**
     * Constructs a new Customer.
     *
     * @param cusId unique identifier
     * @param name  customer's full name, must be non-blank
     * @param email customer's email, must be non-blank
     * @param phone customer's phone number, must be non-blank
     * @throws IllegalArgumentException if any string is null or blank
     */

    @JsonCreator
    public Customer(@JsonProperty("cusId") UUID cusId,
                    @JsonProperty("name") String name,
                    @JsonProperty("email") String email,
                    @JsonProperty("phone") String phone) {
        if(cusId == null || name == null || email == null || phone == null ||
                name.isBlank() || email.isBlank() || phone.isBlank()) {
            throw new IllegalArgumentException("Customer fields must be non-empty");
        }
        this.cusId = cusId;
        this.name = name;
        this.email = email;
        this.phone = phone;
    }
    /** @return the unique customer ID */
    public UUID getCusId() {
        return cusId;
    }

    /** @return the full name */
    public String getName() {
        return name;
    }

    /** @return the email address */
    public String getEmail() {
        return email;
    }

    /** @return the phone number */
    public String getPhone() {
        return phone;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Customer)) return false;
        Customer other = (Customer) o;
        return cusId.equals(other.cusId)
                && name.equals(other.name)
                && email.equals(other.email)
                && phone.equals(other.phone);
    }

    @Override
    public int hashCode() {
        return Objects.hash(cusId, name, email, phone);
    }

    @Override
    public String toString() {
        return "Customer{" +
                "customer id=" + cusId +
                ", name=" + name + '\'' +
                ", email=" + email + '\'' +
                ", phone=" + phone + '\'' +
                '}';
                    }
}
