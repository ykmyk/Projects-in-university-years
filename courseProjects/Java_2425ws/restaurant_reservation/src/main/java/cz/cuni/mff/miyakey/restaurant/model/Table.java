package cz.cuni.mff.miyakey.restaurant.model;

/**
 * A single table in the restaurant
 * identified by the table number and its capacity.
 * Immutable.
 *
 * @author Yuka Miyake
 * @version 1.0
 */

public class Table {
    /** unique identifier for the table */
    private final int tableNumber;
    /** maximum seats at the table, must be more than 0 */
    private final int capacity;

    /**
     * Constructs a Table with the given number and capacity.
     *
     * @param tableNumber  unique identifier for the table
     * @param capacity     maximum seats at the table, must be more than 0
     * @throws IllegalArgumentException if capacity is less than 0
     */

    public Table(int tableNumber, int capacity) {
        if (capacity < 1){
            throw new IllegalArgumentException("Capacity must be greater than 0");
        }
        this.tableNumber = tableNumber;
        this.capacity = capacity;
    }

    /**
     * @return the unique table number
     */
    public int getTableNumber() {
        return tableNumber;
    }

    /**
     * @return maximum seating capacity of the table
     */
    public int getCapacity() {
        return capacity;
    }

    @Override
    public String toString() {
        return "Table {" +
                "tableNumber=" + tableNumber +
                "capacity=" + capacity +
                "}";
    }

    @Override
    public boolean equals(Object o) {
        if(this == o) return true;
        if(!(o instanceof Table table)) return false;
        return this.tableNumber == table.tableNumber
                && this.capacity == table.capacity;
    }

    @Override
    public int hashCode() {
        int result = Integer.hashCode(tableNumber);
        result = 31 * result + Integer.hashCode(capacity);
        return result;
    }
}
