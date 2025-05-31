package cz.cuni.mff.miyakey.restaurant.main;

import cz.cuni.mff.miyakey.restaurant.model.Customer;
import cz.cuni.mff.miyakey.restaurant.model.Reservation;
import cz.cuni.mff.miyakey.restaurant.model.ReservationManager;
import cz.cuni.mff.miyakey.restaurant.model.Table;
import cz.cuni.mff.miyakey.restaurant.persistence.JsonReservation;

import java.io.File;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.*;

/**
 * Command line entry point for the Restaurant Reservation system.
 *
 * This Main class support interactive commands to create, list, cancel, and confirm
 * the reservations using simple text form.
 *
 * For the webapp entry is in 'webapp' package
 *
 * @author Yuka Miyake
 * @version 1.0
 *
 */

public class Main {
    /** Relative path to the JSON data file. */
    private static final String Data_file = "data/reservations.json";

    /** map of customers by email */
    private static final Map<String, Customer> customers = new HashMap<>();

    /**
     * Bootstraps the reservation manager
     * then loads existing data
     * Enters a REPL loop responding to user commands as well.
     *
     * @param args ignored
     */

    public static void main (String[] args) {
        List<Table> tables = Arrays.asList(
                new Table(1, 2),
                new Table(2, 4),
                new Table(3, 4),
                new Table(4, 6),
                new Table(5, 8)
        );

        // load reservation from Json
        JsonReservation repo = new JsonReservation(new File(Data_file));
        Map<UUID, Reservation> loaded = repo.loadAll();

        // create manager with given data
        ReservationManager manager = new ReservationManager(tables);

        loaded.values().forEach(r -> {
            Customer c = r.getCustomer();
            customers.putIfAbsent(c.getEmail(), c);
        });

        Scanner in = new Scanner(System.in, "UTF-8");
        System.out.println("Welcome! Enter commands: (to see possible command, type 'help'");

        boolean running = true;
        while(running){
            System.out.print("> ");
            String line = in.nextLine().trim();
            if(line.isEmpty()) continue;

            String[] tokens = line.split("\\s+");
            String command = tokens[0].toLowerCase(Locale.ROOT);

            try{
                switch(command){
                    case "help":
                        printHelp();
                        break;

                    case "create":
                        // create <name> <email> <phone> <YYYY-MM-DD> <HH:MM> <partySize>
                        String name  = tokens[1];
                        String email = tokens[2];
                        String phone = tokens[3];
                        Customer customer = customers.get(email);
                        if (customer == null) {
                            customer = new Customer(UUID.randomUUID(), name, email, phone);
                            customers.put(email, customer);
                        }
                        LocalDate date = LocalDate.parse(tokens[4]);
                        LocalTime time = LocalTime.parse(tokens[5]);
                        int size = Integer.parseInt(tokens[6]);
                        BigDecimal dep = BigDecimal.valueOf(size).multiply(BigDecimal.valueOf(5));

                        var res = manager.createReservation(customer, date, time, size, dep);
                        repo.saveAll(manager.getAllReservations());
                        System.out.println("Created " + res.getResId());
                        break;

                    case "list":
                        // list <YYYY-MM-DD>
                        LocalDate d = LocalDate.parse(tokens[1]);
                        manager.listReservations(d).forEach(System.out::println);
                        break;

                    case "cancel":
                        // cancel <UUID>
                        UUID idToCancel = UUID.fromString(tokens[1]);
                        boolean cancelled = manager.cancelReservation(idToCancel);
                        if (cancelled) {
                            repo.saveAll(manager.getAllReservations());
                            System.out.println("Cancelled");
                        }else{
                            System.out.println("Couldn't cancel reservation");
                        }
                        break;

                case "exit":
                    System.out.println("System closed.");
                    in.close();
                    return;

                default:
                    System.out.println("Unknown command. Type 'help' for the possible commands" + command);
                }
            } catch (Exception e){
                System.out.println("Error:" + e.getMessage());
            }
        }
    }

    /**
     * This prints out the possible commands with the necessary arguments
     * and will be executed when it receives 'help' command
     */

    private static void printHelp() {
        System.out.println("""
        Available commands:
          help
          create <name> <email> <phone> <YYYY-MM-DD> <HH:MM> <number of People> <deposit>
          list <YYYY-MM-DD>
          confirm <reservation-UUID>
          cancel <reservation-UUID>
          exit
        """);
    }
}
