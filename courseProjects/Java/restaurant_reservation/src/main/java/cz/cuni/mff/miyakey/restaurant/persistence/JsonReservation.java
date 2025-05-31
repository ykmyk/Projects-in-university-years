package cz.cuni.mff.miyakey.restaurant.persistence;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import cz.cuni.mff.miyakey.restaurant.model.Reservation;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;


import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 *
 * {@code JsonReservation} is an implementation of the interface {@link Repository}
 * that persists the reservation data to a JSON file on disk.
 * It uses Jackson to serialize and deserialize a Map of
 * {@link UUID} -> {@link Reservation}
 *
 * On {@link #loadAll()} function, if the target file does not exist or it is empty.
 * then this returns an empty Map.
 * otherwise, this reads and converts its entire contents.
 *
 * On {@link #saveAll(Map)} function, the Map is written in pretty-print JSON format,
 * and this will overwrite any existing file.
 * Any I/O errors are wrapped in an unchecked {@link RuntimeException}
 *
 * @author Yuka Miyake
 * @version 1.0
 *
 * implementation of Repository that serializes reservations as JSON
 */

public class JsonReservation implements Repository {
    /** Underlying file used for JSON persistence. */
    private final File file;

    /** Jackson object mapper, configured with Java Time support. */
    private final ObjectMapper mapper;

    /**
     * Create a new {@code JsonReservation} for a given file.
     * If the file does not exist, it will be created on first save.
     *
     * @param file the JSON file to read/write reservations
     */
    public JsonReservation(File file) {
        this.file = file;
        this.mapper = new ObjectMapper()
                .registerModule(new JavaTimeModule());
    }


    /**
     *　Load all reservations from disk.
     *  If the file does not exist or the file length is zero, return empty map
     *  otherwise, returns the entire file and parses it into a {@code Map<UUID, Reservation>}.
     *
     * @return a Map of UUID -> {@link Reservation} objects
     * @throws RuntimeException if there is any I/O error or parse error
     */
    @Override
    public Map<UUID, Reservation> loadAll(){
        if(!file.exists() || file.length() == 0) return new HashMap<>();
        try{
            return mapper.readValue(file, new TypeReference<Map<UUID, Reservation>>(){});
        }catch(IOException e){
            throw new RuntimeException("Failed to load reservations", e);
        }
    }
    /**
     * Write out all reservation to disk as pretty-printed JSON format to save.
     * IF the file already exists, it will be overwritten.
     *
     * @param reservations the ful Map of UUID -> {@link Reservation} to persist
     * @throws RuntimeException if any I?O error occurs while writing.
     */
    @Override
    public void saveAll(Map<UUID, Reservation> reservations){
        try{
            mapper.writerWithDefaultPrettyPrinter().writeValue(file, reservations);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }
}
