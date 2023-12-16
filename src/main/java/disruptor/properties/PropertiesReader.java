package disruptor.properties;

import java.io.*;
import java.util.Properties;

/**
 * Util class able to read .properties files.
 * Extend this class and create a new constructor with no arguments that call the super("property filename")
 */
public abstract class PropertiesReader {

    private final Properties properties;

    protected PropertiesReader(String propertyFilePath) throws IOException {
        InputStream is = this.getClass().getResourceAsStream(propertyFilePath);
        InputStreamReader inputStreamReader = new InputStreamReader(is);
        properties = new Properties();
        properties.load(inputStreamReader);
        inputStreamReader.close();
    }

    public String getProperty(String propertyName){
        return properties.getProperty(propertyName);
    }
}
