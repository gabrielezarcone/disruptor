package properties;

import java.io.IOException;

/**
 * Returns the {@link java.util.Properties} object containing all the properties of the Pom
 * <p/>
 * The properties are fetched from the pom.properties file that is an AUTOGENERATED FILE based on the pom.xml file
 * Thus, to change the properties returned by this class, change directly the pom.xml.
 */
public class PomProperties extends PropertiesReader{

    public static final String PROPERTIES_PATH = "/pom.properties";

    public static class Properties{
        private Properties(){}
        public static final String MERGE_EXPERIMENTER_ARFF_VERSION = "merge-experimenter-arff.version";
        public static final String DISRUPTOR_VERSION = "disruptor.version";
    }

    public PomProperties() throws IOException {
        super(PomProperties.PROPERTIES_PATH);
    }
}
