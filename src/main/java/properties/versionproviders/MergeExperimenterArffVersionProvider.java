package properties.versionproviders;

import properties.PomProperties;
import picocli.CommandLine;
import properties.PropertiesReader;

public class MergeExperimenterArffVersionProvider implements CommandLine.IVersionProvider {
    @Override
    public String[] getVersion() throws Exception {
        PomProperties pomProperties = new PomProperties();
        return new String[] { pomProperties.getProperty(PomProperties.Properties.MERGE_EXPERIMENTER_ARFF_VERSION) };
    }
}
