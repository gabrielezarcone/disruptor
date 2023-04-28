package properties.versionproviders;

import picocli.CommandLine;
import properties.PomProperties;

public class DisruptorVersionProvider implements CommandLine.IVersionProvider {
    @Override
    public String[] getVersion() throws Exception {
        PomProperties pomProperties = new PomProperties();
        return new String[] { pomProperties.getProperty(PomProperties.Properties.DISRUPTOR_VERSION) };
    }
}
