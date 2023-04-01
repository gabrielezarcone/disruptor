package core;

import picocli.CommandLine;
import util.ArffUtil;
import weka.core.Instances;

import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.concurrent.Callable;

@CommandLine.Command(
        name = "merge-experimenter-arff",
        description = "\nMerge two Experimenter .arff results\n",
        version = "1.0.1",
        // mixinStandardHelpOptions attribute adds --help and --version options
        mixinStandardHelpOptions = true
)
public class MergeExperimenterArff implements Callable<Integer> {

    // COMMAND LINE PARAMETERS AND OPTIONS -----------------------------------------
    @CommandLine.Parameters(
            index = "0",
            description = "Filepath of the first file to merge",
            paramLabel = "FILE1")
    private File file1;
    @CommandLine.Parameters(
            index = "1",
            description = "Filepath of the second file to merge",
            paramLabel = "FILE2")
    private File file2;

    @CommandLine.Option(
            names = {"-o", "--output-path"},
            description= "Specify a different folder for the output",
            paramLabel="OUTPUT_FOLDER_PATH",
            defaultValue="./")
    private String outputFolderPath;


    // EXECUTION -----------------------------------------
    @Override
    public Integer call() throws Exception {
        try {
            Instances mergedInstances = ArffUtil.mergeArffFiles(file1, file2);
            ArffUtil.exportToArff(mergedInstances, outputFolderPath, "merged-"+new Date());
            System.out.println("---------------- COMPLETED ----------------");
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("IO exception");
        } catch (Exception e) {
            e.printStackTrace();
        }
        return 0;
    }

    public static void main(String... args) {
        int exitCode = new CommandLine(new MergeExperimenterArff()).execute(args);
        System.exit(exitCode);
    }
}
