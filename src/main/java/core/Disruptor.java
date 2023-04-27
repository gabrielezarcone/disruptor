package core;


import attacks.Attack;
import attacks.custom.OverlayCentroids;
import attacks.custom.SideBySide;
import attacks.custom.SideBySideOnTop;
import attacks.labelflipping.RandomLabelFlipping;
import costants.FilePaths;
import picocli.CommandLine;
import properties.versionproviders.DisruptorVersionProvider;
import properties.versionproviders.MergeExperimenterArffVersionProvider;
import saver.Exporter;
import util.ArffUtil;
import util.InstancesUtil;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;


@CommandLine.Command(
        name = "disruptor",
        description = "\nDisrupt the training set of a Machine Learning algorithm useing a set of different attacks.\n",
        versionProvider = DisruptorVersionProvider.class,
        // mixinStandardHelpOptions attribute adds --help and --version options
        mixinStandardHelpOptions = true
)
public class Disruptor {

    private  static String className = "class";

    private static Instances dataset;
    private static String folderName;
    private static ArrayList<Attack> attacksList = new ArrayList<>();
    private static ArrayList<Double> capacitiesList = new ArrayList<>();
    private static double trainPercentage = 0.8;

    // CLI PARAMS:
    @CommandLine.Parameters(
            index = "0",
            description = "Filepath of the CSV file containing the dataset. Use --arff to pass a .arff file instead",
            paramLabel = "DATASET")
    private static File datasetFile;

    // CLI OPTIONS:
    @CommandLine.Option(
            names = {"-a", "--arff"},
            description = "Use this option if the dataset file format is .arff",
            paramLabel = "ARFF")
    private static boolean isArff;


    public static void main(String[] args) throws Exception {
        // Read the arff file
        if(isArff){
            dataset = ArffUtil.readArffFile(datasetFile, className);
        }
        else {
            dataset = csvToInstances();
        }

        // Set folder name
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HHmmss");
        folderName = simpleDateFormat.format(new Date());

        // Split Train and Test set
        Instances[] splitTrainTest = InstancesUtil.splitTrainTest(dataset, trainPercentage, true);
        Instances trainset = splitTrainTest[0];
        Instances testSet = splitTrainTest[1];

        // Export test set
        exportTestSet(testSet);

        // Populate the attacks list
        populateAttacksList(trainset);

        // Populate the capacities list
        populateCapacitiesList();

        // Attack main loop
        performAttacks(trainset, attacksList, capacitiesList);
    }


    private static Instances csvToInstances() {
        // Stub method waiting for the implementation
        return new Instances(dataset);
    }


    /**
     * Fill the attacks list with all the attacks
     * @param dataset dataset to perturbate during the attacks
     */
    private static void populateAttacksList(Instances dataset) {
        attacksList.add(new RandomLabelFlipping(dataset));
        attacksList.add(new SideBySide(dataset, 1));
        attacksList.add(new SideBySideOnTop(dataset, 1));
        attacksList.add(new OverlayCentroids(dataset));
    }

    /**
     * Fill the capacities list
     */
    private static void populateCapacitiesList() {
        capacitiesList = new ArrayList<>();
        capacitiesList.add(0.5);
        capacitiesList.add(1.0);
    }

    /**
     * Perform all the attacks defined in the attacksList using all the capacities defined in the capacitiesList
     * @param trainingSet training set to perturb
     * @param attacksList list of attacks to perform
     * @param capacitiesList list of capacities
     */
    private static void performAttacks(Instances trainingSet, ArrayList<Attack> attacksList, ArrayList<Double> capacitiesList){
        // Nested loop between attacks list and capacities list
        attacksList.forEach( attack -> capacitiesList.forEach(capacity -> {

            // Define an attack code unique for this attack run
            String attackCode = trainingSet.relationName() + "_" + attack.getClass().getSimpleName() + "_" + capacity;

            // Perform this attack with this capacity
            Instances trainingSetCopy = new Instances(trainingSet);
            attack.setTarget( trainingSetCopy );
            attack.setCapacity( capacity );
            Instances perturbedInstances = attack.attack();
            perturbedInstances.setRelationName(attackCode);

            // Export the perturbed instances
            try {
                exportPerturbedDataset(attackCode, perturbedInstances);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }));
    }

    /**
     * Export the perturbed dataset in the same folder of the others attack
     * @param attackCode String used to identify this particular attack execution used as file name
     * @param perturbedDataset The perturbed dataset after the attack
     * @throws IOException if problems during the export
     */
    private static void exportPerturbedDataset(String attackCode, Instances perturbedDataset) throws IOException {
        // Export ARFF
        Exporter arffExport = new Exporter( new ArffSaver() );
        arffExport.exportInFolder( perturbedDataset, folderName, attackCode );
        // Export CSV
        Exporter csvExport = new Exporter( new CSVSaver() );
        csvExport.exportInFolder( perturbedDataset, folderName, attackCode );
    }

    private static void exportTestSet(Instances testSet) throws IOException {
        // Export ARFF
        Exporter arffExport = new Exporter( new ArffSaver() );
        arffExport.exportInFolder( testSet, folderName, testSet.relationName()+"_TEST" );
        // Export CSV
        Exporter csvExport = new Exporter( new CSVSaver() );
        csvExport.exportInFolder( testSet, folderName, testSet.relationName()+"_TEST" );
    }
}
