package core;


import attacks.Attack;
import attacks.custom.OverlayCentroids;
import attacks.custom.SideBySide;
import attacks.custom.SideBySideOnTop;
import attacks.labelflipping.RandomLabelFlipping;
import picocli.CommandLine;
import properties.versionproviders.DisruptorVersionProvider;
import saver.Exporter;
import util.ArffUtil;
import util.CSVUtil;
import util.InstancesUtil;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.Callable;


@CommandLine.Command(
        name = "disruptor",
        description = "\nDisrupt the training set of a Machine Learning algorithm useing a set of different attacks.\n",
        versionProvider = DisruptorVersionProvider.class,
        // mixinStandardHelpOptions attribute adds --help and --version options
        mixinStandardHelpOptions = true
)
public class Disruptor implements Callable<Integer> {

    private Instances dataset;
    private String folderName;
    private ArrayList<Attack> attacksList = new ArrayList<>();

    // CLI PARAMS ---------------------------------------------------------------------------------------------------------------------------
    @CommandLine.Parameters(
            index = "0",
            description = "Filepath of the CSV file containing the dataset.\nUse --arff to pass a .arff file instead\n",
            paramLabel = "DATASET")
    private File datasetFile;

    // CLI OPTIONS ---------------------------------------------------------------------------------------------------------------------------
    @CommandLine.Option(
            names = {"-a", "--arff"},
            description = "Use this option if the dataset file format is .arff\n",
            paramLabel = "ARFF")
    private boolean isArff;

    @CommandLine.Option(
            names = {"-c", "--class"},
            description= "Specify the class attribute name.\nIf this param is not set, the program use “class” as the class attribute name\nDefault: class\n",
            paramLabel="CLASS",
            defaultValue="class")
    private String className;

    @CommandLine.Option(
            names = {"-C", "--capacities"},
            description= "Comma-separated capacities for the attacks.\nThe capacity is a percentage between 0 and 1.\ne.g. -C 0.2,0.5,1\nDefault: 1\n",
            paramLabel="CAPACITY",
            defaultValue="1",
            split = "," )
    private ArrayList<Double> capacitiesList = new ArrayList<>();

    @CommandLine.Option(
            names = {"-t", "--train-percent"},
            description= "Percentage of the training set.\nSet to 1 if they want to use all the dataset as training set \nThe percentage is a number between 0 and 1.\nDefault: 0.8\n",
            paramLabel="TRAIN_PERCENTAGE",
            defaultValue="0.8" )
    private double trainPercentage;

    @CommandLine.Option(
            names = {"-b", "--balance"},
            description = "[[ TO BE IMPLEMENTED ]]\nPerform other 2 run of the attacks on blanced dataset\nThe first additional run the instances are balanced with Resample.\nFor the second run is used  SMOTE\n",
            paramLabel = "BALANCE")
    private boolean toBalance;


    public static void main(String[] args) {
        int exitCode = new CommandLine(new Disruptor()).execute(args);
        System.exit(exitCode);
    }


    // EXECUTION ---------------------------------------------------------------------------------------------------------------------------
    @Override
    public Integer call() throws Exception {

        // Read the arff file
        if(isArff){
            dataset = ArffUtil.readArffFile(datasetFile, className);
        }
        else {
            dataset = CSVUtil.readCSVFile(datasetFile, className);
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

        // Attack main loop
        performAttacks(trainset, attacksList, capacitiesList);


        return 0;
    }


    /**
     * Fill the attacks list with all the attacks
     * @param dataset dataset to perturbate during the attacks
     */
    private void populateAttacksList(Instances dataset) {
        attacksList.add(new RandomLabelFlipping(dataset));
        attacksList.add(new SideBySide(dataset, 1));
        attacksList.add(new SideBySideOnTop(dataset, 1));
        attacksList.add(new OverlayCentroids(dataset));
    }

    /**
     * Perform all the attacks defined in the attacksList using all the capacities defined in the capacitiesList
     * @param trainingSet training set to perturb
     * @param attacksList list of attacks to perform
     * @param capacitiesList list of capacities
     */
    private void performAttacks(Instances trainingSet, ArrayList<Attack> attacksList, ArrayList<Double> capacitiesList){
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
    private void exportPerturbedDataset(String attackCode, Instances perturbedDataset) throws IOException {
        // Export ARFF
        Exporter arffExport = new Exporter( new ArffSaver() );
        arffExport.exportInFolder( perturbedDataset, folderName, attackCode );
        // Export CSV
        Exporter csvExport = new Exporter( new CSVSaver() );
        csvExport.exportInFolder( perturbedDataset, folderName, attackCode );
    }

    private void exportTestSet(Instances testSet) throws IOException {
        // Export ARFF
        Exporter arffExport = new Exporter( new ArffSaver() );
        arffExport.exportInFolder( testSet, folderName, testSet.relationName()+"_TEST" );
        // Export CSV
        Exporter csvExport = new Exporter( new CSVSaver() );
        csvExport.exportInFolder( testSet, folderName, testSet.relationName()+"_TEST" );
    }
}
