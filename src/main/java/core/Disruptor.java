package core;


import attacks.Attack;
import attacks.custom.OverlayCentroids;
import attacks.custom.SideBySide;
import attacks.custom.SideBySideDuplicate;
import attacks.labelflipping.LabelFlipping;
import attacks.labelflipping.RandomLabelFlipping;
import experiment.DisruptorExperiment;
import lombok.extern.slf4j.Slf4j;
import picocli.CommandLine;
import properties.versionproviders.DisruptorVersionProvider;
import roc.ROCGenerator;
import saver.Exporter;
import util.ArffUtil;
import util.CSVUtil;
import util.ExceptionUtil;
import util.InstancesUtil;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.Callable;

@Slf4j
@CommandLine.Command(
        name = "disruptor",
        description = "\nDisrupt the training set of a Machine Learning algorithm useing a set of different attacks.\n",
        versionProvider = DisruptorVersionProvider.class,
        // mixinStandardHelpOptions attribute adds --help and --version options
        mixinStandardHelpOptions = true
)
public class Disruptor implements Callable<Integer> {

    public static final String PARENT_FOLDER = "output";
    public static final String EXPERIMENT_FOLDER = "experiment";

    private String folderName = PARENT_FOLDER;
    private String experimentFolderName = EXPERIMENT_FOLDER;
    private ArrayList<Attack> attacksList = new ArrayList<>();
    private ArrayList<Classifier> classifiersList = new ArrayList<>();
    private ArrayList<Instances> perturbedDatasets = new ArrayList<>();
    private Instances testSet;
    private final Exporter arffExport = new Exporter( new ArffSaver() );
    private final Exporter csvExport = new Exporter( new CSVSaver() );

    // CLI PARAMS ---------------------------------------------------------------------------------------------------------------------------
    @CommandLine.Parameters(
            index = "0",
            description = "Filepath of the CSV file containing the dataset.\nIt is MANDATORY that the first row of the CSV file should contains the features names.\nUse --arff to pass a .arff file instead\n",
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

    @CommandLine.Option(
            names = {"-e", "--experimenter"},
            description = "Evaluate the effectiveness of the attacks using several ML algorithms\n",
            paramLabel = "EXP")
    private boolean experimenter;

    @CommandLine.Option(
            names = {"-r", "--roc"},
            description = "Show the ROC curves for each attack\n",
            paramLabel = "ROC")
    private static boolean roc;

    @CommandLine.Option(
            names = {"-R", "--runs"},
            description = "Define the number of runs for each attack\nDefault: 10\n",
            paramLabel = "NUMBER_OF_RUNS",
            defaultValue="10")
    private static int runs;


    public static void main(String[] args) {
        int exitCode = new CommandLine(new Disruptor()).execute(args);
        if(!roc){
            System.exit(exitCode);
        }
    }


    // EXECUTION ---------------------------------------------------------------------------------------------------------------------------
    @Override
    public Integer call() throws Exception {

        // Read the dataset file
        Instances dataset;
        if(isArff){
            dataset = ArffUtil.readArffFile(datasetFile, className);
        }
        else {
            dataset = CSVUtil.readCSVFile(datasetFile, className);
        }

        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HHmmss");
        String startDate = simpleDateFormat.format(new Date());

        for( int run=0; run<runs; run++ ){

            log.info("\n\nRUN {} ----------------------------------\n", run);

            // Creates a copy of the starting instances otherwise a test set is added at every run growing exponentially
            Instances runDataset = new Instances(dataset);

            if(experimenter){
                // To use as a reference, add the input dataset as the first list element
                perturbedDatasets.add(runDataset);
            }

            // Set folder name
            folderName = PARENT_FOLDER + File.separator + startDate + File.separator + "run" + run;

            // Split Train and Test set
            Instances[] splitTrainTest = InstancesUtil.splitTrainTest(runDataset, trainPercentage, true, run);
            Instances trainset = splitTrainTest[0];
            testSet = splitTrainTest[1];

            // Export test set
            exportTestSet(testSet);

            // Populate the attacks and the classifiers lists
            populateAttacksList(trainset);
            populateClassifiersList();

            // Attack main loop
            performAttacks(trainset, attacksList, capacitiesList);

            if(experimenter){
                // Append the test set to each dataset
                appendTestSet(true);
            }
        }

        if(experimenter){
            // Evaluate the effectiveness of the attacks
            evaluateAttacks();
        }

        return 0;
    }


    /**
     * Fill the attacks list with all the attacks
     * @param dataset dataset to perturbate during the attacks
     */
    private void populateAttacksList(Instances dataset) {
        attacksList.clear();
        attacksList.add(new LabelFlipping(dataset));
        attacksList.add(new RandomLabelFlipping(dataset));
        attacksList.add(new SideBySide(dataset, 1));
        attacksList.add(new SideBySideDuplicate(dataset));
        attacksList.add(new OverlayCentroids(dataset));
    }
    /**
     * Fill the classifiers list with a subset of classifiers
     */
    private void populateClassifiersList() {
        classifiersList.clear();
        classifiersList.add( new RandomForest() );
        classifiersList.add( new NaiveBayes() );
        classifiersList.add( new J48() );
        classifiersList.add( new BayesNet() );
    }

    /**
     * Perform all the attacks defined in the attacksList using all the capacities defined in the capacitiesList
     * @param trainingSet training set to perturb
     * @param attacksList list of attacks to perform
     * @param capacitiesList list of capacities
     */
    private void performAttacks(Instances trainingSet, ArrayList<Attack> attacksList, ArrayList<Double> capacitiesList){
        // Nested loop between attacks list and capacities list
        attacksList.forEach( attack -> {
            String attackClassName = attack.getClass().getSimpleName();
            log.info("Started attack {}", attackClassName);
            String attackName = trainingSet.relationName() + "_" + attackClassName;
            // save in a list all the perturbed datasets of this attack
            ArrayList<Instances> attackPerturbedDatasets = new ArrayList<>();
            capacitiesList.forEach(capacity -> {

                log.info("\tcapacity: {}", capacity);

                // Define an attack code unique for this attack run
                String attackCode = attackName + "_" + capacity;

                // Perform this attack with this capacity
                Instances trainingSetCopy = new Instances(trainingSet);
                attack.setTarget( trainingSetCopy );
                attack.setCapacity( capacity );
                Instances perturbedInstances = attack.attack();
                perturbedInstances.setRelationName(attackCode);

                if(experimenter){
                    perturbedDatasets.add(perturbedInstances);
                }
                if(roc){
                    attackPerturbedDatasets.add(perturbedInstances);
                }

                // Export the perturbed instances
                try {
                    exportPerturbedDataset(attackCode, perturbedInstances);
                } catch (Exception e) {
                    log.error("Problem during the export of the perturbed dataset");
                    log.debug(attackCode);
                    ExceptionUtil.logException(e, log);
                }

            });
            if(roc){
                log.info("Started ROC curves visualization for attack {}", attackClassName);
                log.info("Running...");
                for(Classifier classifier : classifiersList){
                    log.debug("Started ROC for attack {} and classifier {}", attackClassName, classifier.getClass().getSimpleName());
                    ROCGenerator rocGenerator = new ROCGenerator(testSet, classifier, attackName);
                    rocGenerator.visualizeROCCurves(attackPerturbedDatasets);
                    log.debug("Finished ROC for attack {} and classifier {}", attackClassName, classifier.getClass().getSimpleName());
                }
                log.info("Finished ROC curves visualization for attack {}", attackClassName);
            }
        });
    }

    /**
     * Export the perturbed dataset in the same folder of the others attack
     * @param attackCode String used to identify this particular attack execution used as file name
     * @param perturbedDataset The perturbed dataset after the attack
     * @throws IOException if problems during the export
     */
    private void exportPerturbedDataset(String attackCode, Instances perturbedDataset) throws IOException {
        // Export ARFF
        arffExport.exportInFolder( perturbedDataset, folderName, attackCode );
        // Export CSV
        csvExport.exportInFolder( perturbedDataset, folderName, attackCode );
    }

    private void exportTestSet(Instances testSet) throws IOException {
        // Export ARFF
        arffExport.exportInFolder( testSet, folderName, testSet.relationName()+"_TEST" );
        // Export CSV
        csvExport.exportInFolder( testSet, folderName, testSet.relationName()+"_TEST" );
    }

    private void exportTrainTestSet(Instances trainTestSet) throws IOException {
        // Export ARFF
        arffExport.exportInFolder( trainTestSet, folderName + File.separator + "trainTest", trainTestSet.relationName());
        // Export CSV
        csvExport.exportInFolder( trainTestSet, folderName + File.separator + "trainTest", trainTestSet.relationName() );
    }


    /**
     * Append the test set to every dataset present in perturbedDatasets
     * @param export true if the train+test file should be exported
     */
    private void appendTestSet(boolean export){
        perturbedDatasets.forEach( dataset -> {
            try {
                InstancesUtil.addAllInstances(dataset, testSet);

                if(export){
                    exportTrainTestSet(dataset);
                }

            } catch (Exception e) {
                log.error("Problem appending the test set to the train set");
                ExceptionUtil.logException(e, log);
            }
        });
    }


    /**
     * Evaluate the effectiveness of the attacks using several ML algorithms
     */
    private void evaluateAttacks() throws Exception {
        DisruptorExperiment experiment = new DisruptorExperiment(perturbedDatasets, trainPercentage, folderName);
        experiment.setClassifiersList(classifiersList);
        experiment.start();
    }
}
