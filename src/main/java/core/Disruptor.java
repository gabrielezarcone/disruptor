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
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Range;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.experiment.*;

import javax.swing.*;
import java.beans.IntrospectionException;
import java.beans.PropertyDescriptor;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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

    private String folderName = "output";
    private String experimentFolderName = "experiment";
    private ArrayList<Attack> attacksList = new ArrayList<>();
    private ArrayList<Instances> perturbedDatasets = new ArrayList<>();
    private Instances testSet;

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


    public static void main(String[] args) {
        int exitCode = new CommandLine(new Disruptor()).execute(args);
        System.exit(exitCode);
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

        if(experimenter){
            // To use as a reference, add the input dataset as the first list element
            perturbedDatasets.add(dataset);
        }

        // Set folder name
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HHmmss");
        folderName = folderName + File.separator + simpleDateFormat.format(new Date());
        experimentFolderName = folderName + File.separator + experimentFolderName;

        // Split Train and Test set
        Instances[] splitTrainTest = InstancesUtil.splitTrainTest(dataset, trainPercentage, true);
        Instances trainset = splitTrainTest[0];
        testSet = splitTrainTest[1];

        // Export test set
        exportTestSet(testSet);

        // Populate the attacks list
        populateAttacksList(trainset);

        // Attack main loop
        performAttacks(trainset, attacksList, capacitiesList);

        if(experimenter){
            // Append the test set to each dataset
            appendTestSet();
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

            if(experimenter){
                perturbedDatasets.add(perturbedInstances);
            }

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


    /**
     * Append the test set to every dataset present in perturbedDatasets
     */
    private void appendTestSet(){
        perturbedDatasets.forEach( dataset -> {
            try {
                InstancesUtil.addAllInstances(dataset, testSet);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }


    /**
     * Evaluate the effectiveness of the attacks using several ML algorithms
     */
    private void evaluateAttacks() throws Exception {
        // Use the Experimenter Weka API for the evaluation

        // 1. setup the experiment ------------------------------------------------------------------------------------------
        Experiment experiment = new Experiment();
        experiment.setPropertyArray(new Classifier[0]);
        experiment.setUsePropertyIterator(true);

        SplitEvaluator splitEvaluator = null;
        Classifier classifier = null;
        Instances dataset = perturbedDatasets.get(0);
        boolean classification = false;
        if ( dataset.classAttribute().isNominal() ) {
            classification = true;
            splitEvaluator  = new ClassifierSplitEvaluator();
            classifier = ((ClassifierSplitEvaluator) splitEvaluator).getClassifier();
        }
        else if ( dataset.classAttribute().isNumeric() ) {
            splitEvaluator  = new RegressionSplitEvaluator();
            classifier = ((RegressionSplitEvaluator) splitEvaluator).getClassifier();
        }

        RandomSplitResultProducer rsrp = new RandomSplitResultProducer();
        rsrp.setRandomizeData(false);
        rsrp.setTrainPercent(trainPercentage*100);
        rsrp.setSplitEvaluator(splitEvaluator);

        PropertyNode[] propertyPath = new PropertyNode[2];
        try {
            propertyPath[0] = new PropertyNode(
                    splitEvaluator,
                    new PropertyDescriptor("splitEvaluator", RandomSplitResultProducer.class),
                    RandomSplitResultProducer.class
            );
            propertyPath[1] = new PropertyNode(
                    classifier,
                    new PropertyDescriptor("classifier", splitEvaluator.getClass()),
                    splitEvaluator.getClass()
            );
        }
        catch (IntrospectionException e) {
            e.printStackTrace();
        }

        experiment.setResultProducer(rsrp);
        experiment.setPropertyPath(propertyPath);

        // Set the runs number:
        experiment.setRunLower(1);
        experiment.setRunUpper(10);

        // Set classifiers
        ArrayList<Classifier> classifiersList = new ArrayList<>();
        classifiersList.add( new RandomForest() );
        classifiersList.add( new NaiveBayes() );
        classifiersList.add( new J48() );
        classifiersList.add( new BayesNet() );
        experiment.setPropertyArray( classifiersList.toArray() );

        // Multiple datasets
        DefaultListModel<File> model = new DefaultListModel<>();
        perturbedDatasets.forEach( perturbedDataset -> {
            Exporter arffExport = new Exporter( new ArffSaver() );
            try {
                arffExport.exportInFolder( perturbedDataset, experimentFolderName, perturbedDataset.relationName()+"_EXP" );
                File datasetFile = arffExport.getExportedFile();
                model.addElement(datasetFile);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } );
        experiment.setDatasets(model);

        // Save result in an arff file
        InstancesResultListener irl = new InstancesResultListener();
        irl.setOutputFile(new File(experimentFolderName+File.separator+"experimenterOutput.arff"));
        experiment.setResultListener(irl);


        // 2. run experiment -------------------------------------------------------------------------------------------
        System.out.println("Initializing...");
        experiment.initialize();
        System.out.println("Running...");
        experiment.runExperiment();
        System.out.println("Finishing...");
        experiment.postProcess();


        // 3. calculate statistics and output them -------------------------------------------------------------------------------------------
        System.out.println("Evaluating...");
        PairedTTester tester = new PairedCorrectedTTester();
        Instances result = new Instances(new BufferedReader(new FileReader(irl.getOutputFile())));
        tester.setInstances(result);
        tester.setSortColumn(-1);
        tester.setRunColumn(result.attribute("Key_Run").index());
        //TODO capire se questo if serve
        /*if (classification)
            tester.setFoldColumn(result.attribute("Key_Fold").index());*/
        tester.setResultsetKeyColumns(
                new Range(
                        ""
                                + (result.attribute("Key_Dataset").index() + 1)));
        tester.setDatasetKeyColumns(
                new Range(
                        ""
                                + (result.attribute("Key_Scheme").index() + 1)
                                + ","
                                + (result.attribute("Key_Scheme_options").index() + 1)
                                + ","
                                + (result.attribute("Key_Scheme_version_ID").index() + 1)));
        tester.setResultMatrix(new ResultMatrixPlainText());
        tester.setDisplayedResultsets(null);
        tester.setSignificanceLevel(0.05);
        tester.setShowStdDevs(true);
        // fill result matrix (but discarding the output)
        if (classification)
            tester.multiResultsetFull(0, result.attribute("Percent_correct").index());
        else
            tester.multiResultsetFull(0, result.attribute("Correlation_coefficient").index());
        // output results for reach dataset
        System.out.println("\nResult:");
        ResultMatrix matrix = tester.getResultMatrix();
        System.out.println(matrix);
        for (int i = 0; i < matrix.getColCount(); i++) {
            System.out.println(matrix.getColName(i));
            System.out.println("    Perc. correct (mean): " + matrix.getMean(i, 0));
            System.out.println("    StdDev: " + matrix.getStdDev(i, 0));
        }
    }
}
