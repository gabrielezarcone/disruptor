package core;


import attacks.Attack;
import attacks.horizontal.labelflipping.LabelFlipping;
import attacks.horizontal.labelflipping.RandomLabelFlipping;
import attacks.vertical.*;
import attributeselection.RandomSelector;
import experiment.DisruptorExperiment;
import attributeselection.AbstractAttributeSelector;
import attributeselection.InfoGainEval;
import filters.ApplyClassBalancer;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import picocli.CommandLine;
import properties.versionproviders.DisruptorVersionProvider;
import roc.ROCDatasetsList;
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
import java.util.*;
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

    private String runFolderName = PARENT_FOLDER;
    private String baseFolderName = "";
    String startDate = "";
    private String experimentFolderName = EXPERIMENT_FOLDER;
    private ArrayList<Attack> attacksList = new ArrayList<>();
    private ArrayList<Classifier> classifiersList = new ArrayList<>();
    private ArrayList<Instances> perturbedDatasets = new ArrayList<>();
    ROCDatasetsList perturbedDataMapForROC = new ROCDatasetsList();
    private Instances testSet;
    private final Exporter arffExport = new Exporter( new ArffSaver() );
    private final Exporter csvExport = new Exporter( new CSVSaver() );

    /**
     * List of feature selection algorithms
     * @param featureSelectionAlgorithms  List of feature selection algorithms
     * @return List of feature selection algorithms
     */
    @Getter
    @Setter
    private List<AbstractAttributeSelector> featureSelectionAlgorithms = new ArrayList<>();

    /**
     * Map between the feature selection algorithm and the corresponding selected features
     * @param selectedFeatureMap Map between the feature selection algorithm and the corresponding selected features
     * @return Map between the feature selection algorithm and the corresponding selected features
     */
    @Getter @Setter
    private Map<AbstractAttributeSelector, double[][]> selectedFeatureMap = new HashMap<>();


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
            names = {"-K", "--knowledge"},
            description= "Comma-separated knowledge for the attacks.\nThe knowledge is a percentage between 0 and 1.\ne.g. -K 0.2,0.5,1\nDefault: 1\n",
            paramLabel="KNOWLEDGE",
            defaultValue="1",
            split = "," )
    private ArrayList<Double> knowledgeList = new ArrayList<>();

    @CommandLine.Option(
            names = {"-t", "--train-percent"},
            description= "Percentage of the training set.\nSet to 1 if they want to use all the dataset as training set \nThe percentage is a number between 0 and 1.\nDefault: 0.8\n",
            paramLabel="TRAIN_PERCENTAGE",
            defaultValue="0.8" )
    private double trainPercentage;

    @CommandLine.Option(
            names = {"-b", "--balance"},
            description = "Perform other 2 run of the attacks on balanced dataset\nIn the first additional run, the instances are balanced with Resample.\nFor the second run is used SMOTE instead\n",
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

        populateFeatureSelectionAlgorithmsList( dataset );
        performFeatureSelection();

        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HHmmss");
        startDate = simpleDateFormat.format(new Date());

        baseFolderName = PARENT_FOLDER
                + File.separator
                + startDate;

        if(toBalance) {
            log.info("\n------------------------------------------------------------------------------------------------------------------------------------\n" +
                    "\t-- WITHOUT BALANCING --" +
                    "\n------------------------------------------------------------------------------------------------------------------------------------");
            // Run the main disruptor loop without balancing
            disrupt(dataset);

            // Run the main disruptor loop balancing with Resample filter
            log.info("\n------------------------------------------------------------------------------------------------------------------------------------\n" +
                    "\t-- RESAMPLE BALANCING --" +
                    "\n------------------------------------------------------------------------------------------------------------------------------------");
            disrupt(ApplyClassBalancer.resample(dataset));

            // Run the main disruptor loop balancing with SMOTE filter
            log.info("\n------------------------------------------------------------------------------------------------------------------------------------\n" +
                    "\t-- SMOTE BALANCING --" +
                    "\n------------------------------------------------------------------------------------------------------------------------------------");
            disrupt(ApplyClassBalancer.smote(dataset));
        }
        else {
            // Run the main disruptor loop without balancing
            disrupt(dataset);
        }

        return 0;
    }

    /**
     *
     * Main disruptor loop.
     * For each feature selection algorithm perform the number of runs defined.
     * Each run perform every attack present in the attacksList
     *
     * @param dataset input dataset
     * @throws Exception
     */
    private void disrupt(Instances dataset) throws Exception {

        for( AbstractAttributeSelector attributeSelectorAlgorithm : selectedFeatureMap.keySet() ){

            log.info("\n\n===========================================\nfeature selection algorithm: {}\n===========================================\n", attributeSelectorAlgorithm.getName());

            for( int runNumber=0; runNumber<runs; runNumber++ ){
                executeRun(dataset, attributeSelectorAlgorithm, runNumber);
            }

            if(experimenter){
                // Append the test set to each dataset
                appendTestSet(true);
                // Evaluate the effectiveness of the attacks
                evaluateAttacks();
            }

            if(roc){
                showROCsForAttacks();
            }

            clearFieldsAfterAllRuns();

        }
    }

    private void executeRun(Instances dataset, AbstractAttributeSelector attributeSelectorAlgorithm, int run) throws Exception {
        log.info("\n\n{}: RUN {} ----------------------------------\n", attributeSelectorAlgorithm.getName(), run);

        // Creates a copy of the starting instances otherwise a test set is added at every run growing exponentially
        Instances runDataset = new Instances(dataset);

        // Set folder name
        runFolderName = baseFolderName
                + File.separator
                + attributeSelectorAlgorithm.getName()
                + File.separator
                + "run" + run;

        // Split Train and Test set
        Instances[] splitTrainTest = InstancesUtil.splitTrainTest(runDataset, trainPercentage, run);
        Instances trainset = splitTrainTest[0];
        testSet = splitTrainTest[1];

        if(experimenter){
            // To use as a reference, add the input dataset as the first list element
            perturbedDatasets.add(trainset);
        }

        // Export test set
        exportTestSet(testSet);

        // Populate the attacks and the classifiers lists
        populateAttacksList(trainset, selectedFeatureMap.get(attributeSelectorAlgorithm));
        populateClassifiersList();

        // Attack main loop
        performAttacks(trainset, attacksList, capacitiesList, attributeSelectorAlgorithm);
    }

    private void clearFieldsAfterAllRuns() {
        perturbedDatasets.clear();
        if(roc){
            perturbedDataMapForROC.clear();
        }
    }

    private void showROCsForAttacks() {
        perturbedDataMapForROC.keySet().forEach( attackName -> {

            Map<Double, Instances> attackPerturbedDatasets = perturbedDataMapForROC.getCapacitiesMap(attackName) ;

            log.info("Started ROC curves visualization for attack {}", attackName);
            log.info("Running...");

            for(Classifier classifier : classifiersList){
                log.debug("Started ROC for attack {} and classifier {}", attackName, classifier.getClass().getSimpleName());
                ROCGenerator rocGenerator = new ROCGenerator(testSet, classifier, attackName);
                rocGenerator.visualizeROCCurves(new ArrayList<>(attackPerturbedDatasets.values()));
                log.debug("Finished ROC for attack {} and classifier {}", attackName, classifier.getClass().getSimpleName());
            }

            log.info("Finished ROC curves visualization for attack {}", attackName);

        } );
    }

    /**
     * Fill the attacks list with all the attacks
     * @param dataset dataset to perturbate during the attacks
     * @param selectedFeatures features to perturbate during the attacks
     */
    private void populateAttacksList(Instances dataset, double[][] selectedFeatures) {
        attacksList.clear();

        attacksList.add(new LabelFlipping(dataset));
        attacksList.add(new RandomLabelFlipping(dataset));

        NullAttack nullAttack = new NullAttack(dataset);
        nullAttack.setFeatureSelected(selectedFeatures);
        attacksList.add(nullAttack);

        MeanAttack meanAttack = new MeanAttack(dataset);
        meanAttack.setFeatureSelected(selectedFeatures);
        attacksList.add(meanAttack);

        MeanPerClassAttack meanPerClassAttack = new MeanPerClassAttack(dataset);
        meanPerClassAttack.setFeatureSelected(selectedFeatures);
        attacksList.add(meanPerClassAttack);

        OutOfRanging outOfRanging = new OutOfRanging(dataset);
        outOfRanging.setFeatureSelected(selectedFeatures);
        attacksList.add(outOfRanging);

        RandomValueFromOtherClass randomValueFromOtherClass = new RandomValueFromOtherClass(dataset);
        randomValueFromOtherClass.setFeatureSelected(selectedFeatures);
        attacksList.add(randomValueFromOtherClass);

        MiddlePoint middlePointAttack = new MiddlePoint(dataset);
        middlePointAttack.setFeatureSelected(selectedFeatures);
        attacksList.add(middlePointAttack);

        MiddlePointByClass middlePointByClassAttack = new MiddlePointByClass(dataset);
        middlePointByClassAttack.setFeatureSelected(selectedFeatures);
        attacksList.add(middlePointByClassAttack);
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
     * @param attributeSelectorAlgorithm
     */
    private void performAttacks(Instances trainingSet, ArrayList<Attack> attacksList, ArrayList<Double> capacitiesList, AbstractAttributeSelector attributeSelectorAlgorithm){
        // Nested loop between attacks list and capacities list
        attacksList.forEach( attack -> {
            String attackClassName = attack.getClass().getSimpleName();
            log.info("Started attack {}", attackClassName);
            String attackName = trainingSet.relationName() + "_" + attackClassName;

            capacitiesList.forEach(capacity -> {
                log.info("\tcapacity: {}\t knowledge: {}", capacity, attributeSelectorAlgorithm.getKnowledge());

                // Define an attack code unique for this attack run
                String attackCode = attackName +
                        "_" + attributeSelectorAlgorithm.getName() +
                        "_K" + attributeSelectorAlgorithm.getKnowledge() +
                        "_C" + capacity ;

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
                    try {
                        perturbedDataMapForROC.addWithCapacity(attackName, capacity, perturbedInstances);
                    } catch (Exception e) {
                        log.error("Problem storing the perturbed dataset for the ROC curve");
                        log.debug(attackCode);
                        ExceptionUtil.logException(e, log);
                    }
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
        arffExport.exportInFolder( perturbedDataset, runFolderName, attackCode );
        // Export CSV
        csvExport.exportInFolder( perturbedDataset, runFolderName, attackCode );
    }

    private void exportTestSet(Instances testSet) throws IOException {
        // Export ARFF
        arffExport.exportInFolder( testSet, runFolderName, testSet.relationName()+"_TEST" );
        // Export CSV
        csvExport.exportInFolder( testSet, runFolderName, testSet.relationName()+"_TEST" );
    }

    private void exportTrainTestSet(Instances trainTestSet) throws IOException {
        // Export ARFF
        arffExport.exportInFolder( trainTestSet, runFolderName + File.separator + "trainTest", trainTestSet.relationName());
        // Export CSV
        csvExport.exportInFolder( trainTestSet, runFolderName + File.separator + "trainTest", trainTestSet.relationName() );
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
        DisruptorExperiment experiment = new DisruptorExperiment(perturbedDatasets, trainPercentage, baseFolderName);
        experiment.setClassifiersList(classifiersList);
        experiment.start();
    }



    /**
     * @param featureSelectionAlgorithm the algorithm that perform the feature selection
     * @return the list of the selected feature based on the feature selection algorithms and the feature set capacity
     */
    public double[][] getSelectedFeatures(AbstractAttributeSelector featureSelectionAlgorithm){
        return selectedFeatureMap.get(featureSelectionAlgorithm);
    }

    /**
     * Populate the list of feature selection algorithms
     * @param dataset dataset on which the feature selection is performed
     */
    public void populateFeatureSelectionAlgorithmsList(Instances dataset){
        featureSelectionAlgorithms.clear();
        featureSelectionAlgorithms.add(new InfoGainEval(dataset));
        featureSelectionAlgorithms.add(new RandomSelector(dataset));

        addKnowledge();
    }

    /**
     * Add to featureSelectionAlgorithms an algorithm for each knowledge selected
     */
    private void addKnowledge(){
        List<AbstractAttributeSelector> newFeatureSelectionAlgorithms = new ArrayList<>();

        featureSelectionAlgorithms.forEach( algorithm -> {
            knowledgeList.forEach( knowledge -> {
                AbstractAttributeSelector newAlgorithm = algorithm.copy();
                newAlgorithm.setKnowledge( knowledge );
                newFeatureSelectionAlgorithms.add( newAlgorithm );
            } );
        } );

        featureSelectionAlgorithms = newFeatureSelectionAlgorithms;

    }

    /**
     * For each Feature selection algorithm of featureSelectionAlgorithms, perform the attribute selection of the target
     * and store the ranked attributes in the selectedFeatureMap
     */
    public void performFeatureSelection(){
        knowledgeList.forEach( knowledge -> {

            log.info("\tknowledge: {}", knowledge);

            for(AbstractAttributeSelector fsAlgorithm : featureSelectionAlgorithms){
                fsAlgorithm.eval();
                double[][] rankedAttributes = fsAlgorithm.getRankedAttributes();
                selectedFeatureMap.put( fsAlgorithm, rankedAttributes );
            }

        });
    }
}
