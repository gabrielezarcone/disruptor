package disruptor.core;


import disruptor.attacks.Attack;
import disruptor.attacks.horizontal.labelflipping.LabelFlipping;
import disruptor.attacks.horizontal.labelflipping.RandomLabelFlipping;
import disruptor.attacks.vertical.*;
import disruptor.attributeselection.AbstractAttributeSelector;
import disruptor.attributeselection.InfoGainEval;
import disruptor.attributeselection.RandomSelector;
import disruptor.perturbeddataset.PerturbedDataset;
import disruptor.perturbeddataset.PerturbedDatasetParams;
import disruptor.properties.versionproviders.DisruptorVersionProvider;
import disruptor.saver.Exporter;
import disruptor.util.CSVUtil;
import disruptor.experiment.DisruptorExperiment;
import disruptor.filters.ApplyClassBalancer;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import picocli.CommandLine;
import disruptor.roc.ROCDatasetsList;
import disruptor.util.ArffUtil;
import disruptor.util.ExceptionUtil;
import disruptor.util.InstancesUtil;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
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
        description = Disruptor.LOGO +
                "\nDisrupt the training set of a Machine Learning algorithm using a set of different attacks.\n",
        versionProvider = DisruptorVersionProvider.class,
        // mixinStandardHelpOptions attribute adds --help and --version options
        mixinStandardHelpOptions = true
)
public class Disruptor implements Callable<Integer> {

    public static final String PARENT_FOLDER = "output";
    public static final String EXPERIMENT_FOLDER = "experiment";
    protected static final String LOGO = "\n" +
            "  ___ ___ ___ ___ _   _ ___ _____ ___  ___ \n" +
            " |   \\_ _/ __| _ \\ | | | _ \\_   _/ _ \\| _ \\\n" +
            " | |) | |\\__ \\   / |_| |  _/ | || (_) |   /\n" +
            " |___/___|___/_|_\\\\___/|_|   |_| \\___/|_|_\\\n" +
            "                                           \n";

    private String runFolderName = PARENT_FOLDER;
    private String baseFolderName = "";
    String startDate = "";
    private String experimentFolderName = EXPERIMENT_FOLDER;
    @Getter
    private ArrayList<Attack> attacksList = new ArrayList<>();
    @Getter
    private ArrayList<Classifier> classifiersList = new ArrayList<>();
    @Getter
    private ArrayList<PerturbedDataset> perturbedDatasets = new ArrayList<>();
    ROCDatasetsList perturbedDataMapForROC = new ROCDatasetsList();
    private final Exporter arffExport = new Exporter( new ArffSaver() );
    private final Exporter csvExport = new Exporter( new CSVSaver() );
    private int executionCounter = 0;

    protected enum ExportType {ALL, NONE, ARFF, CSV}

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
    @Getter @Setter
    @CommandLine.Parameters(
            index = "0",
            description = "Filepath of the CSV file containing the dataset.\nIt is MANDATORY that the first row of the CSV file contains the features names.\nUse --arff to pass a .arff file instead\n",
            paramLabel = "DATASET")
    private File datasetFile;

    // CLI OPTIONS ---------------------------------------------------------------------------------------------------------------------------
    @Getter @Setter
    @CommandLine.Option(
            names = {"-a", "--arff"},
            description = "Use this option if the dataset file format is .arff\n",
            paramLabel = "ARFF")
    private boolean isArff = false;

    @Getter @Setter
    @CommandLine.Option(
            names = {"-c", "--class"},
            description= "Specify the class attribute name.\nIf this param is not set, the program use “class” as the class attribute name\nDefault: class\n",
            paramLabel="CLASS",
            defaultValue="class")
    private String className = "class";

    @Getter @Setter
    @CommandLine.Option(
            names = {"-C", "--capacities"},
            description= "Comma-separated horizontal capacities for the attacks.\nThe capacity is a percentage between 0 and 1.\ne.g. -C 0.2,0.5,1\nDefault: 1\n",
            paramLabel="CAPACITY",
            defaultValue="1",
            split = "," )
    private ArrayList<Double> capacitiesList = new ArrayList<>(Collections.singletonList(1d));

    @Getter @Setter
    @CommandLine.Option(
            names = {"-F", "--features-capacities"},
            description= "Comma-separated vertical capacities for the attacks.\nThe capacity is a percentage between 0 and 1.\ne.g. -C 0.2,0.5,1\nDefault: 1\n",
            paramLabel="FEATURES_CAPACITY",
            defaultValue="1",
            split = "," )
    private ArrayList<Double> featuresCapacitiesList = new ArrayList<>(Collections.singletonList(1d));

    @Getter @Setter
    @CommandLine.Option(
            names = {"-K", "--knowledge"},
            description= "Comma-separated knowledge for the attacks.\nThe knowledge is a percentage between 0 and 1.\ne.g. -K 0.2,0.5,1\nDefault: 1\n",
            paramLabel="KNOWLEDGE",
            defaultValue="1",
            split = "," )
    private ArrayList<Double> knowledgeList = new ArrayList<>(Collections.singletonList(1d));

    @Getter @Setter
    @CommandLine.Option(
            names = {"-t", "--train-percent"},
            description= "Percentage of the training set.\nSet to 1 if they want to use all the dataset as training set \nThe percentage is a number between 0 and 1.\nDefault: 0.8\n",
            paramLabel="TRAIN_PERCENTAGE",
            defaultValue="0.8" )
    private double trainPercentage = 0.8;

//    @Getter @Setter
//    @CommandLine.Option(
//            names = {"-b", "--balance"},
//            description = "Perform other 2 run of the attacks on balanced dataset\nIn the first additional run, the instances are balanced with Resample.\nFor the second run is used SMOTE instead\n",
//            paramLabel = "BALANCE")
    private boolean toBalance;

    @Getter @Setter
    @CommandLine.Option(
            names = {"-e", "--experimenter"},
            description = "Evaluate the effectiveness of the attacks using several ML algorithms\n",
            paramLabel = "EXP")
    private boolean experimenter = false;

//    @Getter @Setter
//    @CommandLine.Option(
//            names = {"-r", "--roc"},
//            description = "Show the ROC curves for each attack\n",
//            paramLabel = "ROC")
    private static boolean roc;

    @Getter @Setter
    @CommandLine.Option(
            names = {"-R", "--runs"},
            description = "Define the number of runs for each attack\nDefault: 10\n",
            paramLabel = "NUMBER_OF_RUNS",
            defaultValue="10")
    private static int runs = 10;

    @Getter @Setter
    @CommandLine.Option(
            names = {"-E", "--export"},
            description = "Valid values: ${COMPLETION-CANDIDATES}\nDefine what kind of export the application should produce.\nIf --experiment is present, this option does not influence the Experimenter exports.",
            defaultValue="ALL"
    )
    private ExportType exportType = ExportType.ALL;


    public static void main(String[] args) {
        int exitCode = new CommandLine(new Disruptor()).execute(args);
        if(!roc){
            System.exit(exitCode);
        }
    }


    // EXECUTION ---------------------------------------------------------------------------------------------------------------------------
    @Override
    public Integer call() throws Exception {

        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HHmmss");
        startDate = simpleDateFormat.format(new Date());

        log.info(LOGO+"\tFILE: "+datasetFile.getName()+"\n\tSTART TIMESTAMP: "+startDate+"\n");

        baseFolderName = PARENT_FOLDER
                + File.separator
                + startDate;

        // Read the dataset file
        Instances dataset;
        if(isArff){
            dataset = ArffUtil.readArffFile(datasetFile, className);
        }
        else {
            dataset = CSVUtil.readCSVFile(datasetFile, className);
            arffExport.exportInFolder( dataset, baseFolderName, dataset.relationName() );
        }

        featureSelectionAlgorithms.clear();
        populateFeatureSelectionAlgorithmsList( dataset );
        performFeatureSelection();

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

            executionCounter++;

            log.info("\n\n===========================================\nfeature selection algorithm: {} K:{}\n===========================================\n", attributeSelectorAlgorithm.getName(), attributeSelectorAlgorithm.getKnowledge());

            for( int runNumber=0; runNumber<runs; runNumber++ ){
                executeRun(dataset, attributeSelectorAlgorithm, runNumber);
            }

            if(experimenter){
                // Append the test set to each dataset
                appendTestSet(true);
                // Evaluate the effectiveness of the attacks
                SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                String now = simpleDateFormat.format(new Date());
                String expResultTitle = "["+now+"]\t"+ attributeSelectorAlgorithm.getName() + "\tknowledge: " + attributeSelectorAlgorithm.getKnowledge() + "\nRanked features: " + Arrays.deepToString(attributeSelectorAlgorithm.getRankedAttributes());
                evaluateAttacks(expResultTitle);
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
        Instances testSet = splitTrainTest[1];

        if(experimenter){
            // To use as a reference, add the input dataset as the first list element
            PerturbedDatasetParams params = new PerturbedDatasetParams(attributeSelectorAlgorithm.getName(), run);
            PerturbedDataset perturbedDataset = new PerturbedDataset(trainset, testSet, params);
            perturbedDatasets.add(perturbedDataset);
        }

        // Export test set
        exportTestSet(testSet);

        // Populate the attacks and the classifiers lists
        attacksList.clear();
        populateAttacksList(trainset, selectedFeatureMap.get(attributeSelectorAlgorithm));
        classifiersList.clear();
        populateClassifiersList();

        // Attack main loop
        performAttacks(trainset, testSet, attacksList, capacitiesList, featuresCapacitiesList, attributeSelectorAlgorithm, run);
    }

    private void clearFieldsAfterAllRuns() {
        perturbedDatasets.clear();
        if(roc){
            perturbedDataMapForROC.clear();
        }
    }

    private void showROCsForAttacks() {
//        perturbedDataMapForROC.keySet().forEach( attackName -> {
//
//            HashMap<ROCDatasetsList.CapacitiesPair, Instances> attackPerturbedDatasets = perturbedDataMapForROC.getCapacitiesMap(attackName) ;
//
//            log.info("Started ROC curves visualization for attack {}", attackName);
//            log.info("Running...");
//
//            for(Classifier classifier : classifiersList){
//                log.debug("Started ROC for attack {} and classifier {}", attackName, classifier.getClass().getSimpleName());
//                ROCGenerator rocGenerator = new ROCGenerator(testSet, classifier, attackName);
//                rocGenerator.visualizeROCCurves(new ArrayList<>(attackPerturbedDatasets.values()));
//                log.debug("Finished ROC for attack {} and classifier {}", attackName, classifier.getClass().getSimpleName());
//            }
//
//            log.info("Finished ROC curves visualization for attack {}", attackName);
//
//        } );
    }

    /**
     * Fill the attacks list with all the attacks
     * @param dataset dataset to perturbate during the attacks
     * @param selectedFeatures features to perturbate during the attacks
     */
    protected void populateAttacksList(Instances dataset, double[][] selectedFeatures) {

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
    protected void populateClassifiersList() {
        classifiersList.add( new J48() );
        classifiersList.add( new SMO());
        classifiersList.add( new JRip() );
    }

    /**
     * Perform all the attacks defined in the attacksList using all the capacities defined in the capacitiesList
     * @param trainingSet training set to perturb
     * @param attacksList list of attacks to perform
     * @param capacitiesList list of capacities
     * @param featuresCapacitiesList list of capacities for features
     * @param attributeSelectorAlgorithm
     */
    private void performAttacks(Instances trainingSet, Instances testSet, ArrayList<Attack> attacksList, ArrayList<Double> capacitiesList, ArrayList<Double> featuresCapacitiesList, AbstractAttributeSelector attributeSelectorAlgorithm, int run){
        // Nested loop between attacks list and capacities list
        attacksList.forEach( attack -> {
            String attackClassName = attack.getClass().getSimpleName();
            log.info("Started attack {}", attackClassName);
            String attackName = trainingSet.relationName() + "_" + attackClassName;

            featuresCapacitiesList.forEach( featureCapacity -> {
                capacitiesList.forEach(capacity -> {

                    String fsAlgorithmName = attributeSelectorAlgorithm.getName();
                    double knowledge = attributeSelectorAlgorithm.getKnowledge();

                    log.info("\tfeatures capacity: {}\tcapacity: {}\t knowledge: {}", featureCapacity, capacity, knowledge);

                    // Define an attack code unique for this attack run
                    String attackCode = attackName +
                            "_" + fsAlgorithmName +
                            "_K" + knowledge +
                            "_F" + featureCapacity +
                            "_C" + capacity ;

                    // Perform this attack with this capacity
                    Instances trainingSetCopy = new Instances(trainingSet);
                    attack.setTarget( trainingSetCopy );
                    attack.setCapacity( capacity );
                    attack.setFeaturesCapacity( featureCapacity );
                    Instances perturbedInstances = attack.attack();
                    perturbedInstances.setRelationName(attackCode);

                    if(experimenter){
                        PerturbedDatasetParams params = new PerturbedDatasetParams(fsAlgorithmName, attack, capacity, featureCapacity, knowledge, run);
                        PerturbedDataset perturbedDatasetObject = new PerturbedDataset(perturbedInstances, testSet, params);
                        perturbedDatasets.add(perturbedDatasetObject);
                    }
                    if(roc){
//                        try {
//                            perturbedDataMapForROC.addWithCapacity(attackName, capacity, featureCapacity, perturbedInstances);
//                        } catch (Exception e) {
//                            log.error("Problem storing the perturbed dataset for the ROC curve");
//                            log.debug(attackCode);
//                            ExceptionUtil.logException(e, log);
//                        }
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
            log.info("Finished attack {}", attackClassName);
        });
    }

    /**
     * Export the perturbed dataset in the same folder of the others attack
     * @param attackCode String used to identify this particular attack execution used as file name
     * @param perturbedDataset The perturbed dataset after the attack
     * @throws IOException if problems during the export
     */
    private void exportPerturbedDataset(String attackCode, Instances perturbedDataset) throws IOException {
        if (exportType != ExportType.NONE){
            if(exportType == ExportType.ARFF || exportType == ExportType.ALL){
                // Export ARFF
                arffExport.exportInFolder( perturbedDataset, runFolderName, attackCode );
            }
            if(exportType == ExportType.CSV || exportType == ExportType.ALL){
                // Export CSV
                csvExport.exportInFolder( perturbedDataset, runFolderName, attackCode );
            }
        }
    }

    private void exportTestSet(Instances testSet) throws IOException {
        if (exportType != ExportType.NONE){
            if(exportType == ExportType.ARFF || exportType == ExportType.ALL){
                // Export ARFF
                arffExport.exportInFolder( testSet, runFolderName, testSet.relationName()+"_TEST" );
            }
            if(exportType == ExportType.CSV || exportType == ExportType.ALL){
                // Export CSV
                csvExport.exportInFolder( testSet, runFolderName, testSet.relationName()+"_TEST" );
            }
        }
    }

    private void exportTrainTestSet(Instances trainTestSet) throws IOException {
        if (exportType != ExportType.NONE){
            if(exportType == ExportType.ARFF || exportType == ExportType.ALL){
                // Export ARFF
                arffExport.exportInFolder( trainTestSet, runFolderName + File.separator + "trainTest", trainTestSet.relationName());
            }
            if(exportType == ExportType.CSV || exportType == ExportType.ALL){
                // Export CSV
                csvExport.exportInFolder( trainTestSet, runFolderName + File.separator + "trainTest", trainTestSet.relationName() );
            }
        }
    }


    /**
     * Append the test set to every dataset present in perturbedDatasets
     * @param export true if the train+test file should be exported
     */
    private void appendTestSet(boolean export){
        perturbedDatasets.forEach( perturbedDataset -> {
            try {
                InstancesUtil.addAllInstances( perturbedDataset.getDataset(), perturbedDataset.getTestSet());

                if(export){
                    exportTrainTestSet( perturbedDataset.getDataset() );
                }

            } catch (Exception e) {
                log.error("Problem appending the test set to the train set");
                ExceptionUtil.logException(e, log);
            }
        });
    }


    /**
     * Evaluate the effectiveness of the attacks using several ML algorithms
     * @param resultsTitle title to show with the results in the log
     */
    private void evaluateAttacks(String resultsTitle) throws Exception {
        DisruptorExperiment experiment = new DisruptorExperiment(perturbedDatasets, trainPercentage, baseFolderName);
        if(executionCounter==1){
            experiment.logInfo( "\n" +
                    "=============================================================\n" +
                    "             START OF THE EXPERIMENT ["+startDate+"]                    \n" +
                    "=============================================================\n"
            );
        }
        experiment.setClassifiersList(classifiersList);
        experiment.setResultsTitle(resultsTitle);
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
    protected void populateFeatureSelectionAlgorithmsList(Instances dataset){
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
                AbstractAttributeSelector newAlgorithm = null;
                try {
                    newAlgorithm = algorithm.copy();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
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

        for(AbstractAttributeSelector fsAlgorithm : featureSelectionAlgorithms){
            fsAlgorithm.eval();
            double[][] rankedAttributes = fsAlgorithm.getRankedAttributes();
            selectedFeatureMap.put( fsAlgorithm, rankedAttributes );
        }

    }
}
