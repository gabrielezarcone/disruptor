package experiment;

import experiment.resultmatrix.ResultMatrixPlainTextArray;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import saver.Exporter;
import util.ExceptionUtil;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Range;
import weka.core.converters.ArffSaver;
import weka.experiment.*;

import javax.swing.*;
import java.beans.IntrospectionException;
import java.beans.PropertyDescriptor;
import java.io.*;
import java.util.ArrayList;

@Slf4j
public class DisruptorExperiment {
    /**
     * @param experiment this experiment
     * @return this experiment
     */
    @Getter @Setter
    private Experiment experiment = new Experiment();

    /**
     * @param perturbedDatasets list of dataset to use during the experiment
     * @return list of dataset to use during the experiment
     */
    @Getter @Setter
    private ArrayList<Instances> perturbedDatasets;

    /**
     * @param trainPercentage percentage to use for the train set
     * @return percentage to use for the train set
     */
    @Getter @Setter
    private double trainPercentage;

    /**
     * @param experimentFolderName name of the folder for the outputs
     * @return name of the folder for the outputs
     */
    @Getter @Setter
    private String experimentFolderName = "experiment";

    /**
     * @param classification true if is a classification experiment
     * @return true if is a classification experiment
     */
    @Getter @Setter
    private boolean classification = false;

    /**
     * @param instancesResultListener Outputs the received results in arff format to a Writer
     * @return Outputs the received results in arff format to a Writer
     */
    @Getter @Setter
    private InstancesResultListener instancesResultListener = new InstancesResultListener();

    /**
     * @param instancesResultListener Outputs the received results in database to a Writer
     * @return Outputs the received results in database to a Writer
     */
    @Getter @Setter
    private DatabaseResultListener databaseResultListener = new DatabaseResultListener();

    /**
     * @param classifiersList list of classifier used for the evaluation
     * @return list of classifier used for the evaluation
     */
    @Getter @Setter
    private ArrayList<Classifier> classifiersList = new ArrayList<>();

    @Getter @Setter
    private String resultsTitle = "";



    public DisruptorExperiment(ArrayList<Instances> perturbedDatasets, double trainPercentage, String outputFolderName) throws Exception {
        setPerturbedDatasets(perturbedDatasets);
        setTrainPercentage(trainPercentage);
        setExperimentFolderName(outputFolderName + File.separator + experimentFolderName);
    }



    /**
     * Evaluate the effectiveness of the attacks using several ML algorithms
     */
    public void start() throws Exception {
        // Use the Experimenter Weka API for the evaluation
        // 1. setup the experiment ------------------------------------------------------------------------------------------
        setupExperiment();
        // 2. run experiment -------------------------------------------------------------------------------------------
        runExperiment();
        // 3. calculate statistics and output them -------------------------------------------------------------------------------------------
        analyseExperiment();
    }


    /**
     * Weka Experimenter "Setup"
     */
    private void setupExperiment() {
        experiment.setPropertyArray(new Classifier[0]);
        experiment.setUsePropertyIterator(true);


        SplitEvaluator splitEvaluator = null;
        Classifier classifier = null;
        Instances dataset = perturbedDatasets.get(0);
        if ( dataset.classAttribute().isNominal() ) {
            classification = true;
            splitEvaluator  = new ClassifierSplitEvaluator();
            classifier = ((ClassifierSplitEvaluator) splitEvaluator).getClassifier();
        }
        else if ( dataset.classAttribute().isNumeric() ) {
            splitEvaluator  = new RegressionSplitEvaluator();
            classifier = ((RegressionSplitEvaluator) splitEvaluator).getClassifier();
        }
        else {
            throw new IllegalArgumentException("The class attribute is neither nominal nor numeric ");
        }

        // Split train and test preserving the order
        setupSplitPercentageInOrder(splitEvaluator, classifier);

        // Set the runs number:
        experiment.setRunLower(1);
        experiment.setRunUpper(1);

        // Set classifiers
        setupClassifiers();

        // Multiple datasets
        setupDatasets();
    }

    private void setupSplitPercentageInOrder(SplitEvaluator splitEvaluator, Classifier classifier) {
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
            log.error("Problem during introspection");
            ExceptionUtil.logException(e, log);
        }

        experiment.setResultProducer(rsrp);
        experiment.setPropertyPath(propertyPath);
    }

    private void setupClassifiers() {
        experiment.setPropertyArray( classifiersList.toArray() );
    }

    private void setupDatasets() {
        DefaultListModel<File> model = new DefaultListModel<>();
        for ( int i=0; i<perturbedDatasets.size(); i++){
            Instances perturbedDataset = perturbedDatasets.get(i);
            Exporter arffExport = new Exporter( new ArffSaver() );
            try {
                arffExport.exportInFolder( perturbedDataset, experimentFolderName, perturbedDataset.relationName()+"_EXP_"+i );
                File datasetFile = arffExport.getExportedFile();
                model.addElement(datasetFile);
            } catch (IOException e) {
                log.error("Problem adding datasets to the experiment");
                log.debug("dataset relation name: " + perturbedDataset.relationName() );
                ExceptionUtil.logException(e, log);
            }
        }
        experiment.setDatasets(model);
    }


    /**
     * Weka Experimenter "Run"
     * @throws Exception
     */
    private void runExperiment() throws Exception {
        // Save result in an arff file
        instancesResultListener.setOutputFile(new File(experimentFolderName+File.separator+"experimenterOutput.arff"));
        experiment.setResultListener(instancesResultListener);

        log.info("\n\n:::::::: {} ::::::::\n", getResultsTitle());
        log.info("Initializing...");
        experiment.initialize();
        log.info("Running...");
        boolean verbose = false;
        experiment.runExperiment(verbose);
        log.info("Finishing...");
        experiment.postProcess();

        // Save result in database
//        File fileProps = new File(getClass().getClassLoader().getResource("DatabaseUtils.props").toURI());
//        databaseResultListener.initialize(fileProps);
//        databaseResultListener.setDatabaseURL("jdbc:postgresql://localhost:5432/postgres");
//        databaseResultListener.setUsername("yourUserName");
//        databaseResultListener.setPassword("yourPassword");
//        experiment.setResultListener(databaseResultListener);
//
//        log.info("\n\n:::::::: {} ::::::::\n", getResultsTitle());
//        log.info("Initializing...");
//        experiment.initialize();
//        log.info("Running...");
//        experiment.runExperiment(true);
//        log.info("Finishing...");
//        experiment.postProcess();
    }

    /**
     * Weka Experimenter "Analyse"
     * @throws Exception
     */
    private void analyseExperiment() throws Exception {
        log.info("Evaluating...");

        PairedTTester tester = new PairedCorrectedTTester();
        Instances result = new Instances(new BufferedReader(new FileReader(instancesResultListener.getOutputFile())));
        tester.setInstances(result);
        tester.setSortColumn(-1);
        tester.setRunColumn(result.attribute("Key_Run").index());

        setupTesterRowsCols(tester, result);

        tester.setResultMatrix(new ResultMatrixPlainTextArray());
        tester.setDisplayedResultsets(null);
        tester.setSignificanceLevel(0.05);
        tester.setShowStdDevs(true);

        // fill result matrix (but discarding the output)
        setupTesterComparisionField(tester, result);

        // output results for reach dataset
        printResults(tester);
    }

    private void setupTesterRowsCols(PairedTTester tester, Instances result) {
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
    }

    private void setupTesterComparisionField(PairedTTester tester, Instances result) throws Exception {
        if (classification)
            tester.multiResultsetFull(result.attribute("Key_Dataset").index(), result.attribute("Percent_correct").index());
        else
            tester.multiResultsetFull(result.attribute("Key_Dataset").index(), result.attribute("Correlation_coefficient").index());
    }

    private void printResults(PairedTTester tester) {
        ResultMatrixPlainTextArray matrix = (ResultMatrixPlainTextArray) tester.getResultMatrix();
        log.info("Results:\n\n{}\n", getResultsTitle());
        printResultsPlainText(matrix);
        printResultsCSV(matrix);
    }

    private static void printResultsPlainText(ResultMatrix matrix) {
        log.info("{}", matrix);
        for (int i = 0; i < matrix.getColCount(); i++) {
            log.info(matrix.getColName(i));
            log.info("    Perc. correct (mean): " + matrix.getMean(i, 0));
            log.info("    StdDev: " + matrix.getStdDev(i, 0));
        }
    }

    private void printResultsCSV(ResultMatrixPlainTextArray matrix) {
        String[][] matrixArray = matrix.toArray();
        String columnName = "";
        int columnsSpan = 2; //Number of columns of the matrix used by the datasets. It's 2 only for the base dataset (the first). FOR THE OTHERS IS 3

        StringWriter sw = new StringWriter();
        CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                .setHeader("dataset", "classifier", "correctness", "stDev", "v/ /*")
                .build();

        final CSVPrinter printer;
        try {
            printer = new CSVPrinter(sw, csvFormat);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        for (int c = 1; c < matrixArray[0].length; c+=columnsSpan) {
            columnName = matrixArray[0][c].equals("") ? columnName : matrixArray[0][c]; // for the columns without a name use the previous name

            for (int r = 1; r < matrixArray.length; r++) {
                String rowName = matrixArray[r][0];
                try {

                    String evaluation;
                    if(c==1){
                        evaluation = "";
                    }
                    else {
                        evaluation = matrixArray[r][c+2];
                        columnsSpan = 3;
                    }

                    printer.printRecord(columnName, rowName, matrixArray[r][c], matrixArray[r][c+1], evaluation);

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

        }
        log.info("{}", sw.toString().trim());

    }

    public void logInfo(String message){
        log.info(message);
    }


}
