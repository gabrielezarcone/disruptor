package experiment;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import saver.Exporter;
import util.ExceptionUtil;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Range;
import weka.core.converters.ArffSaver;
import weka.experiment.*;

import javax.swing.*;
import java.beans.IntrospectionException;
import java.beans.PropertyDescriptor;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

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
     * @param classifiersList list of classifier used for the evaluation
     * @return list of classifier used for the evaluation
     */
    @Getter @Setter
    private ArrayList<Classifier> classifiersList = new ArrayList<>();



    public DisruptorExperiment(ArrayList<Instances> perturbedDatasets, double trainPercentage, String outputFolderName){
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

        log.info("Initializing...");
        experiment.initialize();
        log.info("Running...");
        experiment.runExperiment();
        log.info("Finishing...");
        experiment.postProcess();
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

        tester.setResultMatrix(new ResultMatrixPlainText());
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
        ResultMatrix matrix = tester.getResultMatrix();
        log.info("Result:\n\n{}", matrix);
        for (int i = 0; i < matrix.getColCount(); i++) {
            log.info(matrix.getColName(i));
            log.info("    Perc. correct (mean): " + matrix.getMean(i, 0));
            log.info("    StdDev: " + matrix.getStdDev(i, 0));
        }
    }


}
