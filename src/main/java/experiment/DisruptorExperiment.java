package experiment;

import lombok.Getter;
import lombok.Setter;
import saver.Exporter;
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
        experiment.setRunUpper(10);

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
            e.printStackTrace();
        }

        experiment.setResultProducer(rsrp);
        experiment.setPropertyPath(propertyPath);
    }

    private void setupClassifiers() {
        ArrayList<Classifier> classifiersList = new ArrayList<>();
        classifiersList.add( new RandomForest() );
        classifiersList.add( new NaiveBayes() );
        classifiersList.add( new J48() );
        classifiersList.add( new BayesNet() );
        experiment.setPropertyArray( classifiersList.toArray() );
    }

    private void setupDatasets() {
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
    }


    /**
     * Weka Experimenter "Run"
     * @throws Exception
     */
    private void runExperiment() throws Exception {
        // Save result in an arff file
        instancesResultListener.setOutputFile(new File(experimentFolderName+File.separator+"experimenterOutput.arff"));
        experiment.setResultListener(instancesResultListener);

        System.out.println("Initializing...");
        experiment.initialize();
        System.out.println("Running...");
        experiment.runExperiment();
        System.out.println("Finishing...");
        experiment.postProcess();
    }

    /**
     * Weka Experimenter "Analyse"
     * @throws Exception
     */
    private void analyseExperiment() throws Exception {
        System.out.println("Evaluating...");

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
            tester.multiResultsetFull(0, result.attribute("Percent_correct").index());
        else
            tester.multiResultsetFull(0, result.attribute("Correlation_coefficient").index());
    }

    private void printResults(PairedTTester tester) {
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
