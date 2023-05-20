package roc;

import costants.Colors;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import util.ExceptionUtil;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

@Slf4j
public class ROCGenerator {

    @Getter @Setter
    Instances testSet;
    @Getter @Setter
    Classifier classifier;
    @Getter @Setter
    String rocName;
    @Getter @Setter
    Color curveColor = null;

    public ROCGenerator(Instances testSet, Classifier classifier, String rocName) {
        this.testSet = testSet;
        this.classifier = classifier;
        this.rocName = rocName;
    }

    /**
     * Generate a ROC curve as an Instances object using the defined classifier.
     * We use an Instances object because this is the way Weka handle ROC Curves
     * @param trainSet the train set we want tio calculate the curve
     * @return the instances object representing the ROC curve
     * @throws Exception if errors during the evaluation of the classifier
     */
    public Instances generateROC(Instances trainSet) throws Exception {
        classifier.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(classifier, testSet);

        ThresholdCurve thresholdCurve = new ThresholdCurve();
        return thresholdCurve.getCurve(eval.predictions());
    }

    /**
     * Shows one panel containing all the ROC curves of the different train set defined (Using the same test set)
     */
    public void visualizeROCCurves(List<Instances> trainSetList){
        boolean first = true;
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();

        for(int i=0; i<trainSetList.size(); i++){
        Instances trainSet = trainSetList.get(i);
            try {
                setCurveColor(Colors.distinctColors[i]);

                Instances rocCurve = generateROC(trainSet);
                PlotData2D plotData2D = generatePlotData2D(rocCurve, trainSet.relationName());
                setupROCPanel(vmc, rocCurve, plotData2D);

                if (first)
                    vmc.setMasterPlot(plotData2D);
                else
                    vmc.addPlot(plotData2D);
                first = false;

            } catch (Exception e) {
                log.error("Problem during the visualization of a ROC curve");
                log.debug("\ttrain set of the ROC: " + trainSet.relationName());
                ExceptionUtil.logException(e, log);
            }
        }

        showROCPanel(vmc);
    }

    /**
     *
     * @param trainSet
     */
    public void visualizeROCCurve(Instances trainSet){
        ArrayList<Instances> trainList = new ArrayList<>();
        trainList.add(trainSet);

        visualizeROCCurves(trainList);
    }



    /**
     * Visualize a panel containing the ROC curve using the defined classifier
     * @throws Exception if errors during the generation of the curve
     */
    private void setupROCPanel(ThresholdVisualizePanel vmc, Instances rocCurve, PlotData2D plotData2D) throws Exception {

        connectROCCurvePoints(rocCurve, plotData2D);

        decorateROCPanel(vmc, rocCurve);

    }



    /**
     * Create the panel to show the ROC curve
     * @param rocCurve ROC curve to show
     * @return the panel
     * @throws Exception
     */
    private void decorateROCPanel(ThresholdVisualizePanel vmc, Instances rocCurve) throws Exception {
        double rocArea = ThresholdCurve.getROCArea(rocCurve);
        String rocAreaString = Utils.doubleToString(rocArea, 4);
        vmc.setROCString("(Area under ROC = " + rocAreaString + ")");
        vmc.setName( rocCurve.relationName() );
    }

    /**
     * Generate plottable data of the ROC Curve
     * @param rocCurve ROC curve to plot
     * @param plotName
     * @return
     */
    private PlotData2D generatePlotData2D(Instances rocCurve, String plotName) {
        PlotData2D plotData2D = new PlotData2D(rocCurve);
        plotData2D.setPlotName(plotName);
        plotData2D.addInstanceNumberAttribute();
        if(curveColor!=null){
            plotData2D.setCustomColour(curveColor);
        }
        return plotData2D;
    }

    /**
     * Connect with a line every consecutive instance
     * @param rocCurve ROC curve to plot
     * @param plotData2D plottable Data of the ROC curve
     * @throws Exception
     */
    private void connectROCCurvePoints(Instances rocCurve, PlotData2D plotData2D) throws Exception {
        boolean[] connectedPoints = new boolean[rocCurve.numInstances()];
        for (int n = 1; n < connectedPoints.length; n++)
            connectedPoints[n] = true;
        plotData2D.setConnectPoints(connectedPoints);
    }

    /**
     * Show the panel containing the ROC curve
     * @param vmc the panel
     */
    private void showROCPanel(ThresholdVisualizePanel vmc) {
        final JFrame jf = new JFrame("Weka Classifier Visualize: " + rocName + "-" + classifier.getClass().getSimpleName());
        jf.setSize(1000,800);
        jf.getContentPane().setLayout(new BorderLayout());

        jf.getContentPane().add(vmc, BorderLayout.CENTER);
//        jf.addWindowListener(new java.awt.event.WindowAdapter() {
//            @Override
//            public void windowClosing(java.awt.event.WindowEvent e) {
//                jf.dispose();
//            }
//        });
//
        jf.setVisible(true);
        jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    }

}
