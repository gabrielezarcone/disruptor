package attributeselection;

import lombok.Getter;
import lombok.Setter;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;

import java.util.HashMap;

public class InfoGainEval {

    @Getter @Setter
    Instances targetInstances;
    @Getter @Setter
    HashMap<Double, Double> featureRanksMap = new HashMap<>();
    @Getter @Setter
    double[][] rankedAttributes;

    public InfoGainEval(Instances instances){
        this.targetInstances = instances;
    }

    public void eval() {
        Instances instances = getTargetInstances();

        AttributeSelection attsel = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();

        attsel.setEvaluator(eval);
        attsel.setSearch(search);

        double[][] attrRanks = new double[instances.numAttributes()][2];
        try {
            attsel.SelectAttributes(instances);
            attrRanks = attsel.rankedAttributes();
        } catch (Exception e) {
            e.printStackTrace();
        }
        populateFields(attrRanks);
    }

    private void populateFields(double[][] attrRanks) {
        rankedAttributes = attrRanks;

        for ( double[] rank : attrRanks){
            double featureIndex = rank[0];
            double featureRank = rank[1];
            featureRanksMap.put(featureIndex, featureRank);
        }
    }

    public double getWorstFeatureIndex(){
        return rankedAttributes[rankedAttributes.length-1][0];
    }

    public double getWorstFeatureRank(){
        return rankedAttributes[rankedAttributes.length-1][1];
    }

    public double getBestFeatureIndex(){
        return rankedAttributes[0][0];
    }

    public double getBestFeatureRank(){
        return rankedAttributes[0][1];
    }


}