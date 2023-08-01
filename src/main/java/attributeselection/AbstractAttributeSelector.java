package attributeselection;

import lombok.Getter;
import lombok.Setter;
import weka.core.Instances;

import java.util.HashMap;

public abstract class AbstractAttributeSelector {

    @Getter
    @Setter
    private Instances targetInstances;
    @Getter @Setter
    private HashMap<Double, Double> featureRanksMap = new HashMap<>();
    @Getter @Setter
    private double[][] rankedAttributes;
    @Getter @Setter
    private String name = this.getClass().getSimpleName();

    protected AbstractAttributeSelector(Instances instances){
        this.targetInstances = instances;
    }

    /**
     * Perform the attribute selection returning a 2D double array.
     * 1D: ranked attribute indexes.
     * 2D: their associated merit scores as doubles.
     * @return a two dimensional array of ranked attribute indexes and their associated merit scores as doubles
     */
    protected abstract double[][] selectAttributes();


    /**
     * Start the feature selection
     */
    public void eval() {
        double[][] attrRanks = selectAttributes();
        populateFields(attrRanks);
    }

    /**
     * Populate rankedAttribute and featureRanksMap using the attrRanks param
     * @param attrRanks
     */
    protected void populateFields(double[][] attrRanks) {
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
