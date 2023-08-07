package attributeselection;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.Arrays;
import java.util.HashMap;


@Slf4j
public abstract class AbstractAttributeSelector {

    @Setter
    private Instances targetInstances;
    @Getter @Setter
    private HashMap<Double, Double> featureRanksMap = new HashMap<>();
    @Getter @Setter
    private double[][] rankedAttributes;
    @Getter @Setter
    private String name = this.getClass().getSimpleName();
    @Getter @Setter
    private double knowledge = 1;

    protected AbstractAttributeSelector(Instances instances){
        this.targetInstances = new Instances(instances);
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
        reduceAttributesByKnowledge();
        double[][] attrRanks = selectAttributes();
        log.debug("Selected features: {}", Arrays.deepToString(attrRanks));
        populateFields(attrRanks);
    }

    /**
     * Feature selection must be performed only on the percentage of attributes defined by the knowledge attribute.
     * This method, thus, removes attributes from targetInstances in order to match the knowledge before the feature selection.
     */
    protected void reduceAttributesByKnowledge() {
        int numAttributes = targetInstances.numAttributes();
        int numAttributesToConsider = (int) (numAttributes * knowledge);
        int numAttributesToDelete = numAttributes - numAttributesToConsider - 1; //-1 to not count the class feature
        for (int i=0; i<numAttributesToDelete; i++){
            Attribute currentAttribute = targetInstances.attribute(i);
            if(currentAttribute != targetInstances.classAttribute()){
                targetInstances.deleteAttributeAt(i);
            }
        }
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
