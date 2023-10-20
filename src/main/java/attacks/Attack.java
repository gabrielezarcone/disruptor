package attacks;

import lombok.Getter;
import lombok.Setter;
import org.w3c.dom.Attr;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;


public abstract class Attack {

    /**
     * Percentage of the Instances the attacker can EDIT
     * @return the capacity of the attack
     */
    @Getter
    private double capacity = 0;

    /**
     * Percentage of the Instances the attacker can VIEW
     * @return the knowledge of the attack
     */
    @Getter
    private double knowledge = 0;

    /**
     * Percentage of the features the attack use
     * @return the horizontal capacity of the attack
     */
    @Getter
    private double featuresCapacity = 0;

    /**
     * instances target of the attack
     * @return instances target of the attack
     */
    @Getter @Setter
    private Instances target;

    @Getter
    private List<Attribute> featureSelected = new ArrayList<>();



    // --------------------------------------------------------------------------------------------------------
    // -- Constructors ----------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------

    protected Attack(Instances target){
        this(target, 1, 1, 1);
    }

    protected Attack(Instances target, double capacity, double featuresCapacity, double knowledge){
        Instances targetCopy = new Instances(target);
        setTarget(targetCopy);
        setCapacity(capacity);
        setFeaturesCapacity(featuresCapacity);
        setKnowledge(knowledge);
    }



    // --------------------------------------------------------------------------------------------------------
    // -- Abstract Methods ----------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------

    /**
     * Perform the attack
     * @return the target instances after the attack
     */
    public abstract Instances attack();



    // --------------------------------------------------------------------------------------------------------
    // -- Methods ----------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------

    /**
     * Number of instances attacked
     * @return Number of target instances specified by the capacity
     */
    public int attackSize(){
        int targetDimension = getTarget().size();
        return (int) (targetDimension*getCapacity());
    }

    /**
     * Number of features attacked
     * @return Number of target features specified by the features capacity
     */
    public int attackedFeaturesSize(){
        int targetFeaturesNumber = getTarget().numAttributes()-1; // -1 because we want all but the class attribute
        return (int) ( targetFeaturesNumber * getFeaturesCapacity() );
    }



    // --------------------------------------------------------------------------------------------------------
    // -- Getters and Setters ----------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------

    /**
     * Capacity is the percentage of the Instances the attacker can EDIT
     * @param capacity of the attack. Should be between 0 and 1
     * @throws IllegalArgumentException if capacity is not between 0 and 1
     */
    public void setCapacity(double capacity) {
        if(capacity>=0 && capacity<=1){
            this.capacity = capacity;
        }
        else {
            throw new IllegalArgumentException("The capacity should be between 0 and 1");
        }
    }

    /**
     * Knowledge is the percentage of the Instances the attacker can VIEW
     * @param knowledge of the attack. Should be between 0 and 1
     * @throws IllegalArgumentException if capacity is not between 0 and 1
     */
    public void setKnowledge(double knowledge) {
        if(knowledge>=0 && knowledge<=1){
            this.knowledge = knowledge;
        }
        else {
            throw new IllegalArgumentException("The knowledge should be between 0 and 1");
        }
    }

    /**
     * Percentage of the features the attack use
     * @param featuresCapacity of the attack. Should be between 0 and 1
     * @throws IllegalArgumentException if featuresCapacity is not between 0 and 1
     */
    public void setFeaturesCapacity(double featuresCapacity) {
        if(featuresCapacity >=0 && featuresCapacity <=1){
            this.featuresCapacity = featuresCapacity;
        }
        else {
            throw new IllegalArgumentException("The featuresCapacity should be between 0 and 1");
        }
    }

    /**
     * Set the selected feature list in the rank corresponding order
     * @param selectedFeaturesRanks the array of attributes in the order defined by the ranks
     */
    public void setFeatureSelected(double[][] selectedFeaturesRanks) {
        for (double[] featureRank : selectedFeaturesRanks) {
            int featureIndex = (int) featureRank[0];
            // Fetch the selected attribute
            Attribute feature = target.attribute(featureIndex);
            featureSelected.add(feature);
        }
    }

    /**
     * The attack must be performed on the percentage of features defined by featuresCapacity.
     * This method, thus, returns a list where are removed the least significant features among the selected features.
     */
    public List<Attribute> getReducedFeatureSelected(){
        ArrayList<Attribute> reducedFeaturesList = new ArrayList<>();
        for (int i=0; i<attackedFeaturesSize(); i++){
            reducedFeaturesList.add(i, getFeatureSelected().get(i));
        }
        return reducedFeaturesList;
    }
}
