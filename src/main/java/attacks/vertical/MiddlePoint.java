package attacks.vertical;

import attacks.Attack;
import lombok.Getter;
import lombok.Setter;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class MiddlePoint extends Attack {

    /**
     * Multiplication factor used for moving the instances towards the middle point
     * <p></p>
     * es: instance is 3, the multiplication factor is 0.5 and the middle point is 5, then the new
     * value is ((5-3)*0.5)+3 = 4 -> the instances is moved towards the middle point by the 50%
     */
    @Setter @Getter
    private double multiplicationFactor = 0.5;

    public MiddlePoint(Instances target) {
        super(target);
    }

    public MiddlePoint(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());
        // Perform the attack only in the part of the target specified by the capacity
        for(int i=0; i<attackSize(); i++){
            Instance instanceToAttack = perturbedInstances.instance(i);

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){
                double oldValue = instanceToAttack.value(feature);
                double distanceFromMiddle = featureMiddlePoint(feature) - oldValue;
                double newValue = getMultiplicationFactor() * distanceFromMiddle + oldValue;
                instanceToAttack.setValue(feature, newValue);
            }

            perturbedInstances.set(i, instanceToAttack);
        }
        return perturbedInstances;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }

    private double featureMiddlePoint(Attribute feature){
        double sumValues = 0;
        int instancesNumber = getTarget().size();
        for (int i = 0; i< instancesNumber; i++){
            Instance instance = getTarget().instance(i);
            double featureValue = instance.value(feature);
            // TODO verificare se il valore è null cosa viene messo in featureValue e nel caso non considerare i null
            sumValues += featureValue;
        }
        return sumValues / instancesNumber;
    }
}
