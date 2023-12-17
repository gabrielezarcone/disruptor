package disruptor.attacks.vertical;

import lombok.Getter;
import lombok.Setter;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class MiddlePoint extends VerticalAttack {

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
    protected void verticalDisrupt(Instance instanceToAttack, Attribute featureToAttack) {
        double oldValue = instanceToAttack.value(featureToAttack);
        double distanceFromMiddle = featureMiddlePoint(featureToAttack) - oldValue;
        double newValue = getMultiplicationFactor() * distanceFromMiddle + oldValue;
        instanceToAttack.setValue(featureToAttack, newValue);
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
