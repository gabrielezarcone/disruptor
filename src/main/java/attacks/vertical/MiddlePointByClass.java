package attacks.vertical;

import attacks.Attack;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import util.InstancesUtil;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.stream.IntStream;

@Slf4j
public class MiddlePointByClass extends Attack {

    /**
     * Multiplication factor used for moving the instances towards the middle point
     * <p></p>
     * es: instance is 3, the multiplication factor is 0.5 and the middle point is 5, then the new
     * value is ((5-3)*0.5)+3 = 4 -> the instances is moved towards the middle point by the 50%
     */
    @Setter @Getter
    private double multiplicationFactor = 0.5;

    public MiddlePointByClass(Instances target) {
        super(target);
    }

    public MiddlePointByClass(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());
        int numClasses = perturbedInstances.numClasses();
        ArrayList<Object> classValuesList = Collections.list(perturbedInstances.classAttribute().enumerateValues());

        // Perform the attack only in the part of the target specified by the capacity
        IntStream.range(0, attackSize()).parallel().forEach(i -> {
            Instance instanceToAttack = perturbedInstances.instance(i);

            // Calculate the next class value
            double classValue = instanceToAttack.classValue();
            double nextClassValueIndex = ( classValue + 1 ) % numClasses;
            // Fetch the class obj to perform get from the bucket map
            Object nextClassValue = classValuesList.get((int) nextClassValueIndex);

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){
                double oldValue = instanceToAttack.value(feature);
                double distanceFromMiddle = featureMiddlePoint(feature, nextClassValue) - oldValue;
                double newValue = getMultiplicationFactor() * distanceFromMiddle + oldValue;
                instanceToAttack.setValue(feature, newValue);
            }

            perturbedInstances.set(i, instanceToAttack);
        });
        return perturbedInstances;
    }

    private double featureMiddlePoint(Attribute feature, Object classValue){

        // create buckets of instances grouped by class value
        HashMap<Object, Instances> bucketsMap = InstancesUtil.bucketsByClassInstances(getTarget());
        Instances instancesPerClass = bucketsMap.get(classValue);

        double sumValues = 0;
        int instancesNumber = instancesPerClass.size();
        for (int i = 0; i< instancesNumber; i++){
            Instance instance = instancesPerClass.instance(i);
            double featureValue = instance.value(feature);
            // TODO verificare se il valore è null cosa viene messo in featureValue e nel caso non considerare i null
            sumValues += featureValue;
        }
        return sumValues / instancesNumber;
    }
}
