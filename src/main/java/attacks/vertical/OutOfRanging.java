package attacks.vertical;

import attacks.Attack;
import util.InstancesUtil;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.stream.IntStream;

public class OutOfRanging extends Attack {
    public OutOfRanging(Instances target) {
        super(target);
    }

    public OutOfRanging(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    private final Random random = new Random();

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());

        // create buckets of instances grouped by class value
        HashMap<Object, Instances> bucketsMap = InstancesUtil.bucketsByClassInstances(perturbedInstances);

        // Perform the attack only in the part of the target specified by the capacity
        IntStream.range(0, attackSize()).parallel().forEach(i -> {
            Instance instanceToAttack = perturbedInstances.instance(i);

            // Calculate the next class value
            double classValueIndex = instanceToAttack.classValue();
            // Fetch the class obj to perform get from the bucket map
            ArrayList<Object> classValuesList = Collections.list(perturbedInstances.classAttribute().enumerateValues());
            Object classValue = classValuesList.get((int) classValueIndex);

            Instances instancesPerClass = bucketsMap.get(classValue);
            int instancesLength = instancesPerClass.size();

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){

                // Get a random value from the same class
                Instance randomInstanceFromSameClass = instancesPerClass.get( random.nextInt(instancesLength) );
                double randomValueFromSameClass = randomInstanceFromSameClass.value(feature);
                instanceToAttack.setValue(feature, randomValueFromSameClass);

            }

            perturbedInstances.set(i, instanceToAttack);
        });
        return perturbedInstances;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }
}
