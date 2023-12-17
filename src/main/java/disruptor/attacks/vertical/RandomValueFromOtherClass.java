package disruptor.attacks.vertical;

import disruptor.attacks.Attack;
import disruptor.util.InstancesUtil;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.stream.IntStream;

public class RandomValueFromOtherClass extends Attack {

    public RandomValueFromOtherClass(Instances target) {
        super(target);
    }

    public RandomValueFromOtherClass(Instances target, double capacity, double featuresCapacity, double knowledge) {
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
            double classValue = instanceToAttack.classValue();
            double nextClassValueIndex = ( classValue + 1 ) % perturbedInstances.numClasses();
            // Fetch the class obj to perform get from the bucket map
            ArrayList<Object> classValuesList = Collections.list(perturbedInstances.classAttribute().enumerateValues());
            Object nextClassValue = classValuesList.get((int) nextClassValueIndex);

            Instances instancesPerClass = bucketsMap.get(nextClassValue);
            int instancesLength = instancesPerClass.size();

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){

                Instance randomInstanceFromOtherClass = instancesPerClass.get( random.nextInt(instancesLength) );
                double randomValueFromOtherClass = randomInstanceFromOtherClass.value(feature);
                instanceToAttack.setValue(feature, randomValueFromOtherClass);

            }

            perturbedInstances.set(i, instanceToAttack);
        });
        return perturbedInstances;
    }

}
