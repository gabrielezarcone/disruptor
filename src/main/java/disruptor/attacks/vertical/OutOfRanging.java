package disruptor.attacks.vertical;

import disruptor.attacks.Attack;
import disruptor.util.InstancesUtil;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
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

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){

                //Create a set that contains the unique values assumed by the feature with the selected class
                Set<Double> range = new HashSet<>();
                Collections.list( instancesPerClass.enumerateInstances()).forEach( in -> range.add(in.value(feature)));

                // Get a random value out of the range of the same class
                double randomValueOutOfRange;
                do {
                    randomValueOutOfRange = random.nextDouble();
                }while ( range.contains(randomValueOutOfRange));

                instanceToAttack.setValue(feature, randomValueOutOfRange);

            }

            perturbedInstances.set(i, instanceToAttack);
        });
        return perturbedInstances;
    }

}
