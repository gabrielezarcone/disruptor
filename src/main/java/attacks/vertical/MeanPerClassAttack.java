package attacks.vertical;

import attacks.Attack;
import util.InstancesUtil;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class MeanPerClassAttack extends Attack {

    public MeanPerClassAttack(Instances target) {
        super(target);
    }

    public MeanPerClassAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());

        // create buckets of instances grouped by class value
        HashMap<Object, Instances> bucketsMap = InstancesUtil.bucketsByClassInstances(perturbedInstances);

        // Perform the attack only in the part of the target specified by the capacity
        for(int i=0; i<attackSize(); i++){
            Instance instanceToAttack = perturbedInstances.instance(i);

            // Calculate the next class value
            double classValue = instanceToAttack.classValue();
            double nextClassValueIndex = ( classValue + 1 ) % perturbedInstances.numClasses();
            // Fetch the class obj to perform get from the bucket map
            ArrayList<Object> classValuesList = Collections.list(perturbedInstances.classAttribute().enumerateValues());
            Object nextClassValue = classValuesList.get((int) nextClassValueIndex);

            Instances instancesPerClass = bucketsMap.get(nextClassValue);

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){
                // Select the mean or mode value of all the instances that have the next class value
                double meanOrMode = instancesPerClass.meanOrMode(feature);
                instanceToAttack.setValue(feature, meanOrMode);
            }

            perturbedInstances.set(i, instanceToAttack);
        }
        return perturbedInstances;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }
}
