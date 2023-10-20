package attacks.vertical;

import attacks.Attack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.stream.IntStream;

public class NullAttack extends Attack {

    public NullAttack(Instances target) {
        super(target);
    }

    public NullAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());
        // Perform the attack only in the part of the target specified by the capacity
        IntStream.range(0, attackSize()).parallel().forEach(i -> {
            Instance instanceToAttack = perturbedInstances.instance(i);

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){
                instanceToAttack.setValue(feature, 0);
            }

            perturbedInstances.set(i, instanceToAttack);
        });
        return perturbedInstances;
    }

}
