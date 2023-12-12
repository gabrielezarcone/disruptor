package attacks.vertical;

import attacks.Attack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.stream.IntStream;

public abstract class VerticalAttack extends Attack {
    protected VerticalAttack(Instances target) {
        super(target);
    }

    protected VerticalAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
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
                verticalDisrupt(instanceToAttack, feature);
            }

            perturbedInstances.set(i, instanceToAttack);
        });
        return perturbedInstances;
    }

    /**
     * Vertical disruption.
     * Use this method to disrupt the selected instance at the selected attribute.
     *
     * @param instanceToAttack instance selected for the disruption
     * @param featureToAttack feature selected for the disruption
     */
    protected abstract void verticalDisrupt(Instance instanceToAttack, Attribute featureToAttack) ;

}
