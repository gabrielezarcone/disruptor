package disruptor.attacks.horizontal;

import disruptor.attacks.Attack;
import weka.core.Instance;
import weka.core.Instances;

import java.util.stream.IntStream;

public abstract class HorizontalAttack extends Attack {
    protected HorizontalAttack(Instances target) {
        super(target);
    }

    protected HorizontalAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());

        // Perform the attack only in the part of the target specified by the capacity
        IntStream.range(0, attackSize()).parallel().forEach(i -> {
            Instance instanceToAttack = perturbedInstances.instance(i);

            horizontalDisrupt(instanceToAttack);

            perturbedInstances.set(i, instanceToAttack);
        });
        return perturbedInstances;
    }

    /**
     * Horizontal disruption.
     * Use this method to disrupt the selected instance
     *
     * @param instanceToAttack instance selected for the disruption
     */
    protected abstract void horizontalDisrupt(Instance instanceToAttack) ;
}
