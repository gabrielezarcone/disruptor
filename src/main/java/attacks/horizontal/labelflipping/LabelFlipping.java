package attacks.horizontal.labelflipping;

import attacks.Attack;
import lombok.extern.slf4j.Slf4j;
import util.InstanceUtil;
import util.InstancesUtil;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.stream.IntStream;

/**
 * A standard Label flipping attack that cycles the class values:
 * A → B
 * B → C
 * C → A
 */
@Slf4j
public class LabelFlipping extends Attack{

    public LabelFlipping(Instances target) {
        super(target);
    }

    protected LabelFlipping(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());

        IntStream.range(0, attackSize()).parallel().forEach(i -> {
            Instance instance = perturbedInstances.instance(i);
            int classValuesNumber = instance.numClasses();
            double newClassValue = ( instance.classValue() + 1 ) % classValuesNumber;

            if(newClassValue >= classValuesNumber){
                log.error("Invalid class value: " + newClassValue);
            }

            instance.setClassValue(newClassValue);
            perturbedInstances.set(i, instance);
        });

        return perturbedInstances;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }
}
