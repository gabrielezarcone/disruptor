package disruptor.attacks.horizontal.labelflipping;

import disruptor.attacks.horizontal.HorizontalAttack;
import lombok.extern.slf4j.Slf4j;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A standard Label flipping attack that cycles the class values:
 * A → B
 * B → C
 * C → A
 */
@Slf4j
public class LabelFlipping extends HorizontalAttack {

    public LabelFlipping(Instances target) {
        super(target);
    }

    protected LabelFlipping(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    protected void horizontalDisrupt(Instance instance) {
        int classValuesNumber = instance.numClasses();
        double newClassValue = ( instance.classValue() + 1 ) % classValuesNumber;

        if(newClassValue >= classValuesNumber){
            log.error("Invalid class value: " + newClassValue);
        }

        instance.setClassValue(newClassValue);
    }
}
