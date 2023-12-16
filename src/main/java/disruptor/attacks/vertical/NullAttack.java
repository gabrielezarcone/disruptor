package disruptor.attacks.vertical;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class NullAttack extends VerticalAttack {

    public NullAttack(Instances target) {
        super(target);
    }

    public NullAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    protected void verticalDisrupt(Instance instanceToAttack, Attribute featureToAttack) {
        instanceToAttack.setValue(featureToAttack, 0);
    }

}
