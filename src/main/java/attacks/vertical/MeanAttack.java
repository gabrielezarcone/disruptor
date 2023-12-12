package attacks.vertical;

import attacks.Attack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.stream.IntStream;

public class MeanAttack extends VerticalAttack {

    public MeanAttack(Instances target) {
        super(target);
    }

    public MeanAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    protected void verticalDisrupt(Instance instanceToAttack, Attribute featureToAttack) {
        double meanOrMode = getTarget().meanOrMode(featureToAttack);
        instanceToAttack.setValue(featureToAttack, meanOrMode);
    }
}
