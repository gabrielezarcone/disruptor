package attacks.vertical;

import attacks.Attack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.stream.IntStream;

public class NullAttack extends VerticalAttack {

    public NullAttack(Instances target) {
        super(target);
    }

    public NullAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    protected void disruptDataset(Instances datasetToAttack) {
        // Do nothing
    }

    @Override
    protected void disruptInstance(Instances datasetToAttack, Instance instanceToAttack) {
        // Do nothing
    }

    @Override
    protected void disruptFeature(Instance instanceToAttack, Attribute featureToAttack) {
        instanceToAttack.setValue(featureToAttack, 0);
    }

}
