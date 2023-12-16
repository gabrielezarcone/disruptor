package disruptor.attacks.custom;

import weka.core.Instance;
import weka.core.Instances;

/**
 * As the attack of the Side by Side every instance is put on the side of one of the reference class, but in this case
 * every instance is put in the same position all on top of each other
 */
public class SideBySideOnTop extends SideBySide{
    public SideBySideOnTop(Instances target, double featureSet) {
        super(target, featureSet);
    }

    public SideBySideOnTop(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    protected double getAdding(Instance instance) {
        return 0.001;
    }
}
