package attacks.horizontal;

import attacks.Attack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class OppositeAttack extends Attack {

    public OppositeAttack(Instances target) {
        super(target);
    }

    public OppositeAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {

        Instances perturbedInstances = new Instances(getTarget());
        // Perform the attack only in the part of the target specified by the capacity
        for(int i=0; i<attackSize(); i++){

            Instance instanceToAttack = perturbedInstances.instance(i);

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){
                double actualValue = instanceToAttack.value( feature );
                instanceToAttack.setValue( feature, actualValue*(-1) );
            }

            perturbedInstances.set(i, instanceToAttack);
        }
        return perturbedInstances;

    }
}
