package attacks.vertical;

import attacks.Attack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class MeanAttack extends Attack {

    public MeanAttack(Instances target) {
        super(target);
    }

    public MeanAttack(Instances target, double capacity, double knowledge) {
        super(target, capacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());
        // Perform the attack only in the part of the target specified by the capacity
        for(int i=0; i<attackSize(); i++){
            Instance instanceToAttack = perturbedInstances.instance(i);

            // Perform the attack only for the selected feature
            for( Attribute feature : getFeatureSelected() ){
                double meanOrMode = perturbedInstances.meanOrMode(feature);
                instanceToAttack.setValue(feature, meanOrMode);
            }

            perturbedInstances.set(i, instanceToAttack);
        }
        return perturbedInstances;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }
}