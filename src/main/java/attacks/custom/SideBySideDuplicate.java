package attacks.custom;

import attacks.Attack;
import lombok.extern.slf4j.Slf4j;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Instead of putting side by side of the instance one other instance of another class duplicate instead the instance and do a label flipping of it
 * Create a duplicate for every other class different from the one of the instance selected
 */
@Slf4j
public class SideBySideDuplicate extends Attack {

    public SideBySideDuplicate(Instances target) {
        super(target);
    }

    protected SideBySideDuplicate(Instances target, double capacity, double knowledge) {
        super(target, capacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());

        for(int i=0; i<attackSize(); i++){
            Instance instance = perturbedInstances.instance(i);
            int classValuesNumber = instance.numClasses();
            double newClassValue = ( instance.classValue() + 1 ) % classValuesNumber;

            if(newClassValue >= classValuesNumber){
                log.error("Invalid class value: " + newClassValue);
            }

            Instance newInstance = new DenseInstance(instance);
            newInstance.setDataset(perturbedInstances);
            newInstance.setClassValue(newClassValue);

            perturbedInstances.add(newInstance);
        }

        return perturbedInstances;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }
}









