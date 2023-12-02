package attacks.vertical;

import attacks.Attack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.stream.IntStream;

public abstract class VerticalAttack extends Attack {
    protected VerticalAttack(Instances target) {
        super(target);
    }

    protected VerticalAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {
        Instances perturbedInstances = new Instances(getTarget());

        disruptDataset(perturbedInstances);

        // Perform the attack only in the part of the target specified by the capacity
        IntStream.range(0, attackSize()).parallel().forEach(i -> {
            Instance instanceToAttack = perturbedInstances.instance(i);

            disruptInstance(perturbedInstances, instanceToAttack);

            // Perform the attack only for the selected feature
            for( Attribute feature : getReducedFeatureSelected() ){
                disruptFeature(instanceToAttack, feature);
            }

            perturbedInstances.set(i, instanceToAttack);
        });
        return perturbedInstances;
    }

    /**
     * Disruption.
     * Use this method to disrupt the entire dataset or to use information contained in the dataset
     * to perform the disruption of the instance or of the feature
     *
     * @param datasetToAttack dataset to attack
     */
    protected abstract void disruptDataset(Instances datasetToAttack);

    /**
     * Horizontal disruption.
     * Use this method to disrupt the entire instance or to use information contained in the instance
     * to perform the disruption of the feature.
     *
     * @param datasetToAttack dataset to attack
     * @param instanceToAttack instance selected for the disruption
     */
    protected abstract void disruptInstance(Instances datasetToAttack, Instance instanceToAttack) ;

    /**
     * Vertical disruption.
     * Use this method to disrupt the selected instance at the selected attribute.
     *
     * @param instanceToAttack instance selected for the disruption
     * @param featureToAttack feature selected for the disruption
     */
    protected abstract void disruptFeature(Instance instanceToAttack, Attribute featureToAttack) ;

}
