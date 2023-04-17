package attacks.custom;

import attacks.Attack;
import attributeselection.InfoGainEval;
import filters.ApplyClassBalancer;
import lombok.Getter;
import lombok.Setter;
import util.InstanceUtil;
import util.InstancesUtil;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class SideBySide extends Attack {

    /**
     * Class to use as reference. The instances with this class will not be changed.
     * The other instances will be positioned near the instances of this class
     * @param referenceClass double corresponding to the class to use as reference
     * @return the reference class
     */
    @Getter @Setter
    protected double referenceClass = 0.0;

    protected SideBySide(Instances target) {
        super(target);
    }

    protected SideBySide(Instances target, double capacity, double knowledge) {
        super(target, capacity, knowledge);
    }


    @Override
    public Instances attack() {
        // TODO bilancia le classi oppure parti dalla classe più numerosa
        Instances perturbedInstances = new Instances(getTarget());
        try {

            // balance the classes
            perturbedInstances = ApplyClassBalancer.classBalancer(getTarget());


            // create buckets of instances grouped by class value
            HashMap<Object, ArrayList<Instance>> bucketsMap = InstancesUtil.bucketsByClass(perturbedInstances);


            // Use as reference feature the feature selected by the feature selection algorithm
            InfoGainEval infoGainEval = new InfoGainEval(perturbedInstances);
            infoGainEval.eval();
            int worstAttributeIndex = (int) infoGainEval.getWorstFeatureIndex();
            Attribute worstAttribute = perturbedInstances.attribute( worstAttributeIndex );


            // get the bucket corresponding to the reference class
            ArrayList<Instance> referenceBucketList = bucketsMap.get(getReferenceClassObject());
            ArrayList<Instance> perturbedList = new ArrayList<>();


            // Cycle the buckets
            for (Map.Entry<Object,ArrayList<Instance>> bucketsMapEntry : bucketsMap.entrySet()){

                ArrayList<Instance> bucketList = bucketsMapEntry.getValue();
                // Do not cycle the bucket corresponding to  the reference class
                if( !bucketList.equals(referenceBucketList )){
                    //Cycle on the instances
                    // Perform the attack only in the part of the target instances specified by the capacity
                    int attackSize = attackSize()/perturbedInstances.classAttribute().numValues();
                    for(int i = 0; i< attackSize; i++){

                        Instance instance = bucketList.get( i );
                        Instance referenceInstance = referenceBucketList.get( i );

                        attackInstance(instance, referenceInstance, worstAttribute );

                        bucketList.set(i, instance);
                    }
                }
                // append all the bucketList together
                perturbedList.addAll(bucketList);
            }

            // Save perturbed instances in an Instances object
            for( int i=0; i<perturbedList.size(); i++){
                Instance instance = perturbedList.get(i);
                perturbedInstances.set(i, instance);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return perturbedInstances;
    }



    /**
     * Perturb the single instance with the SideBySide attack.
     * <p/>
     * The value of the features of the instance is replaced by the value of the respective feature in the reference instance
     * added with the double value of the class divided with 1000. This is done for every feature but the class feature and the
     * one defined in referenceAttribute.
     * @param instance instance to perturb
     * @param referenceInstance instance used as reference to copy the features valud.
     * @param referenceAttribute the only feature with the class feature that is not perturbed by the attack
     */
    private void attackInstance( Instance instance, Instance referenceInstance, Attribute referenceAttribute ) {
        // Cycle on the features
        for(int j=0; j<featuresSetSize(); j++){

            Attribute attribute = instance.attribute( j );

            boolean isBestAttribute = attribute.equals(referenceAttribute);
            boolean isClassAttribute = attribute.equals(instance.classAttribute());
            if( !isBestAttribute && !isClassAttribute){
                // if is not the class feature
                // and
                // if is not the reference feature add a value dependent from the class value
                double value = referenceInstance.value(attribute);
                double adding = getAdding(instance);
                instance.setValue( attribute, value+adding );
            }
        }
    }

    /**
     * Override this method to change the logic of the attack
     * @param instance
     * @return the value added to every instance perturbed
     */
    protected double getAdding(Instance instance) {
        double instanceClass = instance.classValue();
        return instanceClass / 1000;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }

    public Object getReferenceClassObject() {
        return InstanceUtil.getClassValueObject(getTarget().firstInstance(), referenceClass);
    }
}
