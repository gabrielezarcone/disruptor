package attacks.horizontal.labelflipping;

import attacks.Attack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class RandomLabelFlipping extends Attack {

    public RandomLabelFlipping(Instances target){
        super(target);
    }

    public RandomLabelFlipping(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }


    @Override
    public Instances attack() {
        Attribute classAttribute =  getTarget().classAttribute();

        Instances perturbedInstances = new Instances(getTarget());
        // Perform the attack only in the part of the target specified by the capacity
        for(int i=0; i<attackSize(); i++){
            Instance flippedInstance = perturbedInstances.instance(i);
            double flippedClass = newClassValue(classAttribute, flippedInstance);
            flippedInstance.setClassValue(flippedClass);

            perturbedInstances.set(i, flippedInstance);
        }
        return perturbedInstances;
    }

    /**
     * Calculate a random double in the values range but different from the current class value
     * @param classAttribute class attribute used to fetch all the possible values
     * @param instance the double returned will be different from the one corresponding to this instance class value
     * @return a random double in the classAttribute values range but different from the instance class value
     */
    private double newClassValue(Attribute classAttribute, Instance instance) {
        ArrayList<Object> classValues = Collections.list(classAttribute.enumerateValues());
        double currentClassValue = instance.classValue();

        // calculate a random double in the values range but different from the current class value
        double newRandomClass = currentClassValue;
        Random random = new Random();
        while (newRandomClass==currentClassValue){
            newRandomClass = random.nextInt(classValues.size());
        }

        return newRandomClass;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }


}
