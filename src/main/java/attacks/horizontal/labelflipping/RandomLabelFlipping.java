package attacks.horizontal.labelflipping;

import attacks.Attack;
import attacks.horizontal.HorizontalAttack;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.stream.IntStream;

public class RandomLabelFlipping extends HorizontalAttack {

    Attribute classAttribute;

    public RandomLabelFlipping(Instances target){
        super(target);
        classAttribute =  getTarget().classAttribute();
    }

    public RandomLabelFlipping(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
        classAttribute =  getTarget().classAttribute();
    }

    @Override
    protected void horizontalDisrupt(Instance flippedInstance) {
        double flippedClass = newClassValue(classAttribute, flippedInstance);
        flippedInstance.setClassValue(flippedClass);
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


}
