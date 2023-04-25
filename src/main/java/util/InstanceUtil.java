package util;

import weka.core.Instance;

import java.util.ArrayList;
import java.util.Collections;

public class InstanceUtil {
    private InstanceUtil(){}

    /**
     * The classValue method of Instance object returns only the double value. This method returns the corresponding class Object
     * @param instance instance defined
     * @return the Object class of the defined instance instead of the double value
     */
    public static Object getClassValueObject(Instance instance){
        double classValue = instance.classValue();
        return getClassValueObject(instance, classValue);
    }

    /**
     * The classValue method of Instance object returns only the double value. This method returns the corresponding class Object
     * @param instance instance used to get the class values
     * @return the Object class corresponding to the classValue specified
     */
    public static Object getClassValueObject(Instance instance, double classValue){
        ArrayList<Object> classValuesList = Collections.list(instance.classAttribute().enumerateValues());
        return classValuesList.get((int) classValue);
    }
}
