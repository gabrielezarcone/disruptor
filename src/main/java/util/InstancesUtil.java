package util;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

public class InstancesUtil {

    /**
     * @param instances original Instances
     * @param attributeName name of the attribute to mantain
     * @return new Instances with only the attribute with the specified name
     */
    public static Instances deleteAllAttributesButOne(Instances instances, String attributeName){
        Instances result = new Instances(instances);
        Enumeration<Attribute> attributesEnumeration = result.enumerateAttributes();
        while (attributesEnumeration.hasMoreElements()) {
            Attribute currentAttribute = attributesEnumeration.nextElement();
            String currentName = currentAttribute.name();
            if (!currentName.equals(attributeName)) {
                int newAttributeIndex = result.attribute(currentName).index();
                result.deleteAttributeAt(newAttributeIndex);
            }
        }
        return result;
    }

    public static void addAllInstances(Instances destination, Instances source) {
        ArrayList<Instance> sourceInstancesList = Collections.list(source.enumerateInstances());
        destination.addAll(sourceInstancesList);
    }
}
