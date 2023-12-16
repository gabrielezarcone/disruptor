package disruptor.util;

import weka.core.Attribute;

import java.util.Enumeration;

public class AttributeUtil {

    public static String attributeValuesToString(Attribute attribute) {
        StringBuilder sb = new StringBuilder();
        Enumeration<Object> valuesEnumeration = attribute.enumerateValues();

        while (valuesEnumeration.hasMoreElements()){
            Object currentValue = valuesEnumeration.nextElement();
            sb.append(currentValue);
            sb.append(",");
        }
        sb.deleteCharAt(sb.lastIndexOf(","));

        return sb.toString();
    }
}
