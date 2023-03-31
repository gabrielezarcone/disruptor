package util;

import weka.core.Attribute;

import java.util.ArrayList;
import java.util.Collections;
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

    public static Attribute addValue(Attribute attribute, Object value){
        ArrayList<Object> valuesList = Collections.list(attribute.enumerateValues());
        valuesList.add(value);

        ArrayList<String> valuesNamesList = new ArrayList<>();
        for(Object valueObj : valuesList){
            valuesNamesList.add(valueObj.toString());
        }

        //TODO non va bene così perchè il new sostituisce l'attributo, ma cancella anche tutti i valori contenuti nelle instances
        return  new Attribute(attribute.name(), valuesNamesList );
    }
}
