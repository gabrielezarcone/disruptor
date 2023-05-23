package util;

import filters.ApplyFilter;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

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

    public static void addAllInstances(Instances destination, Instances source) throws Exception {
        source = sameOrderAttributes(destination,source);
        ArrayList<Instance> sourceInstancesList = Collections.list(source.enumerateInstances());
        destination.addAll(sourceInstancesList);
    }

    public static ArrayList<Attribute> listSortedAttributes(Instances source) {
        ArrayList<Attribute> sourceAttributesList = Collections.list(source.enumerateAttributes());
        sourceAttributesList.sort(new Comparator<Attribute>() {
            @Override
            public int compare(Attribute attribute1, Attribute attribute2) {
                return attribute1.name().compareTo(attribute2.name());
            }
        });
        return sourceAttributesList;
    }

    /**
     * Transform the instancesToChange attributes order accordingly to the order of the template instances.
     * <p/>
     * The two instances should have the same attributes in a different order
     * <p/>
     * @param template instances whose attributes position is used to reorganize the attributes of instancesToChange
     * @param instancesToChange instances whose attributes will be reorganized in the same position of the attributes present in template
     * @return the new Instances object filtered as described
     * @throws IllegalArgumentException if the attributes of the two instances are not the same
     */
    public static Instances sameOrderAttributes(Instances template, Instances instancesToChange) throws Exception {
        boolean isSameNumerOfAttributes = template.numAttributes() == instancesToChange.numAttributes();
        if(isSameNumerOfAttributes){
            boolean isSameAttributes = Collections.list(template.enumerateAttributes()).containsAll(Collections.list(instancesToChange.enumerateAttributes()));
            if(isSameAttributes){
                
                ArrayList<Attribute> templateAttributesList = Collections.list(template.enumerateAttributes());
                templateAttributesList.add(template.classAttribute());
                ArrayList<Attribute> toChangeAttributesList = Collections.list(instancesToChange.enumerateAttributes());
                toChangeAttributesList.add(instancesToChange.classAttribute());
                // Save where the instancesToChange attributes are in the template storing their indices
                ArrayList<Integer> attributesIndicesForFilter = new ArrayList<>();

                for( Attribute templateAttribute : templateAttributesList ){
                    // where the template attribute is inside the instancesToChange
                    int indexOfTemplateAttributeInToChange = toChangeAttributesList.indexOf(templateAttribute);
                    attributesIndicesForFilter.add(indexOfTemplateAttributeInToChange);
                }
                int[] indicesIntArray = attributesIndicesForFilter.stream().mapToInt(i->i).toArray();
                // Apply the filter that reorder the instancesToChange accordingly to the position of the attributes in the template
                instancesToChange = ApplyFilter.reorderAttributes(instancesToChange, indicesIntArray);
                return instancesToChange;
            }
            else {
                throw new IllegalArgumentException("To order the attributes the two instances should have the same attributes");
            }
        }
        else {
            throw new IllegalArgumentException("To order the attributes the two instances should have the same number of attributes");
        }
    }

    /**
     * Replace all the empty values (with name="") present inthe specified instances
     * @param instances
     * @param newValue new value that will replace ""
     */
    public static void replaceEmptyValues(Instances instances, String newValue){
        ArrayList<Attribute> attributeList = Collections.list(instances.enumerateAttributes());
        for(Attribute attribute : attributeList){
            boolean hasEmptyValue = attribute.indexOfValue("") != -1;
            if( hasEmptyValue && (attribute.isNominal() || attribute.isString()) )
                instances.renameAttributeValue(attribute, "", newValue);
        }
    }

    public static HashMap<Object, ArrayList<Instance>> bucketsByClass(Instances perturbedInstances) {
        ArrayList<Object> classValuesList = Collections.list(perturbedInstances.classAttribute().enumerateValues());
        HashMap<Object, ArrayList<Instance>> bucketsMap = new HashMap<>();
        for( Object value : classValuesList){
            bucketsMap.put(value, new ArrayList<>());
        }
        ArrayList<Instance> instancesList = Collections.list(perturbedInstances.enumerateInstances());
        for( Instance instance : instancesList){
            Object classValueObject = InstanceUtil.getClassValueObject(instance);
            bucketsMap.get(classValueObject).add(instance);
        }
        return bucketsMap;
    }

    /**
     * @return the class with the most instances
     */
    public static double getBiggestClass(Instances instances) {
        HashMap<Double, Integer> classMap = classCardinalityMap(instances);
        return Collections.max(classMap.entrySet(), Comparator.comparingInt(Map.Entry::getValue)).getKey();
    }

    /**
     * @param instances instances to evaluate
     * @return a map containing for each class their number of instances
     */
    public static HashMap<Double, Integer> classCardinalityMap(Instances instances) {
        HashMap<Double, Integer> classMap = new HashMap<>();
        ArrayList<Instance> instancesList = Collections.list(instances.enumerateInstances());
        instancesList.forEach( instance -> {
            double classValue = instance.classValue();
            if (classMap.containsKey(classValue)){
                int counter = classMap.get(classValue);
                classMap.put( classValue, counter+1 );
            }
            else {
                classMap.put( classValue, 1 );
            }
        } );
        return classMap;
    }

    /**
     *
     * @param dataset Dataset to split
     * @param trainPercentage Percentage of the dataset to dedicate to the train set.
     * @param toRandomize true if the dataset should be randomized
     * @return The first element of the array is the TRAIN set. The second element of the array is the TEST set
     * @throws Exception if problems applying the RemovePercentage weka filter
     * @throws IllegalArgumentException if trainPercentage is not between 0 and 1
     */
    public static Instances[] splitTrainTest( Instances dataset,  double trainPercentage, boolean toRandomize ) throws Exception {
        return splitTrainTest(dataset, trainPercentage, toRandomize, 1);
    }

    /**
     *
     * @param dataset Dataset to split
     * @param trainPercentage Percentage of the dataset to dedicate to the train set.
     * @param toRandomize true if the dataset should be randomized
     * @param seed seed used to randomize
     * @return The first element of the array is the TRAIN set. The second element of the array is the TEST set
     * @throws Exception if problems applying the RemovePercentage weka filter
     * @throws IllegalArgumentException if trainPercentage is not between 0 and 1
     */
    public static Instances[] splitTrainTest( Instances dataset,  double trainPercentage, boolean toRandomize, int seed ) throws Exception {
        // Check the trainPercentage
        if(trainPercentage<0 || trainPercentage>1){
            throw new IllegalArgumentException("The train percentage should be a double between 0 and 1");
        }

        Instances instances = new Instances(dataset);

        // Save the relation name
        String relationName = dataset.relationName();
        Instances[] result = new Instances[2];

        // Randomize the instances
        if (toRandomize){
            instances = ApplyFilter.randomize(instances, seed);
        }

        // Split train and test
        Instances trainSet = ApplyFilter.removePercentage(instances, trainPercentage*100, true);
        Instances testSet = ApplyFilter.removePercentage(instances, trainPercentage*100, false);

        // Maintain the start relation name even after the filter
        trainSet.setRelationName(relationName);
        testSet.setRelationName(relationName);

        result[0] = trainSet;
        result[1] = testSet;

        return result;
    }
}
