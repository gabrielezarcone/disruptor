package util;

import costants.FilePaths;
import filters.ApplyFilter;
import lombok.NonNull;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

public class ArffUtil {

    /**
     * Return a {@link Instances} object from a .arff file
     * @param filePath file path of the .arff source file
     * @return {@link Instances} object based on an .arff file
     * @throws IOException if problems fetching the file
     */
    private static Instances readArffFile(String filePath) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath));

        Instances dataset = new Instances(bufferedReader);
        bufferedReader.close();
        return dataset;
    }

    /**
     * Return a {@link Instances} object from a .arff file setting the class attribute as specified
     * @param filePath file path of the .arff source file
     * @param className name of the attribute to be setted as the class attribute
     * @return {@link Instances} object based on an .arff file
     * @throws IOException if problems fetching the file
     */
    public static Instances readArffFile(String filePath, String className) throws IOException {

        Instances dataset = readArffFile(filePath);
        Attribute classAttribute = dataset.attribute(className);
        dataset.setClass(classAttribute);


        return dataset;
    }

    /**
     *
     * @param filePath1
     * @param filePath2
     * @return
     * @throws Exception
     */
    public static Instances mergeArffFiles(String filePath1, String filePath2) throws Exception{
        // read the two files
        @NonNull
        Instances instances1 = readArffFile(filePath1);
        @NonNull
        Instances instantces2 = readArffFile(filePath2);

        return mergeInstances(instances1, instantces2);
    }

    /**
     *
     * @param instances1
     * @param instances2
     * @return
     * @throws Exception
     */
    public static Instances mergeInstances(Instances instances1, Instances instances2) throws Exception {

        // save the Instances object with the bigger number of instances in biggerInstances
        Instances biggerInstances = instances1;
        Instances smallerInstances = instances2;
        if( biggerInstances.numInstances() < smallerInstances.numInstances() ){
            Instances temp = biggerInstances;
            biggerInstances = smallerInstances;
            smallerInstances = temp;
        }

        // save relation name before the elaboration
        String biggerInstancesRelationName = biggerInstances.relationName();
        String smallerInstancesRelationName = smallerInstances.relationName();

        // align the attributes of the two Instances object
        biggerInstances = addAllMissingAttributes(biggerInstances, smallerInstances);
        smallerInstances = addAllMissingAttributes(smallerInstances, biggerInstances);

        // add all instances of smallerInstances into biggerInstances
        Instances mergedInstances = biggerInstances;
        InstancesUtil.addAllInstances(mergedInstances, smallerInstances);

        // set the relation name of the merged file
        mergedInstances.setRelationName(biggerInstancesRelationName+"-"+smallerInstancesRelationName);

        return mergedInstances;
    }

    /**
     * Add to destinationInstances all attributes of sourceInsances that are not present in destinationInstances.
     * For nominal attributes of the same name the values are merged.
     * <p/>
     * @param destinationInstances
     * @param sourceInsances
     * @return a new version of destinationInstances with its own attributes plus all the sourceInsances attributes that were not present at first.
     * @throws Exception if problems with the merging of nominal attributes values
     */
    private static Instances addAllMissingAttributes(Instances destinationInstances, Instances sourceInsances) throws Exception {
        Enumeration<Attribute> sourceAttributesEnumeration = sourceInsances.enumerateAttributes();
        while(sourceAttributesEnumeration.hasMoreElements()){
            Attribute currentSourceAttribute = sourceAttributesEnumeration.nextElement();
            Attribute currentDestinationAttribute = destinationInstances.attribute(currentSourceAttribute.name());

            if(currentDestinationAttribute==null){
                // if currentDestinationAttribute IS null it means that currentSourceAttribute IS NOT present in destinationInstances
                destinationInstances.insertAttributeAt(currentSourceAttribute, destinationInstances.numAttributes());
            }
            else {
                // if currentDestinationAttribute IS NOT null it means that currentSourceAttribute IS present in destinationInstances
                boolean isSameAttributeName = currentSourceAttribute.name().equals(currentDestinationAttribute.name());
                boolean isSameAttributeType = currentSourceAttribute.type() == currentDestinationAttribute.type();

                if(isSameAttributeName && isSameAttributeType){
                    if( currentSourceAttribute.type()==Attribute.NOMINAL && currentSourceAttribute.numValues() > 0){
                        // add labels in attribute
                        destinationInstances = ApplyFilter.addValues(destinationInstances, currentSourceAttribute, currentDestinationAttribute);
                    }
                }
            }

        }
        return destinationInstances;
    }

    /**
     * Export the {@link Instances} object specified in instances as a .arff file.
     * <p/>
     * The output destination is specified by the constant {@link FilePaths}.OUTPUT_FOLDER
     * <p/>
     * @param instances {@link Instances} object to export
     * @param outputFilename filename of the exported file
     * @throws IOException if problem during the export
     */
    public static void exportToArff(Instances instances, String outputFilename) throws IOException {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        saver.setFile(new File( FilePaths.OUTPUT_FOLDER + outputFilename + ".arff"));
        saver.writeBatch();
    }

}
