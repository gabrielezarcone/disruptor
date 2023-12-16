package disruptor.util;

import disruptor.filters.ApplyFilter;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;
import java.util.Collections;

public class CSVUtil {
    private CSVUtil(){}

    /**
     * Return a {@link Instances} object from a CSV file
     * @param file The CSV source file
     * @return {@link Instances} object based on the CSV file
     * @throws IOException if problems fetching the file
     */
    public static Instances readCSVFile(File file) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(file);
        return loader.getDataSet();
    }

    /**
     * Return a {@link Instances} object from a CSV file and set the class attribute
     * @param file The CSV source file
     * @param className Name of the attribute to be setted as the class attribute
     * @return {@link Instances} object based on the CSV file
     * @throws IOException if problems fetching the file
     */
    public static Instances readCSVFile(File file, String className) throws Exception {
        Instances instances = readCSVFile(file);
        Attribute classAttribute = instances.attribute(className);

        if( classAttribute.isNumeric() ){
            instances = ApplyFilter.numericToNominal(instances, Collections.singletonList(classAttribute));
        }

        instances.setClass(classAttribute);

        return instances;
    }
}
