package disruptor.filters;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;

public class ApplyClassBalancer {


    /**
     * Apply the filter ClassBalancer to balance the classes
     *
     * @param instances instances to balance
     * @return the balanced instances
     * @throws Exception if problems during the execution of the filter
     */
    public static Instances classBalancer(Instances instances) throws Exception {
        ClassBalancer classBalancer = new ClassBalancer();
        classBalancer.setInputFormat(instances);

        return Filter.useFilter(instances, classBalancer);
    }


    /**
     * Apply the filter SMOTE to balance the classes
     *
     * @param instances instances to balance
     * @return the balanced instances
     * @throws Exception if problems during the execution of the filter
     */
    public static Instances smote(Instances instances) throws Exception {
        SMOTE smote = new SMOTE();
        smote.setInputFormat(instances);

        return Filter.useFilter(instances, smote);
    }


    /**
     * Apply the filter Resample to balance the classes
     *
     * @param instances instances to balance
     * @return the balanced instances
     * @throws Exception if problems during the execution of the filter
     */
    public static Instances resample(Instances instances) throws Exception {
        Resample resample = new Resample();
        resample.setInputFormat(instances);

        return Filter.useFilter(instances, resample);
    }
}
