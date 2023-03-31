package filters;

import util.AttributeUtil;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddValues;

public class ApplyFilter {


    /**
     *  Apply the weka AddValues filter to the destinationAttribute of destinationInstances.
     *  <p/>
     *  When applied the sourceAttribute's values will be added to the values of the destinationAttribute.
     *  Only the values that are not already present will be added.
     *  <p/>
     * @param destinationInstances the filter will be applied to this {@link Instances} object
     * @param sourceAttribute the new values are fetched from this {@link Attribute} object
     * @param destinationAttribute attribute of destinationInstances where the new values are added
     * @return a new version of destinationInstances filtered by the weka AddValues filter
     * @throws Exception if the destinationAttribute is not found in destinationInstances
     * @throws Exception if problems applying the filter
     */
    public static Instances addValues(Instances destinationInstances, Attribute sourceAttribute, Attribute destinationAttribute) throws Exception {

        AddValues addValuesFilter = new AddValues();
        addValuesFilter.setAttributeIndex( String.valueOf(destinationAttribute.index()+1) );
        addValuesFilter.setLabels( AttributeUtil.attributeValuesToString(sourceAttribute) );
        addValuesFilter.setInputFormat(destinationInstances);

        return  Filter.useFilter(destinationInstances, addValuesFilter);

    }
}
