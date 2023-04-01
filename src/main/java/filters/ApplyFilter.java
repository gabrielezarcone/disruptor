package filters;

import util.AttributeUtil;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddValues;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.SortLabels;

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

    /**
     * Sort in ascending order the values (aka labels) of destinationInstances
     * @param destinationInstances instances whose values should be sorted
     * @return a new Instances object with the values of every attribute sorted in ascending order
     * @throws Exception if problems applying the filter
     */
    public static Instances sortLabels(Instances destinationInstances) throws Exception {

        SortLabels sortLabels = new SortLabels();
        sortLabels.setInputFormat(destinationInstances);

        return  Filter.useFilter(destinationInstances, sortLabels);

    }

    /**
     * Reorder instances using the order specified by attributesIndices.
     * <p/>
     * e.g. if instances attributes are, in order, a,b,c,d and attributesIndices is [0,3,2,1],
     * this filter returns a new {@link Instances} object with the attribute in the following order: a,d,c,b
     * <p/>
     *
     * @param instances Instances whose attributes are to reorder
     * @param attributesIndices Array of positions.
     * @return
     * @throws Exception if problems applying the filter
     */
    public static Instances reorderAttributes(Instances instances, int[] attributesIndices) throws Exception {
        Reorder reorder = new Reorder();
        reorder.setAttributeIndicesArray(attributesIndices);
        reorder.setInputFormat(instances);

        return Filter.useFilter(instances, reorder);
    }
}
