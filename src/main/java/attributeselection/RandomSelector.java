package attributeselection;

import weka.core.Instances;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RandomSelector extends AbstractAttributeSelector{

    public RandomSelector(Instances instances) {
        super(instances);
    }

    public RandomSelector(AbstractAttributeSelector selector) {
        super(selector);
    }

    @Override
    public AbstractAttributeSelector copy() {
        return new RandomSelector(this);
    }

    @Override
    protected double[][] selectAttributes() {
        Instances instances = getTargetInstances();
        int numAttributes = instances.numAttributes();

        double[][] attrRanks = new double[numAttributes][2];

        // Create a list of all the integers between 0 and the number of attributes
        List<Integer> rangeList = IntStream.range(0, numAttributes).boxed().collect(Collectors.toList());
        Collections.shuffle(rangeList);

        for( int i=0; i<rangeList.size(); i++ ){
            int index = rangeList.get(i);
            attrRanks[i] = new double[] {index, rangeList.size()-i};
        }

        return attrRanks;
    }
}
