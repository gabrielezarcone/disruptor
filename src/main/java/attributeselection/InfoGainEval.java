package attributeselection;

import lombok.extern.slf4j.Slf4j;
import util.ExceptionUtil;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;

import java.util.Arrays;

@Slf4j
public class InfoGainEval extends AbstractAttributeSelector {

    public InfoGainEval(Instances instances){
        super(instances);
    }

    public InfoGainEval(AbstractAttributeSelector selectorToCopy) throws Exception {
        super(selectorToCopy);
    }

    @Override
    protected double[][] selectAttributes() {
        Instances instances = getTargetInstances();

        AttributeSelection attsel = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();

        attsel.setEvaluator(eval);
        attsel.setSearch(search);

        double[][] attrRanks = new double[instances.numAttributes()][2];
        try {
            attsel.SelectAttributes(instances);
            attrRanks = attsel.rankedAttributes();
        } catch (Exception e) {
            log.error("Problem ranking attributes using InfoGainEvaluator");
            log.debug("attrRanks: "+ Arrays.toString(attrRanks));
            ExceptionUtil.logException(e, log);
        }

        return attrRanks;
    }

    @Override
    public AbstractAttributeSelector copy() throws Exception {
        return new InfoGainEval(this);
    }


}
