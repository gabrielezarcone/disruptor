package attacks.labelflipping;

import attacks.Attack;
import costants.FilePaths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import util.ArffUtil;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

class RandomLabelFlippingTest {

    private static Instances dataset;

    @BeforeAll
    static void beforeAll() throws IOException {
        dataset = ArffUtil.readArffFile(FilePaths.TestPath.IRIS_FILE_PATH, "class");
    }

    @Test
    void attackFull() {
        RandomLabelFlipping randomLabelFlipping = new RandomLabelFlipping(dataset);
        Instances perturbedDataset = randomLabelFlipping.attack();

        compareAttackResultClassValues(randomLabelFlipping, perturbedDataset);
    }

    @Test
    void attack50PercentCapacity() {
        RandomLabelFlipping randomLabelFlipping = new RandomLabelFlipping(dataset,0.5,1);
        Instances perturbedDataset = randomLabelFlipping.attack();
        compareAttackResultClassValues(randomLabelFlipping, perturbedDataset);
    }

    private void compareAttackResultClassValues(Attack attack, Instances perturbedDataset) {
        ArrayList<Instance> datasetList = Collections.list(dataset.enumerateInstances());
        ArrayList<Instance> perturbedList = Collections.list(perturbedDataset.enumerateInstances());

        for(int i = 0; i<attack.getTarget().size(); i++){
            double originalClassValue = datasetList.get(i).classValue();
            double newClassValue = perturbedList.get(i).classValue();
            if(i<attack.attackSize()){
                assertNotEquals(originalClassValue, newClassValue);
            }
            else {
                assertEquals(originalClassValue, newClassValue);
            }
        }
    }
}