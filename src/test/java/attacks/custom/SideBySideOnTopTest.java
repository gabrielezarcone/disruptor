package attacks.custom;

import costants.FilePaths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import util.ArffUtil;
import weka.core.Instances;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class SideBySideOnTopTest {


    private static Instances dataset;

    @BeforeAll
    static void beforeAll() throws IOException {
        dataset = ArffUtil.readArffFile(FilePaths.TestPath.IRIS_FILE_PATH, "class");
    }


    @Test
    void attackFull() throws IOException {
        SideBySide sideBySideAttack = new SideBySideOnTop(dataset);
        sideBySideAttack.setFeaturesSet(1);
        Instances perturbedDataset = sideBySideAttack.attack();

        ArffUtil.exportToArffWithDate( perturbedDataset, "SideBySideOnTopTest_" );
        assertNotEquals(dataset, perturbedDataset);
    }

}