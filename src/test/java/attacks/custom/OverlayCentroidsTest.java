package attacks.custom;

import costants.FilePaths;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import util.ArffUtil;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class OverlayCentroidsTest {


    private static Instances dataset;

    @BeforeAll
    static void beforeAll() throws IOException {
        dataset = ArffUtil.readArffFile(FilePaths.TestPath.IRIS_FILE_PATH, "class");
    }


    @Test
    void attackFull() throws Exception {
        OverlayCentroids overlayCentroids = new OverlayCentroids(dataset);
        overlayCentroids.setFeaturesSet(1);
        overlayCentroids.setClustersNumber(3);
        Instances perturbedDataset = overlayCentroids.attack();

        ArffUtil.exportToArffWithDate( perturbedDataset, "OverlayCentroidsTest_" );
        assertNotEquals(dataset, perturbedDataset);
    }
}