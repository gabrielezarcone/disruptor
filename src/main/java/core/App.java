package core;

import attacks.Attack;
import attacks.custom.OverlayCentroids;
import attacks.custom.SideBySide;
import attacks.custom.SideBySideOnTop;
import util.ArffUtil;
import costants.FilePaths;
import util.InstancesUtil;
import weka.core.*;

import java.text.SimpleDateFormat;
import java.util.Date;

public class App {
    public static final String FILE_PATH = FilePaths.TestPath.DIABETIC_FILE_PATH;
    public static final String CLASS_NAME = "Class";

    private static Instances dataset;
    private static String folderName;


    public static void main(String[] args) throws Exception {
        dataset = ArffUtil.readArffFile(FILE_PATH, CLASS_NAME);
        // Set folder name
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HHmmss");
        folderName = simpleDateFormat.format(new Date());
        // Save the original arff in the same folder:
        ArffUtil.exportToArffWithDateinFolder(dataset, folderName, dataset.relationName());
        // Run all the attacks
        performAttack("SideBySide_50", new SideBySide(dataset, 0.5, 1));
        performAttack("SideBySide_100", new SideBySide(dataset, 1, 1) );
        performAttack("SideBySideOnTop_50", new SideBySideOnTop(dataset, 0.5, 1));
        performAttack("SideBySideOnTop_100", new SideBySideOnTop(dataset, 1, 1));
        performAttack("OverlayCentroids_50", new OverlayCentroids(dataset, 0.5, 1, 4) );
        performAttack("OverlayCentroids_100", new OverlayCentroids(dataset, 1, 1, 4) );
    }

    /**
     * Perform the attack
     * @param attackCode String used to identify this particular attack execution
     * @param attack the attack to be performed
     * @throws Exception if problems during the execution of the attack
     */
    private static void performAttack(String attackCode, Attack attack) throws Exception {
        Instances perturbedDataset = attack.attack();
        exportPerturbedDataset(attackCode, perturbedDataset);
    }

    /**
     * Export the perturbed dataset in the same folder of the others attack
     * @param attackCode String used to identify this particular attack execution
     * @param perturbedDataset The perturbed dataset after the attack
     * @throws Exception if problems during the export
     */
    private static void exportPerturbedDataset(String attackCode, Instances perturbedDataset) throws Exception {
        perturbedDataset.setRelationName(dataset.relationName() + "-" + attackCode);
        // Merge Perturbed with the original one to use it as the test instances
        InstancesUtil.addAllInstances(perturbedDataset, dataset);
        ArffUtil.exportToArffWithDateinFolder(perturbedDataset, folderName, attackCode +"_" );
    }

}
