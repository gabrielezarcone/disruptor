package core;


import attacks.Attack;
import attacks.labelflipping.RandomLabelFlipping;
import costants.FilePaths;
import saver.Exporter;
import util.ArffUtil;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

public class App {

    private static String file_path = FilePaths.IRIS_FILE_PATH;
    private  static String className = "class";

    private static Instances dataset;
    private static String folderName;
    private static ArrayList<Attack> attacksList = new ArrayList<>();
    private static ArrayList<Double> capacitiesList = new ArrayList<>();


    public static void main(String[] args) throws IOException {
        dataset = ArffUtil.readArffFile(file_path, className);
        //---
        attacksList.add(new RandomLabelFlipping(dataset));
        capacitiesList.add(0.5);
        //---
        // Set folder name
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HHmmss");
        folderName = simpleDateFormat.format(new Date());
        // Attack main loop
        performAttacks(dataset, attacksList, capacitiesList);
    }

    private static void performAttacks(Instances trainingSet, ArrayList<Attack> attacksList, ArrayList<Double> capacitiesList){
        // Nested loop between attacks list and capacities list
        attacksList.forEach( attack -> capacitiesList.forEach(capacity -> {

            // Define an attack code unique for this attack run
            String attackCode = trainingSet.relationName() + "_" + attack.getClass().getSimpleName() + "_" + capacity;

            // Perform this attack with this capacity
            Instances trainingSetCopy = new Instances(trainingSet);
            attack.setTarget( trainingSetCopy );
            attack.setCapacity( capacity );
            Instances perturbedInstances = attack.attack();
            perturbedInstances.setRelationName(attackCode);

            // Export the perturbed instances
            try {
                exportPerturbedDataset(attackCode, perturbedInstances);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }));
    }

    /**
     * Export the perturbed dataset in the same folder of the others attack
     * @param attackCode String used to identify this particular attack execution used as file name
     * @param perturbedDataset The perturbed dataset after the attack
     * @throws Exception if problems during the export
     */
    private static void exportPerturbedDataset(String attackCode, Instances perturbedDataset) throws Exception {
        // Export ARFF
        Exporter arffExport = new Exporter( new ArffSaver() );
        arffExport.exportInFolder( perturbedDataset, folderName, attackCode );
        // Export CSV
        Exporter csvExport = new Exporter( new CSVSaver() );
        csvExport.exportInFolder( perturbedDataset, folderName, attackCode );
    }
}
