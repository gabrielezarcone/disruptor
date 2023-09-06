package roc;

import util.InstancesUtil;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class ROCDatasetsList {

    /**
     * Attack name is the String key of the first map
     * Capacity is the Double key of the second map
     * Instances is the dataset with that attack name and that capacity
     */
    private final HashMap<String, HashMap<Double, Instances> > datasetMap = new HashMap<>();

    /**
     *
     * @param attackName this is the key to retrieve the capacities map
     * @param capacity this is the key of the capacity map
     * @return the Instances of the given attack name and the given capacity
     */
    public Instances getWithCapacity(String attackName, Double capacity){
        HashMap<Double, Instances> capacityMap = datasetMap.get(attackName);
        if(capacityMap==null){
            return null;
        }
        else {
            return capacityMap.get(capacity);
        }
    }

    /**
     *
     * @param attackName this is the key to retrieve the capacities map
     * @param capacity this is the key of the capacity map
     * @param instances the instances to put inside the inner map following the two keys specified appending it if some instances are already present or creating a new node of the map
     * @throws Exception if is not possible to append the instances
     */
    public void addWithCapacity(String attackName, Double capacity, Instances instances) throws Exception {
        // Map of all the perturbed datasets by capacity
        Map<Double, Instances> capacitiesMap = getCapacitiesMap(attackName);

        if(capacitiesMap == null || capacitiesMap.isEmpty()) {
            // No other perturbed dataset for this attack name and this capacity added before
            HashMap<Double, Instances> emptyCapacitiesMap = new HashMap<>();
            emptyCapacitiesMap.put(capacity, instances);
            datasetMap.put(attackName, emptyCapacitiesMap);
        }
        else {
            // There is already a dataset for this attack name
            Instances dataset = capacitiesMap.get(capacity);
            if( dataset == null || dataset.isEmpty() ) {
                // There is already a dataset for this attack name but this is the first for this capacity
                datasetMap.get(attackName).put(capacity, instances);
            }
            else {
                // There is already a dataset for this attack name and this capacity
                InstancesUtil.addAllInstances(dataset, instances);
                datasetMap.get(attackName).put(capacity, dataset);
            }
        }
    }

    /**
     *
     * @param attackName key of the attack
     * @return the map of all the perturbed dataset of that attack with different capacities
     */
    public Map<Double, Instances> getCapacitiesMap(String attackName){
        return datasetMap.get(attackName);
    }

    public Set<String> keySet(){
        return datasetMap.keySet();
    }

    public void clear() {
        datasetMap.clear();
    }
}
