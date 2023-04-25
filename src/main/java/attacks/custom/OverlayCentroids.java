package attacks.custom;

import attacks.Attack;
import lombok.Getter;
import lombok.Setter;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.stream.IntStream;

public class OverlayCentroids extends Attack {

    private final SimpleKMeans simpleKMeans = new SimpleKMeans();

    /**
     * @param clustersNumber number of clusters in wich the instances are divided
     * @return the number of clusters
     */
    @Getter @Setter
    private int clustersNumber = 4;

    public OverlayCentroids(Instances target) {
        super(target);
    }

    public OverlayCentroids(Instances target, double capacity, double knowledge, int clustersNumber) {
        super(target, capacity, knowledge);
        setClustersNumber(clustersNumber);
    }

    @Override
    public Instances attack() {
        Instances instances = getTarget();
        try {
            // Cluster the instances and fetch the centroids of each cluster
            initSimpleKMeans();
            Instances clustersCentroids = centroids();

            // Build the mean centroid
            Instance meanCentroid = new DenseInstance( clustersCentroids.firstInstance() );

            ArrayList<Attribute> centroidAttributesList = Collections.list( clustersCentroids.enumerateAttributes() );
            centroidAttributesList.forEach( attribute -> {
                double attributeMean = clustersCentroids.meanOrMode(attribute);
                meanCentroid.setValue(attribute, attributeMean);
            });

            // Translate the instances towards the mean centroid
            IntStream.range( 0, attackSize() ).parallel().forEach( i -> {
                try {
                    Instance currentInstance = instances.instance(i);
                    Instance translatedInstance = translateInstance( currentInstance, clustersCentroids, meanCentroid );
                    instances.set( i, translatedInstance );
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });

        } catch (Exception e) {
            e.printStackTrace();
        }
        // return the perturbed instances
        return instances;
    }

    @Override
    public int evaluateAbility() {
        return 0;
    }

    /**
     * Start the clustering of the instances
     * @throws Exception if it cannot build the clustering
     */
    private void initSimpleKMeans() throws Exception {
        Instances instances = getTarget();
        // Remove the class attribute
        Instances instancesWithoutClass = new Instances(instances);
        instancesWithoutClass.setClassIndex(-1);

        simpleKMeans.setNumClusters(getClustersNumber());
        simpleKMeans.buildClusterer(instancesWithoutClass);
    }

    /**
     * @return Centroids of the simple K-Means cluster
     */
    private Instances centroids() {
        return simpleKMeans.getClusterCentroids();
    }

    /**
     * Translate the instance toward the mean centroid
     * @param instance instance to translate
     * @param clusterCentroids centroids of the difference clusters
     * @param meanCentroid mean centroids toward which the instance is translated
     */
    private Instance translateInstance(Instance instance, Instances clusterCentroids, Instance meanCentroid) throws Exception {
        Instance result = new DenseInstance( instance );
        // Fetch the instance cluster
        int clusterIndex = simpleKMeans.clusterInstance(instance);
        Instance clusterCentroid = clusterCentroids.get(clusterIndex);

        // For each attribute replace the instance value with the translated one
        ArrayList<Attribute> attributesList = Collections.list( instance.enumerateAttributes() );
        attributesList.forEach( attribute -> {

            double instanceValue = instance.value(attribute);
            double clusterCentroidValue = clusterCentroid.value(attribute);
            double meanCentroidValue = meanCentroid.value(attribute);

            double translatedValue = ( instanceValue - clusterCentroidValue) + meanCentroidValue;

            result.setValue(attribute, translatedValue);

        });

        return result;
    }
}
