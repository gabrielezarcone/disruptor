package util;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;

public class DistanceMatrixUtil {

    public Matrix computeDistanceMatrix(Instances dataset, DistanceFunction distanceFunction ){

        int numInstances = dataset.numInstances();

        Matrix matrix = new Matrix(numInstances,numInstances);

        Instance instance1;
        Instance instance2;
        for( int i=0; i<numInstances; i++ ){
            // For each instance:
            instance1 = dataset.instance(i);
            for( int j=0; j<numInstances; j++ ){
                // Compute distance with any other instance :
                instance2 = dataset.instance(j);
                matrix.set( i, j, distanceFunction.distance(instance1, instance2) );
            }
        }

        return matrix;
    }
}
