package core;

import util.ArffUtil;
import util.DistanceMatrixUtil;
import costants.FilePaths;
import weka.core.*;
import weka.core.matrix.Matrix;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Date;

public class App {


    public static void main(String[] args)  {
        distanceMatrices();
    }

    private static void distanceMatrices() {
        try {

            Instances dataset = ArffUtil.readArffFile(FilePaths.IRIS_FILE_PATH, "class");

            printMatrix(dataset, new EuclideanDistance(dataset));
            printMatrix(dataset, new ManhattanDistance(dataset));
            printMatrix(dataset, new ChebyshevDistance(dataset));
            printMatrix(dataset, new MinkowskiDistance(dataset));

        } catch (FileNotFoundException e){
            e.printStackTrace();
            System.out.println("File not found");
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("IO exception");
        }
    }

    private static void printMatrix(Instances instances, DistanceFunction distanceFunction){
        DistanceMatrixUtil distanceMatrixUtil = new DistanceMatrixUtil();
        Matrix matrix = distanceMatrixUtil.computeDistanceMatrix(instances, distanceFunction);
        System.out.println(matrix);
    }

}
