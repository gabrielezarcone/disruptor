package experiment.resultmatrix;

import weka.experiment.ResultMatrixPlainText;

public class ResultMatrixPlainTextArray extends ResultMatrixPlainText {

    @Override
    public String[][] toArray(){
        return super.toArray();
    }

    public String[][] traspose(){
        String[][] matrix = toArray();
        String[][] auxMatrix = new String[matrix[0].length][matrix.length];
        for(int i=0; i<matrix.length; i++){
            for (int j = 0; j < matrix[0].length; j++) {
                auxMatrix[j][i] = matrix[i][j];
            }
        }
        return auxMatrix;
    }
}
