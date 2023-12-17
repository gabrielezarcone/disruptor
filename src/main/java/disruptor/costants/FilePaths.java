package disruptor.costants;

public class FilePaths {

    public static final String IRIS_FILE_PATH = "src/main/resources/iris.arff";
    public static final String OUTPUT_FOLDER = "src/main/resources/output/";

    public static class ExplorerResults{
        private ExplorerResults(){}
        public static final String PROVA_1 = "src/main/resources/ExplorerResults/original/prova1.arff";
        public static final String PROVA_2 = "src/main/resources/ExplorerResults/original/prova2.arff";
        public static final String PROVA_3 = "src/main/resources/ExplorerResults/original/prova3.arff";
        public static final String PROVA_4 = "src/main/resources/ExplorerResults/original/prova4-3alg.arff";
    }

    public static class TestPath{
        private TestPath(){}
        public static final String IRIS_FILE_PATH = "src/main/resources/test/iris.arff";
        public static final String DIABETIC_FILE_PATH = "src/main/resources/test/Diabetic_Retinopathy_Debrecen.arff";
    }
}
