package model.perturbeddataset;

import lombok.Getter;
import lombok.Setter;
import weka.core.Instances;

public class PerturbedDataset {

    @Getter @Setter
    private PerturbedDatasetParams params;

    @Getter @Setter
    private Instances dataset;

}
