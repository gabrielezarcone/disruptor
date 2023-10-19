package model.perturbeddataset;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import weka.core.Instances;

@EqualsAndHashCode
@AllArgsConstructor
public class PerturbedDataset {

    @Getter @Setter
    private Instances dataset;

    @Getter @Setter
    private Instances testSet;

    @Getter @Setter
    private PerturbedDatasetParams params;

}
