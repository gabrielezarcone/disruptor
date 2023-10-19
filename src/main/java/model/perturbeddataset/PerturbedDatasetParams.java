package model.perturbeddataset;

import attacks.Attack;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;


@AllArgsConstructor
@EqualsAndHashCode
public class PerturbedDatasetParams {

    public PerturbedDatasetParams(String featureSelectionAlgorithm,int runNumber){
        this.featureSelectionAlgorithm = featureSelectionAlgorithm;
        this.runNumber = runNumber;
    }

    @Getter @Setter
    private String featureSelectionAlgorithm = "";

    @Getter @Setter
    private Attack attack = null;

    @Getter @Setter
    private double capacity = 0;

    @Getter @Setter
    private double featuresCapacity = 0;

    @Getter @Setter
    private double knowledge = 0;

    @Getter @Setter
    private int runNumber = 0;

}
