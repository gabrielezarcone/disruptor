package model.perturbeddataset;

import attacks.Attack;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;


@AllArgsConstructor
@EqualsAndHashCode
public class PerturbedDatasetParams {

    @Getter @Setter
    private String relationshipName;

    @Getter @Setter
    private Attack attack;

    @Getter @Setter
    private double capacity;

    @Getter @Setter
    private double knowledge;

}
