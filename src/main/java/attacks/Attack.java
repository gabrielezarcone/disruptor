package attacks;

import lombok.Getter;
import lombok.Setter;
import weka.core.Instance;
import weka.core.Instances;


public abstract class Attack {

    /**
     * Percentage of the Instances the attacker can EDIT
     * @return the capacity of the attack
     */
    @Getter
    private double capacity = 0;

    /**
     * Percentage of the Instances the attacker can VIEW
     * @return the knowledge of the attack
     */
    @Getter
    private double knowledge = 0;

    /**
     * instances target of the attack
     * @param target Instances target of the attack
     * @return instances target of the attack
     */
    @Getter @Setter
    private Instances target;



    // --------------------------------------------------------------------------------------------------------
    // -- Constructors ----------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------

    protected Attack(Instances target){
        setTarget(target);
        setCapacity(1);
        setKnowledge(1);
    }

    protected Attack(Instances target, double capacity, double knowledge){
        setTarget(target);
        setCapacity(capacity);
        setKnowledge(knowledge);
    }



    // --------------------------------------------------------------------------------------------------------
    // -- Abstract Methods ----------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------

    /**
     * Perform the attack
     * @return the target instances after the attack
     */
    public abstract Instances attack();

    /**
     * Calculate the ability of the attack to reduce the separation space between the classes
     * @return the calculated ability
     */
    public abstract int evaluateAbility();



    // --------------------------------------------------------------------------------------------------------
    // -- Methods ----------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------

    /**
     * Number of instances attacked
     * @return Number of target instances specified by the capacity
     */
    public int attackSize(){
        int targetDimension = getTarget().size();
        return (int) (targetDimension*getCapacity());
    }



    // --------------------------------------------------------------------------------------------------------
    // -- Getters and Setters ----------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------------------

    /**
     * Capacity is the percentage of the Instances the attacker can EDIT
     * @param capacity of the attack. Should be between 0 and 1
     * @throws IllegalArgumentException if capacity is not between 0 and 1
     */
    public void setCapacity(double capacity) {
        if(capacity>=0 && capacity<=1){
            this.capacity = capacity;
        }
        else {
            throw new IllegalArgumentException("The capacity should be between 0 and 1");
        }
    }

    /**
     * Knowledge is the percentage of the Instances the attacker can VIEW
     * @param knowledge of the attack. Should be between 0 and 1
     * @throws IllegalArgumentException if capacity is not between 0 and 1
     */
    public void setKnowledge(double knowledge) {
        if(knowledge>=0 && knowledge<=1){
            this.knowledge = knowledge;
        }
        else {
            throw new IllegalArgumentException("The knowledge should be between 0 and 1");
        }
    }
}
