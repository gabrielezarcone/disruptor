# Disruptor

This is an Adversarial Machine Learning tool that can be used to attack a  training set used to build a Machine Learning model. 

To do so, this code already contains some attacks, but the tool is designed to be extended with personal attack implementations.

## Use it from CLI
Download from the release section the latest disruptor.jar and run the following command for a simple execution:

`java -jar disruptor.jar datsetPath.csv`

To add more configurations, open the man page of the application with this command:

`java -jar disruptor.jar -v`

## Use it as a library
### Maven
Add github as a Maven repository in your pom.xml:
```
<repositories>
    <repository>
        <id>github</id>
        <url>https://maven.pkg.github.com/gabrielezarcone/disruptor</url>
        <snapshots>
            <enabled>true</enabled>
        </snapshots>
    </repository>
</repositories>
```
Then add the the following dependency:
```
<dependencies>
    <dependency>
        <groupId>gzarcone</groupId>
        <artifactId>disruptor</artifactId>
        <version>1.0.0</version>
    </dependency>
</dependencies>
```


## Extend the attacks

Extend the Disruptor class to change the attack list adding the new attack class named NewAttack:

```java
public class NewDisruptor extends Disruptor {
    @Override
    protected void populateAttacksList(Instances dataset, double[][] selectedFeatures) {
        
        super.populateAttacksList(dataset, selectedFeatures);
        NewAttack newAttack = new NewAttack(dataset);
        newAttack.setFeatureSelected(selectedFeatures);
        getAttacksList().add(newAttack);

    }
}
```
The NewAttack class should be an extension of the abstract class Attack:

```java
public class NewAttack extends Attack {
    protected NewAttack(Instances target) {
        super(target);
    }

    protected NewAttack(Instances target, double capacity, double featuresCapacity, double knowledge) {
        super(target, capacity, featuresCapacity, knowledge);
    }

    @Override
    public Instances attack() {
        // insert here the attack implementation
        return null;
    }
}
```