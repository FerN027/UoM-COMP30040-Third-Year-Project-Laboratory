This project is about the implementation, analysis and extension of a ML optimisation and verification algorithm that combines Bayesian optimisation and SMT solver, trying to solve this problem:

```
Given a trained ML model:

max T s.t. there exists region centered at point (C_i, y_i), whose value >= T everywhere
```

The main algorithm is implemented in **GearOpt_BO.py**, following the pseudocode proposed in the paper: [Combining Constraint Solving and Bayesian Techniques for System Optimization (Brauße et al., 2022)](https://www.ijcai.org/proceedings/2022/249).


**data_preprocesser.py**, **random_samples_generator.py**: used for data generating and preprocessing.

**spec_example.json**: example for the correct format to specify features' bounds and allowed radius.

### **Under folder 'comparison_between_models':**
Trade-off between accuracy and GearOpt_BO solving time are compared for different models such as neural network with ReLu, polynomial models and decision trees, each of which has its own SMT representation in z3.

### **Under folder 'symbolic_representation_simplification_for_trees':**
GearOpt_BO solving times for two different SMT representations of a decision tree model are compared, one representation as nested z3.If and the other as parallel z3.Implies.

### **Under folder 'extension_on_other_models':**
Attempts have been made to apply GearOpt_BO on other models such as boosted trees, requiring translation from model into SMT formulas.
