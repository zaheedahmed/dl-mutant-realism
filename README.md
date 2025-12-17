# An Empirical Study of the Realism of Mutants in Deep Learning

## Data Collection
Collect buggy and fixed versions for databugs and programs bugs.

### Data Bugs
1. CleanML: Only buggy data is provided. We fixed with the CleanML technique and added default classification DNN to complete the bug-fix i.e. buggy data, fixed data, training program.

### Program Bugs
We collected and analyzed program bugs from following dataset.
1. DeepFD
2. DeepLocalize
3. defect4ml

## Inclusion/Exclusion of Bugs
1. Keras/TensorFlow sequential or functional model
2. Classification task
3. Training data available
4. Evaluation data available
5. Bug symptom is not crash
6. Adaptable to python 3.8 without behavioral changes
7. Standard layers are used

## Training Models
1. Train n instances of original models
2. Train n instances faulty models
3. Train pre-training mutants each having n instances
4. Train post-training mutants each having n instances
5. Use model_validator.py to exclude any non-functional model or having instances < n

## Mutant Realism Analysis
1. execution_matrix.py
2. killing_probability.py
3. quantify_realism.py
4. visualization.ipynb