# The-Intel-Academic-Program-for-oneAPI
Credit card transaction fraud detection
Written by python

## Environment Setup
Before running the code, place the data in the "data" folder.
When running in an Intel environment, execute the file with the "Intel" suffix to leverage hardware acceleration.

## Code Section
Two approaches were used to process credit card transaction fraud detection data.

### Approach 1: Random Forest Algorithm
1. Preprocessing: Data set partitioning, clustering using "cluster," and normalization.
2. Grid search to find the best parameters.
3. Prediction using random forest.

Evaluation: Assessing results using accuracy, precision, recall, and F1 score.

### Approach 2: XGBoost
Similar preprocessing and evaluation procedures.
Utilized SMOTE to address significant data set imbalances by resampling the data set.

Both before and after running, timing was performed using the time module to intuitively experience the hardware-accelerated speed.

Translation into English, with a language style similar to that found in a GitHub readme.
