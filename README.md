# models-from-scratch

This repository contains implementations of 3 different machine learning models
from scratch in Python: decision trees (random forest), logistic regression and
k-nearest neighbors. 

Each of these models makes predictions based on the Titanic dataset (titanic_dataset.csv).
This dataset contains demographic and family information about the passengers on the Titanic,
labeled with whether they survived the Titanic's crash (1 representing survival). Each model
partitions the dataset into training and test sets and prints the accuracy of the model. The Python 
modules also provide functions for performing cross-validation.

To run a model, simply run the corresponding file as a Python script. 
For example: `python k_nearest_neighbors.py`.

The goal of this project was to learn about these models as an educational exercise. 
It is not under development.