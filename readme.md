# ML project 2022

For run.py to work, please place the files `test.csv` and `train.csv` in the `project1` folder.

For data processing, we normalize our dataset, then expand it in the following way:
- powers 2, 3, 4 of all features
- logarithms of selected features
- sin, cos of angle features
- products of selected feature pairs
- a feature of constant value 1

We use an MSE Stochastic Gradient Descent Regressor. At each iteration, we clip the predictions to [-1, 1].

The final predictions are mapped to 1 if positive(>0), and -1 otherwise.