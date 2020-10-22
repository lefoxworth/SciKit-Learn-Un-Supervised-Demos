CS465
Homework 2
Laura Foxworth

####################

Part 1:

To run the program:
Run part1.py. The output of the program will be placed in z-predicted.csv.

The learner is implemented using scikit-learn's MLPRegressor class. It uses the following parameters:
Hidden layer size = (3, 3, 3)
Solver = lbfgs
Random_state = 1
Max_iter = 500

The MLPRegressor class was chosen as the learner because of its capability to handle nonlinear relationships between data.
I tried to avoid overfitting my model by limiting its complexity, specifically the number of neurons in each hidden layer. The L-BFGS solver
was used as it is able to converge faster on smaller datasets compared to the default ADAM solver.
On testing, the average coefficient of determination was .95.

####################

Part 2:

Running part2.py provides the means and covariances of the data. It also produces a plot of the data and the clusters as contours.

I analyzed the data using a Gaussian Mixture Model (GMM) class. This model was chosen over KMeans because the data was not well separated
and exhibited high overlap. Kmeans only performs well on highly circular, non-mixed datasets. GMM works better on this data because
it contains a probabilistic model, giving a better guess as to the likelihood of a particular point being in a particular cluster, as
well as being more flexible with skewed and oblongly shaped data clusters.

The model is trained using the following parameters:
n_components = 5
Covariance type = diag
Max_iter = 500

Diagonal covariance was chosen because the dataset appeared to be skewed and oblong. 5 components were chosen via trial and error visual
comparison of n = 1 to n = 10 components, arriving at 5 as the most accurate number.