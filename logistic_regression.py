import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy.stats import norm
from scipy.stats.distributions import chi2
import time
# import math
# Construct data matrices

def get_data(filename):
    # Read dataset
    df = pd.read_csv(filename)
    
    # Sanity check
    print(df.head())
    
    # Format data for training
    # X is the feature matrix
    X = df[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]
    
    # Add a column of 1s for theta_0
    X = X.values
    
    # print(X[:10])
    X = np.insert(X, 0, 1, axis=1)
    print(X[:10])
    
    # Y is the labels array
    y = df[['Survived']]
    y = y.values
    print(y[:10])
    
    # Initialize theta values
    theta = np.zeros((X.shape[1], 1))
    
    # print(theta)
    return X, y, theta

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Likelihood function
def likelihood_function(X, y, theta):
    n = len(y)
    epsilon = 1e-5
    h = 1 / (1 + np.exp(-(X @ theta)))
    log_likelihood = (1 / n) * (((-y).T @ np.log(h + epsilon)) 
                                - ((1 - y).T @ np.log(1 - h + epsilon)))
    return log_likelihood

# Gradient descent/ascent
def gradient_descent(X, y, theta, learning_rate, iterations,
tolerance):
    n = len(y)
    likelihood_progression = np.zeros((iterations,1))
    print(time.perf_counter())
    for i in range(iterations):
        gradient = (X.T @ (sigmoid(X @ theta) - y))
        diff = learning_rate * (1 / n) * gradient
        if np.all(np.abs(diff) <= tolerance):
            break
        theta = theta - diff
        likelihood_progression[i] = likelihood_function(X, y, theta)
    print(time.perf_counter())
    return theta, likelihood_progression, i

# MLE of new sample x
def predict(theta_hat, x):
    return np.round(sigmoid(x @ theta_hat))

# MLE of log-odds
def log_odds_mle(theta_hat, x):
    return (x @ theta_hat)

# Distribution of theta_hat (Fisher Information Matrix)
def distr_theta_hat(X, theta_hat):
    matrix = np.zeros((X.shape[1], X.shape[1]))
    for x in X:
        x = np.reshape(x, (len(x), 1))
        xxT = x @ x.T
        numerator = np.exp(-(theta_hat.T @ x))
        denominator = (1 + np.exp(-(theta_hat.T @ x))) ** 2
        t_matrix = (numerator / denominator) * xxT
        matrix = matrix + t_matrix
    return inv(matrix)

# Distribution of log_odds_mle: x is the new sample and I_inv is the Fisher info matrix
def distr_log_odds_mle(x, I_inv):
    return x @ I_inv @ x.T

# Confidence interval: calculate Tau
def confidence_interval(alpha, stdev):
    return norm.ppf(alpha/2, scale=stdev)

# Which features are significant?
def significant_features(theta_hat, I_inv, alpha):
    hypothesis_tests = np.empty((len(theta_hat), 1))
    inv_chi = chi2.ppf(alpha, df=5)
    print(inv_chi)
    for i, theta_j in enumerate(theta_hat):
        neu_j = I_inv[i][i]
        ratio = (theta_j / neu_j) ** 2
        print(ratio)
        if ratio > inv_chi:
            hypothesis_tests[i] = 1
        else:
            hypothesis_tests[i] = 0
    return hypothesis_tests

if __name__ == '__main__':
    X, y, theta = get_data('titanic_data.csv')
    learning_rate = 0.03
    iterations = 10000
    tolerance = 1e-05
    
    # Call gradient descent
    theta_hat, likelihood_history, iters_required = gradient_descent(X, y, theta, learning_rate, iterations, tolerance)
    print(theta_hat)
    print(iters_required)

    # Personal x
    personal_x = np.array([1, 3, 0, 24, 1, 0, 7.23])
    personal_x = np.reshape(personal_x, (len(personal_x), 1)).T
    
    # Prediction of personal feature vector
    print(predict(theta_hat, personal_x))
    
    # MLE of log-odds likelihood of personal feature vector
    print(log_odds_mle(theta_hat, personal_x))
    
    # Distribution of theta_hat (information matrix)
    info_matrix = distr_theta_hat(X, theta_hat)
    print(info_matrix)
    
    # Distribution of personal log-odds likelihood
    personal_log_odds_stdev = distr_log_odds_mle(personal_x, info_matrix)
    print(personal_log_odds_stdev)
 
    # Tau value
    alpha = 0.05
    tau = confidence_interval(alpha, personal_log_odds_stdev)
    print(tau)
    
    # Likelihood ratio tests for each feature
    ratio_tests = significant_features(theta_hat, inv(info_matrix), alpha)
    print(ratio_tests)