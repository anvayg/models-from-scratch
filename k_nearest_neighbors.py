import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
import math

def get_data(filename):
    """ Construct data matrices """
    # Read dataset
    df = pd.read_csv(filename)

    # Sanity check
    print(df.head())

    # X is the feature matrix
    X = df[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]
    feature_names = X.columns.values
    X = X.values

    # Y is the labels array
    y = df[['Survived']]
    y = y.values

    return feature_names, X, y

# K-Nearest Neighbors
def minkowski_distance(a, b, p):
    """ Calculate distance between a and b using given parameter p """
    size = len(a)
    dist = 0

    for i in range(size):
        dist += abs(a[i] - b[i]) ** p

    dist = dist ** (1 / p)

    return dist

def k_nearest_neighbors(X, x, k, p=2):
    """ Find k-nearest points to x and return as list of tuples of the form (dist, index) """
    distances = []

    for index, sample in enumerate(X):
        distance = minkowski_distance(x, sample, p)
        distances.append((distance, index))

    # k-smallest distances
    return heapq.nsmallest(k, distances)    # Tuples by default are sorted by their first projection

def predict(X, y, x, k, p=2):
    """ Takes consensus of k-nearest neighbors """
    k_nearest = k_nearest_neighbors(X, x, k, p)
    y_zeros = 0

    for (_, index) in k_nearest:
        if y[index] == 0:
            y_zeros += 1

    if y_zeros >= (k / 2):
        return 0
    else:
        return 1

def predict_increasing_k(X, y, x, p=2):
    """ Predictions with k = 1, ..., N """
    predictions = []

    for k in range(1, 100):
        prediction = predict(X, y, x, k, p)
        predictions.append(prediction)

    plt.scatter(range(1, 100), predictions)
    plt.xlabel('k = Number of nearest neighbors')
    plt.ylabel('Prediction')
    plt.show()


# Naive Bayes: hacky implementation, because we know y is binary here
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def prob_y(y):
    """ Computes P(y = 1) and P(y = 0) """
    num_samples = len(y)
    y_ones = 0

    for outcome in y:
        if outcome == 1:
            y_ones += 1

    prob_y_one = y_ones / num_samples
    prob_y_zero = 1 - prob_y_one
    return prob_y_one, prob_y_zero

# Conditional probabilities for each feature
# It would be better to just have one function here, which is parametric in whether a variable is bernoulli/gaussian
def prob_pclass(X, y_class, x, i=0):       # Multinomial
    """ Computes P(X_i = x | y = y_class) for Pclass """
    X_count = 0
    for index, outcome in enumerate(y):
        if outcome == y_class and X[index][i] == x[i]:
            X_count += 1

    return X_count / len(X)

def prob_sex(X, y_class, x, i=1):      # Bernoulli
    """ Computes P(X_i = x | y = y_class) for Sex """
    X_count = 0
    for index, outcome in enumerate(y):
        if outcome == y_class and X[index][i] == x[i]:
            X_count += 1

    return X_count / len(X)

def prob_age(X, y_class, x, i=2):      # Gaussian  
    """ Computes P(X_i = x | y = y_class) for Age """
    X_collection = np.empty([])
    for index, outcome in enumerate(y):
        if outcome == y_class:
            X_collection = np.append(X_collection, X[index][i])

    mean = np.mean(X_collection)
    sd = np.std(X_collection)

    return normpdf(x[i], mean, sd)

def prob_siblings(X, y_class, x, i=3):     # Multinomial
    """ Computes P(X_i = x | y = y_class) for Siblings """
    X_count = 0
    for index, outcome in enumerate(y):
        if outcome == y_class and X[index][i] == x[i]:
            X_count += 1

    return X_count / len(X)

def prob_parents(X, y_class, x, i=4):      # Multinomial
    """ Computes P(X_i = x | y = y_class) for Parents """
    X_count = 0
    for index, outcome in enumerate(y):
        if outcome == y_class and X[index][i] == x[i]:
            X_count += 1

    return X_count / len(X)

def prob_fare(X, y_class, x, i=5):     # Gaussian
    """ Computes P(X_i = x | y = y_class) for Fare """
    X_collection = np.empty([])
    for index, outcome in enumerate(y):
        if outcome == y_class:
            X_collection = np.append(X_collection, X[index][i])

    mean = np.mean(X_collection)
    sd = np.std(X_collection)

    return normpdf(x[i], mean, sd)

if __name__ == '__main__':
    feature_names, X, y = get_data('titanic_data.csv')

    personal_x = np.array([3, 0, 24, 1, 0, 7.23])

    # KNN
    # Personal prediction
    print(predict(X, y, personal_x, 5))

    # Increasing k
    # predict_increasing_k(X, y, personal_x)


    # Naive Bayes
    prob_y_one, prob_y_zero = prob_y(y)

    # posterior_one
    posterior_one = prob_y_one * prob_pclass(X, 1, personal_x) * prob_sex(X, 1, personal_x) * \
                    prob_age(X, 1, personal_x) * prob_siblings(X, 1, personal_x) * prob_parents(X, 1, personal_x) * \
                    prob_fare(X, 1, personal_x)
    print(posterior_one)

    # posterior_zero
    posterior_zero = prob_y_zero * prob_pclass(X, 0, personal_x) * prob_sex(X, 0, personal_x) * \
                    prob_age(X, 0, personal_x) * prob_siblings(X, 0, personal_x) * prob_parents(X, 0, personal_x) * \
                    prob_fare(X, 0, personal_x)
    print(posterior_zero)