import pandas as pd
import numpy as np
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

def make_sample_binary(datum):
    """ Binary transformer for a single data-point """
    # Pclass: column 0. 0 if Pclass is 1
    if datum[0] == 1:
        datum[0] = 0
    else:
        datum[0] = 1

    # Sex is already binary in the data: column 1. Do nothing.

    # Age: column 2. 0 if Age < mean. 1 otherwise.
    if datum[2] < 29:
        datum[2] = 0
    else:
        datum[2] = 1

    # Siblings/spouses onboard: column 3. If any siblings, then 1. 
    if datum[3] > 0:
        datum[3] = 1
    else:
        datum[3] = 0

    # Parents/children onboard: column 4. If any parents/children onboard, then 1.
    if datum[4] > 0:
        datum[4] = 1
    else:
        datum[4] = 0

    # Fare: column 5. If fare < mean, then 0.
    if datum[5] < 32:
        datum[5] = 0
    else:
        datum[5] = 1

    return datum

def make_features_binary(data):
    """ 4.1: Transform each feature into binary variable """
    mean_age = (data[: , 2]).mean()

    mean_fare = (data[: , 5]).mean()

    for i in range(len(data)):
        # Pclass: column 0. 0 if Pclass is 1
        if data[i][0] == 1: 
            data[i][0] = 0
        else:
            data[i][0] = 1

        # Sex is already binary in the data: column 1. Do nothing.

        # Age: column 2. 0 if Age < mean. 1 otherwise.
        if data[i][2] < mean_age:
            data[i][2] = 0
        else:
            data[i][2] = 1

        # Siblings/spouses onboard: column 3. If any siblings, then 1. 
        if data[i][3] > 0:
            data[i][3] = 1
        else:
            data[i][3] = 0

        # Parents/children onboard: column 4. If any parents/children onboard, then 1.
        if data[i][4] > 0:
            data[i][4] = 1
        else:
            data[i][4] = 0

        # Fare: column 5. If fare < mean, then 0.
        if data[i][5] < mean_fare:
            data[i][5] = 0
        else:
            data[i][5] = 1

    #Check result
    print(data[:10])
    return data

def safe_div(x,y):
    if y == 0:
        return 1    # If y is 0, then log(x / y) will get multiplied by 0
    return x / y

def mutual_information(X_j, y):
    """ 4.2: Compute I(X_j, y) where X_j is a feature vector (the j-th column) """

    # We first compute H(X_j)
    # Count number of 0s and 1s (everything is binary)
    num_zeroes = 0
    num_ones = 0

    y_zeroes = 0
    y_ones = 0

    # This function is written in a bit of a hacky way, but it works because everything is binary
    X_zero_y_zero = 0
    X_one_y_zero = 0
    X_zero_y_one = 0
    X_one_y_one = 0

    for i, val in enumerate(X_j):
        if y[i] == 0:
            y_zeroes += 1
            if val == 0:
                num_zeroes += 1
                X_zero_y_zero += 1
            else:
                num_ones += 1
                X_one_y_zero += 1
        else:
            y_ones += 1
            if val == 0:
                num_zeroes += 1
                X_zero_y_one += 1
            else:
                num_ones += 1
                X_one_y_one += 1


    total_samples = len(X_j)

    probability_zero = num_zeroes / total_samples
    probability_one = num_ones / total_samples

    # For conditional probablities
    prob_X_zero_y_zero = X_zero_y_zero / total_samples
    prob_X_one_y_zero = X_one_y_zero / total_samples
    prob_X_zero_y_one = X_zero_y_one / total_samples
    prob_X_one_y_one = X_one_y_one / total_samples

    probability_y_zero = y_zeroes / total_samples
    probability_y_one = y_ones / total_samples

    H_X_j = (probability_zero * math.log(safe_div(1, probability_zero), 2)) + (probability_one * (math.log(safe_div(1, probability_one), 2)))

    # Compute I(X_j, y)
    H_X_j_y = prob_X_zero_y_zero * math.log(safe_div(probability_y_zero, prob_X_zero_y_zero), 2) + \
                prob_X_one_y_zero * math.log(safe_div(probability_y_zero, prob_X_one_y_zero), 2) + \
                prob_X_zero_y_one * math.log(safe_div(probability_y_one, prob_X_zero_y_one), 2) + \
                prob_X_one_y_one * math.log(safe_div(probability_y_one, prob_X_one_y_one), 2)

    return H_X_j - H_X_j_y

def best_split(X, y):
    """ Computes mutual information for each feature to decide which feature to split on """
    num_features = len(X[1])

    # Iterate through each feature to find which one has max information
    best_feature = -1
    max_I = 0

    for i in range(num_features):
        X_i = X[: , i]
        I_X_i_y = mutual_information(X_i, y)
        if (I_X_i_y > max_I):
            best_feature = i
            max_I = I_X_i_y

    return best_feature

def split_data(X, y, i):
    """ Splits data based on best feature. The left split corresponds to the feature value being 0. """
    num_features = len(X[1])
    X_left = np.empty([1, num_features])
    X_right = np.empty([1, num_features])

    y_left = np.empty([1, 1])
    y_right = np.empty([1, 1])

    for index, sample in enumerate(X):
        sample = sample.T.reshape([1, num_features])
        if sample[0][i] == 0:
            X_left = np.append(X_left, sample, axis=0)
            y_left = np.append(y_left, y[index].reshape([1, 1]), axis=0)
        else:
            X_right = np.append(X_right, sample, axis=0)
            y_right = np.append(y_right, y[index].reshape([1, 1]), axis=0)

    return X_left, y_left, X_right, y_right

COUNT = [10]

# Decision Tree Class
class Tree:

    def __init__(self, X, y, depth = 0, max_depth = 4):
        self.left = None
        self.right = None        
        self.feature = None
        self.max_depth = max_depth
        self.depth = depth
        self.data = None
        self.labels = None
        self.survived = None    # invariant: if survived != None, then we have a leaf node
        self.build_decision_tree(X, y)

    def build_decision_tree(self, X, y):
        """ 4.3: build decision tree on given X and y"""
        self.data = X
        self.labels = y

        # Stopping criteria
        if len(X) <= 50:
            if self.labels.mean() >= 0.5:
                self.survived = 1
            else:
                self.survived = 0

        if self.depth >= self.max_depth:
            if self.labels.mean() >= 0.5:
                self.survived = 1
            else:
                self.survived = 0

        if self.survived != None:
            return  # One of the stopping criteria met

        best_feature = best_split(X, y)
        X_left, y_left, X_right, y_right = split_data(X, y, best_feature)
        self.feature = best_feature
        self.left = Tree(X_left, y_left, depth=self.depth + 1, max_depth=self.max_depth)
        self.right = Tree(X_right, y_right, depth=self.depth + 1, max_depth=self.max_depth)

        return

    def predict(self, x):
        """ Prediction method: x is new sample.  """

        # Base case
        if self.survived != None:
            return self.survived

        # Recurse down the tree using the current node's feature
        feature = self.feature
        if x[feature] == 0:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def print_tree_aux(self, feature_names, space):
        """ Auxiliary recursive function """

        # Increase distance between levels
        space += COUNT[0]
    
        # Process right child first
        if self.right != None:
            self.right.print_tree_aux(feature_names, space)
    
        # Print current node after space
        # count
        print()
        for i in range(COUNT[0], space):
            print(end = " ")
        if self.feature != None:
            print(self.feature)
    
        # Process left child
        if self.left != None:
            self.left.print_tree_aux(feature_names, space)

    def print_tree(self, feature_names, space):
        """ Pretty-printer for Tree """
        self.print_tree_aux(feature_names, space)
        print("\n")

def train_and_test(X_train, y_train, X_test, y_test):
    """ Return accuracy """

    # Build tree
    tree = Tree(X_train, y_train)

    # Test prediction of each data-point in the test set
    correct = 0
    for i, x in enumerate(X_test):
        if tree.predict(x) == y_test[i]:
            correct += 1

    # Compute accuracy
    accuracy = correct / len(X_test)
    return accuracy

def predict_forest(forest, x):
    """ Consensus of decision trees """
    results = []

    for tree in forest:
        prediction = tree.predict(x)
        results.append(prediction)

    arr = np.array(results)
    if arr.mean() >= 0.5:
        return 1
    else:
        return 0 

def train_and_test_forest(X_train, y_train, X_test, y_test, size, drop=False):
    """ Return accuracy with a forest """

    # Call appropriate training function
    if drop:
        forest = random_forest_drop(X_train, y_train, size)
    else:
        forest = random_forest(X_train, y_train, size)

    # Test prediction of each data-point in the test set on each tree
    correct = 0
    for i, x in enumerate(X_test):
        if predict_forest(forest, x) == y_test[i]:
            correct += 1

    # Compute accuracy
    accuracy = correct / len(X_test)
    return accuracy

def random_forest(X, y, size, subset=0.8):
    """ 4.7: Training multiple decision trees. Size is number of trees, and subset is percentage of data to use. """
    forest = []
    
    for i in range(size):
        np.random.shuffle(X)
        slice_size = math.floor(subset * len(X))
        X_train = X[0:slice_size]
        y_train = y[0:slice_size]

        tree = Tree(X_train, y_train)
        forest.append(tree)

    return forest

def random_forest_drop(X, y, size):
    """ 4.8: Random forest leaving out one feature at a time """
    forest = []

    # Drop one feature at a time
    for i in range(len(X[0])):
        # Delete i-th column from X
        X_train = np.delete(X, i, axis=1)
        y_train = y

        tree = Tree(X_train, y_train)
        forest.append(tree)

    return forest

def cross_validate(X, y, k, forest=False, size=None, drop=False):
    """ 4.5: Does k-fold cross validation """

    # Randomly shuffle to make partitions random
    np.random.shuffle(X)

    # Partition into k datasets
    size = math.floor(len(X) / k)
    print(size)

    beginning_index = 0
    accuracy_arr = np.empty([0,])
    for i in range(k):
        # Take data slice from beginning_index to beginning_index + size
        end_index = beginning_index + size
        X_test = X[beginning_index:end_index]
        y_test = y[beginning_index:end_index]

        # Training data
        X_train = np.delete(X, slice(beginning_index, end_index), axis=0)
        y_train = np.delete(y, slice(beginning_index, end_index), axis=0)

        # Call train and test
        if forest:
            accuracy = train_and_test_forest(X_train, y_train, X_test, y_test, size, drop)
        else:
            accuracy = train_and_test(X_train, y_train, X_test, y_test)
        accuracy_arr = np.append(accuracy_arr, accuracy)

        # Increment beginning_index
        beginning_index = end_index

    return accuracy_arr

if __name__ == '__main__':
    feature_names, X, y = get_data('titanic_data.csv')
    X = make_features_binary(X)

    # Build decision tree for titanic_data
    tree = Tree(X, y)
    tree.print_tree(feature_names, 0)

    # 4.6: Personal feature vector
    personal_x = np.array([3, 0, 24, 1, 0, 7.23])
    print(tree.predict(make_sample_binary(personal_x)))

    # Test 10-fold cross-validation
    accuracy = cross_validate(X, y, 10)
    print(accuracy.mean())
    print(accuracy.std())

    # Random Forest with 80% of data
    # forest = random_forest(X, y, 5, 0.8)
    # for tree in forest:
    #     tree.print_tree(feature_names, 0)

    # Cross validation with random forest
    # accuracy = cross_validate(X, y, 10, forest=True, size=5)
    # print(accuracy.mean())
    # print(accuracy.std())

    # Personal prediction
    # print(predict_forest(forest, personal_x))

    # Random Forest II
    forest = random_forest_drop(X, y, 6)
    for tree in forest:
        tree.print_tree(feature_names, 0)

    accuracy = cross_validate(X, y, 10, forest=True, size=6, drop=True)
    print(accuracy.mean())
    print(accuracy.std())
    
    # Personal prediction
    print(predict_forest(forest, personal_x))