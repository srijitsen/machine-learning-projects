# FoML Assign 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import os
import csv
import numpy as np
import statistics
# Enter You Name Here
myname = "srijit sen"

lib_path = "C:/Users/300063669/Downloads/MDS related Documents/FOML/Assignment 1/"
os.chdir(lib_path)

class Node:
    """A decision tree node."""
    def __init__(self, EN, n_samples, n_samples_per_class, predicted_class):
        self.EN = EN
        self.n_samples = n_samples
        self.n_samples_per_class = n_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        """Build decision tree."""
        self.n_classes= len(set(Y))  # classes are assumed to go from 0 to n-1
        self.n_features = X.shape[1]
        self.tree = self.learn(X, Y)

    def Gini(self, Y):
        """Compute Gini impurity of a non-empty node.
        """
        total_cases = Y.size
        return 1.0 - sum((np.sum(Y == c) / total_cases) ** 2 for c in range(self.n_classes))

    def calc_impurity(self, X, Y):
        """Find the best split for a node.impurity of the two children, weighted by their
        population, is the smallest and less than the impurity of the parent node.

        best_col: Index of the feature for best split, or None if no split is found.
        best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        total_cases = Y.size
        if total_cases  <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(Y == c) for c in range(self.n_classes)]

        # Gini of current node.
        best_EN = 1.0 - sum((n/total_cases) ** 2 for n in num_parent)
        best_col, best_thr = None, None

        # Loop through all features.
        for col in range(self.n_features):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, col], Y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, total_cases ):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                EN_left_child=  1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes)
                )
                EN_right_child=1.0 - sum(
                    (num_right[x] / (total_cases - i)) ** 2 for x in range(self.n_classes)
                )

                # weighted average of impurity
                EN = (i * EN_left_child + (total_cases - i) * EN_right_child) / total_cases 

                # Avoiding identical values, and skipping the loop till it reaches unique point
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if EN < best_EN:
                    best_EN = EN
                    best_col = col
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_col, best_thr

    def learn(self, X, Y, depth=0):
        
        # Predicted class will be the one with higher population
        n_samples_per_class = [np.sum(Y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(n_samples_per_class)
        node = Node(
            EN=self.Gini(Y),
            n_samples=Y.size,
            n_samples_per_class=n_samples_per_class,
            predicted_class=predicted_class,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth: 
            col, thr = self.calc_impurity(X, Y)
            if col is not None:
                indices_left = X[:, col] < thr
                X_left, Y_left = X[indices_left], Y[indices_left]
                X_right, Y_right = X[~indices_left], Y[~indices_left]
                node.feature_index = col
                node.threshold = thr
                node.left = self.learn(X_left, Y_left, depth + 1)
                node.right = self.learn(X_right, Y_right, depth + 1)
        return node

#    def classify(self, X):
#        """Prediction."""
#        return [self.predict_instance(inputs) for inputs in X]

    def classify(self, inputs):
        """Predict class for a single instance"""
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


def run_decision_tree():
    
   # Load data set
    with open("C:/Users/300063669/Downloads/MDS related Documents/FOML/Assignment 1/wine-dataset.csv") as f:
            next(f, None)
            data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print("Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.
   
#    training_set = [x for i, x in enumerate(data) if i % K != 9]
#    test_set= [x for i, x in enumerate(data) if i % K == 9]       
#    test_set=np.array(test_set).astype('float')
    
    # Creating cross validation sets k=10
    training_set_fold=[]
    validation_set_fold=[]
    X=[]
    Y=[]
    K = 10
     
    for a in range(K):
        print(a)
        training_set_fold.append(np.array([x for i, x in enumerate(data) if i % K !=a]))
        validation_set_fold.append(np.array([x for i, x in enumerate(data) if i % K ==a]))
        X.append(training_set_fold[a][:,:-1].astype('float'))
        Y.append(training_set_fold[a][:,-1].astype('int'))
           
   # Classify the test set using the tree we just constructed
    results=[ [ None for y in range(len(validation_set_fold[0])) ]
                                    for x in range(10)] 
    dtree = DecisionTree(max_depth=50)
     
    for x in range(len(X)):
        dtree.fit(X[x], Y[x])
        print("Fit done for",x)
        for y in range(len(validation_set_fold[x])):
            input=validation_set_fold[x][y,:-1].astype('float').tolist()
            result = dtree.classify(input)
            results[x][y]=(result == validation_set_fold[x][y,-1].astype('float'))
            print("Got result for",x,"-",y)
#        results[i]=(result, result == test_set[i,-1])
    
    # Accuracy
    accuracy=[]
    for x in range(len(results)):
        accuracy.append(float(results[x].count(True))/float(len(results[x])))
        print("accuracy: %.4f" % accuracy[x])
    
    print("final average accuracy",round(statistics.mean(accuracy),4)*100)
        
    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()

if __name__ == "__main__":
    run_decision_tree()
    