import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

myname = "srijit sen"
path = "D:/MDS related Documents/FOML/Assignment 3/Q5/"
os.chdir(path)

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
    def __init__(self,max_depth=None,max_features=None):
        self.max_depth = max_depth
        self.root = None
        self.max_features=max_features
        
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
        col_samples = np.random.choice(a=self.n_features, size=self.max_features, replace=False)
        
        # Loop through all features.
        for col in col_samples:
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
        if len(Y): 
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

    def classify(self, X):
        """Prediction."""
        return [self.predict_instance(inputs) for inputs in X]

    def predict_instance(self, inputs):
        """Predict class for a single instance"""
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


class RandomForest:
    
    def __init__(self, num_trees=25, min_samples_split=2, max_depth=5,max_features=None):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features=max_features
        self.decision_trees = []
        
    @staticmethod    
    def Sampling_data(X, Y):
        '''   boostrap sampling   '''
        n_rows, n_cols = X.shape
        # Sample with replacement
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
        
        ## collecting out of bag samples ###
        oob_x_samples=np.delete(X, samples, axis=0)
        oob_y_samples=np.delete(Y, samples, axis=0)
        return X[samples], Y[samples],oob_x_samples,oob_y_samples
        
    def fit(self, X, Y):
        '''  Trains a Random Forest classifier    '''
        oob_x=[]
        oob_y=[]
        
        if len(self.decision_trees) > 0:
            self.decision_trees = []
            
        # Building forest
        num_built = 0
        while num_built < self.num_trees:
            try:
                clf = DecisionTree(
                    max_depth=self.max_depth,
                    max_features=self.max_features
                )
                # Obtain data sample
                x1, y1,oob_x_samples,oob_y_samples = self.Sampling_data(X,Y)
                # Train
                clf.fit(x1,y1)
                # Save the classifier
                self.decision_trees.append(clf)
                ## collecting out of bag samples ###
                oob_y.append(oob_y_samples)
                oob_x.append(oob_x_samples)
                
                num_built += 1
                print(num_built)
            except Exception as e:
                continue
        return oob_x,oob_y
    
    def predict(self, X,oob_x):
        ''' Prediction of given cases  '''
        y = []
        oob_pred=[]
        
        for tree in self.decision_trees:
            y.append(tree.classify(X))
            oob_pred.append(tree.classify(oob_x))
        # Swapping axes to find the common value
        y = np.swapaxes(a=y, axis1=0, axis2=1)
        oob_pred = np.swapaxes(a=oob_pred, axis1=0, axis2=1)
        
        # final prediction though majority voting
        predictions = []
        oob_predictions=[]
        
        ## prediction of test cases ##
        for preds in y:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        ## prediction of oob cases ##
        for pred in oob_pred:
            counter = Counter(pred)
            oob_predictions.append(counter.most_common(1)[0][0])
        return predictions,oob_predictions

if __name__ == "__main__":
#### Loading dataset #######    
    Data=np.genfromtxt("spam.data") 
    X = Data[:,:-1]
    Y = Data[:,-1]
    Y=Y.astype('int')

#### Splitting data into training and test #######    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

####### Own Classifier #######
    model = RandomForest(num_trees=10,max_features=10)
    import time
    start_time = time.time()
    oob_x,oob_y=model.fit(X_train, Y_train)
    oob_x1=np.concatenate( oob_x, axis=0 )
    oob_y1=np.concatenate( oob_y, axis=0 )
    preds,oob_preds = model.predict(X_test,oob_x1)

#### Accuracy calulation ###
    accuracy_score(Y_test, preds)*100  # 94.19
    accuracy_score(oob_y1, oob_preds)*100 #99.081
    print("--- %s seconds ---" % (time.time() - start_time)) #150 seconds

#### Sklearn classifier #####
    start_time = time.time()
    sk_model = RandomForestClassifier(n_estimators=25)
    sk_model.fit(X_train, Y_train)
    sk_preds = sk_model.predict(X_test)

    accuracy_score(Y_test, sk_preds)*100 #95.87
    print("--- %s seconds ---" % (time.time() - start_time)) # 0.14 seconds
    
'''Solution(a) 93.19% vs 95.87% in terms of accuracy and 150 seconds vs 0.14 seconds in terms of run time'''
'''Solution(b) THere is no change in accuracy with increase in the max_featues parameter'''
'''Solution(c) THere is no change in accuracy with increase in the max_featues parameter'''
    