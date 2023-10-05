import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from collections import Counter

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
    def __init__(self,max_depth=None):
        self.max_depth = max_depth
        self.root = None
        
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
    '''
    A class that implements Random Forest algorithm from scratch.
    '''
    def __init__(self, num_trees=25, min_samples_split=2, max_depth=5):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # Will store individually trained decision trees
        self.decision_trees = []
        
    @staticmethod
    def Sampling_data(X, Y):
        '''
        Helper function used for boostrap sampling.
        
        :param X: np.array, features
        :param y: np.array, target
        :return: tuple (sample of features, sample of target)
        '''
        n_rows, n_cols = X.shape
        # Sample with replacement
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[samples], Y[samples]
        
    def fit(self, X, Y):
        '''
        Trains a Random Forest classifier.
        
        :param X: np.array, features
        :param y: np.array, target
        :return: None
        '''
        # Reset
        if len(self.decision_trees) > 0:
            self.decision_trees = []
            
        # Build each tree of the forest
        num_built = 0
        while num_built < self.num_trees:
            try:
                clf = DecisionTree(
                    max_depth=self.max_depth
                )
                # Obtain data sample
                x1, y1 = self.Sampling_data(X,Y)
                # Train
                clf.fit(x1,y1)
                # Save the classifier
                self.decision_trees.append(clf)
                num_built += 1
                print(num_built)
            except Exception as e:
                continue
    
    def predict(self, X):
        '''
        Predicts class labels for new data instances.
        
        :param X: np.array, new instances to predict
        :return: 
        '''
        # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.classify(X))
        
        # Reshape so we can find the most common value
        y = np.swapaxes(a=y, axis1=0, axis2=1)
        
        # Use majority voting for the final prediction
        predictions = []
        for preds in y:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        return predictions
 
if __name__ == "__main__":
    
    iris = load_iris()
    X = iris['data']
    Y = iris['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = RandomForest()
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)

    accuracy_score(Y_test, preds)*100

#### Sklearn classifier #####
    sk_model = RandomForestClassifier()
    sk_model.fit(X_train, Y_train)
    sk_preds = sk_model.predict(X_test)

    accuracy_score(Y_test, sk_preds)