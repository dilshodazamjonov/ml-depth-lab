"""
Binary Classification Decision Tree Implementation

Stuff to do: 

1. 

"""
import numpy as np
from numpy.typing import NDArray


def gini_impurity(y: NDArray):
    """
    helper for the decision tree classifier.

    1. split the y.nunique
    2. count how many sample belong to each class
    3. each -> propotion 1 − Σ p(i)²
    4. return the number
    """

    if len(y) == 0:
        return 0

    _, counts = np.unique(y, return_counts=True)

    g = 1 - np.sum((counts/len(y))**2)

    return g

gini_impurity(np.array([1, 0, 0, 1]))


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None



class DecisionTreeClassifierScratch:

    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0) -> None:

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

        self.root = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _best_split(self, X: NDArray, y: NDArray):
        """
        1. Loop over range(X.shape[1]) and get its sorted unique values 
        2. find the midpoints(adjacent pair of sorted numbers)
        3. see which midpoint is the best split(gini gain) 
        4. return that index, threshold and gini gain
        """
        parent_gini = gini_impurity(y)

        best_gain, best_idx, best_threshold = 0.0, None, None

        for feat_idx in range(X.shape[1]):
            X_col = X[:, feat_idx]
            X_sorted = np.unique(X_col)

            if len(X_sorted) == 1:
                continue

            mid_points = (X_sorted[1:] + X_sorted[:-1]) / 2

            for point in mid_points:
                mask = point <= X_col
                y_right = y[mask]
                y_left = y[~mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                right_gini, left_gini = gini_impurity(y_right), gini_impurity(y_left)
                weighted_gini = right_gini * len(y_right)/len(y) + left_gini * len(y_left)/len(y)

                gini_gain = parent_gini - weighted_gini

                if best_gain < gini_gain:
                    best_gain = gini_gain
                    best_idx = feat_idx
                    best_threshold = point

        return best_gain, best_idx, best_threshold
                
            
"""
Next step: the recursive build function

This is the method that actually grows the tree. It takes X, y, and the current depth, and returns a Node.

Part A: the leaf helper (write this first)

A small method that returns the most common label in y:

Use np.unique with counts, as in gini_impurity
Find the position of the largest count (NumPy has an "argmax" function for this)
Return the label at that position
Part B: stopping checks

At the start of the build function, return a leaf Node (using the helper) if any of these is true:

max_depth is set and the current depth has reached it
The number of samples is less than min_samples_split
The node is pure: y has only one unique label

For check 1, max_depth can be None, meaning no limit, so check for that before comparing.

Part C: split and recurse (we'll do this after A and B)

Once you've written Parts A and B, send them over and we'll go through the recursion together.
"""

        
