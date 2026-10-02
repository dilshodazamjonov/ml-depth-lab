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

def _most_common_label(y):

    unique, counts = np.unique(y, return_counts=True)
    most_freq = unique[np.argmax(counts)]
                        
    return most_freq
            


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

    def _build_tree(self, X, y: NDArray, depth=0):
        """
        input: X,y - sample of the data and depth -> how many levels down there? 
        output: Node (either question(compare 0 and 40 then go left or right) or the entire Node)
        """        

        if depth == self.max_depth or len(X) < self.min_samples_split or y.min() == y.max(): 

            return Node(value=_most_common_label(y))

        best_gain, best_idx, best_threshold = self._best_split(X, y)

        if best_idx is None or best_gain < self.min_impurity_decrease:
            return Node(value=_most_common_label(y))

        mask = X[:, best_idx] < best_threshold

        X_left = X[mask]
        X_right = X[~mask]
        y_left, y_right = y[mask], y[~mask]

        left_child = self._build_tree(X_left, y_left, depth=depth+1)
        right_child = self._build_tree(X_right, y_right, depth=depth+1)

        return Node(best_idx, best_threshold, left_child, right_child)
