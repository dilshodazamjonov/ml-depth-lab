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

    def _build_tree(self, X, y, depth=0):
        pass


"""
For _build_tree, here's the order to work in.

1. The signature. It's a method on the class, taking X, y, and depth. Give depth a default value so fit can call it with just X and y.

2. The three checks. Write each as its own if, or combine them with or. Either works, but separate ifs are easier to debug while you're learning:

Max depth: first make sure max_depth isn't None, then compare depth to it. Remember the is not None point, so that max_depth=0 isn't treated as "no limit."
Too few samples: the number of samples is strictly less than min_samples_split.
Pure node: y has exactly one distinct label.

3. What each check returns. A Node whose value is the result of your most-common-label function, passed as a keyword argument.

4. Below the checks. Leave a comment like "Part C goes here" for now.
"""