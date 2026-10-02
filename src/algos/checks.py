import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from .ml_algos import DecisionTreeClassifierScratch, Node


def fitted_root(model: DecisionTreeClassifierScratch) -> Node:
    assert model.root is not None
    return model.root


X_sep = np.arange(10, dtype=float).reshape(-1, 1)
y_sep = (X_sep[:, 0] >= 5).astype(int)

t = DecisionTreeClassifierScratch().fit(X_sep, y_sep)
root = fitted_root(t)
assert root.left is not None and root.right is not None
assert root.left.is_leaf() and root.right.is_leaf()
assert (t.predict(X_sep) == y_sep).all()

assert fitted_root(DecisionTreeClassifierScratch(max_depth=0).fit(X_sep, y_sep)).is_leaf()
assert fitted_root(DecisionTreeClassifierScratch().fit([[1.0], [1.0]], [0, 1])).is_leaf()

X, y = make_classification(n_samples=400, n_features=6, n_informative=4, random_state=0)
mine = DecisionTreeClassifierScratch(max_depth=5).fit(X[:300], y[:300])
ref = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X[:300], y[:300])
print("scratch:", (mine.predict(X[300:]) == y[300:]).mean())
print("sklearn:", (ref.predict(X[300:]) == y[300:]).mean())