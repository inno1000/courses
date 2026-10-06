import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
y_bin = (y == 0).astype(int)  # setosa vs reste

for k in [1, 3, 5, 11]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    scores = cross_validate(
        pipe,
        X,
        y_bin,
        cv=5,
        scoring={"acc": "accuracy", "f1": "f1"},
    )
    print(
        f"k={k:2d}  acc={scores['test_acc'].mean():.3f}±{scores['test_acc'].std():.3f}  "
        f"f1={scores['test_f1'].mean():.3f}±{scores['test_f1'].std():.3f}"
    )
