"""Exemple Chapitre — équivalent script de `02_classification_metrics.ipynb`.
Exécutez ce fichier ou le notebook Jupyter selon votre préférence.
"""
# %% [markdown]
# # Chapitre 2 : classification et métriques
#
# Jeu `load_breast_cancer` : comparaison k-NN, régression logistique, arbre ; validation croisée et courbe ROC.
# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)
# %% [markdown]
# ## Comparaison des modèles (test + F1 en CV 5-fold)
# %%
models = {
    "kNN (k=5)": Pipeline(
        [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
    ),
    "LogReg": Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]
    ),
    "Arbre (prof. 4)": DecisionTreeClassifier(max_depth=4, random_state=42),
}

for name, est in models.items():
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))
    scores = cross_val_score(est, X_train, y_train, cv=5, scoring="f1")
    print(f"F1 CV (5-fold) mean={scores.mean():.3f} ± {scores.std():.3f}")
# %% [markdown]
# ## Courbe ROC (régression logistique)
# %%
logreg = models["LogReg"]
y_score = logreg.predict_proba(X_test)[:, 1]
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_score, ax=ax, name="LogReg")
ax.plot([0, 1], [0, 1], "k--", label="Hasard")
ax.set_title("Courbe ROC, Breast Cancer")
ax.legend()
plt.tight_layout()
plt.show()
