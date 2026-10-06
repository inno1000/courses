"""Exemple Chapitre — équivalent script de `01_audit_et_pretraitement.ipynb`.
Exécutez ce fichier ou le notebook Jupyter selon votre préférence.
"""
# %% [markdown]
# # Chapitre 1 : audit qualité et pipeline de prétraitement
#
# Objectifs : simuler un jeu clients, auditer la qualité, construire un `ColumnTransformer` + pipeline scikit-learn.
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)
n = 200
df = pd.DataFrame(
    {
        "age": np.random.randint(22, 65, n).astype(float),
        "revenu": np.random.lognormal(10, 0.4, n),
        "secteur": np.random.choice(["Tech", "Retail", "Sante"], n),
        "score_interne": np.random.uniform(0, 100, n),
    }
)
mask = np.random.choice([True, False], n, p=[0.08, 0.92])
df.loc[mask, "revenu"] = np.nan
df.loc[0, "revenu"] = df["revenu"].max() * 5
df.head()
# %% [markdown]
# ## Audit descriptif
# %%
print("Dimensions:", df.shape)
print("\nTaux de manquants (%):")
print((df.isna().mean() * 100).round(2))
print("Doublons:", df.duplicated().sum())
df.describe(include="all")
# %% [markdown]
# ## Pipeline : imputation, encodage, standardisation
# %%
num_cols = ["age", "revenu", "score_interne"]
cat_cols = ["secteur"]

numeric_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ]
)

X = preprocessor.fit_transform(df)
print("Matrice transformée:", X.shape)
print("Moyenne (≈0) colonnes numériques:", X[:, :3].mean(axis=0).round(4))
print("Écart-type (≈1):", X[:, :3].std(axis=0).round(4))
