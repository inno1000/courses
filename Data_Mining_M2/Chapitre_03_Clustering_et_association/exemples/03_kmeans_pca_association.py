"""Exemple Chapitre — équivalent script de `03_kmeans_pca_association.ipynb`.
Exécutez ce fichier ou le notebook Jupyter selon votre préférence.
"""
# %% [markdown]
# # Chapitre 3 : K-means, ACP et règles d'association
#
# Partie 1 : jeu Wine (clustering + visualisation ACP). Partie 2 : paniers simulés (Apriori).
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# %% [markdown]
# ## K-means : coude, silhouette, ACP 2D
# %%
X, y = load_wine(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

inertias, sil = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, labels))

print("Silhouette par K:", dict(zip(K_range, [round(s, 3) for s in sil])))

pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X_scaled)
print("Variance expliquée (2 axes):", round(pca.explained_variance_ratio_.sum(), 3))

km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(list(K_range), inertias, "o-")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertie")
axes[0].set_title("Méthode du coude")
scatter = axes[1].scatter(X2[:, 0], X2[:, 1], c=km3.labels_, cmap="tab10", s=25, alpha=0.8)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("K-means (K=3) en plan ACP")
plt.colorbar(scatter, ax=axes[1], label="Cluster")
plt.tight_layout()
plt.show()
# %% [markdown]
# ## Règles d'association (mlxtend)
# %%
transactions = [
    ["pain", "lait", "beurre"],
    ["pain", "lait"],
    ["pain", "beurre", "oeufs"],
    ["lait", "cereales"],
    ["pain", "lait", "cereales"],
    ["beurre", "oeufs"],
    ["pain", "lait", "beurre", "cereales"],
    ["pain", "cereales"],
]
items = sorted({i for t in transactions for i in t})
basket = pd.DataFrame(
    [[int(it in t) for it in items] for t in transactions], columns=items
)
freq = apriori(basket, min_support=0.25, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.5)
rules = rules.sort_values("lift", ascending=False)
rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(5)
