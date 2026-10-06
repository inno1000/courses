import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_wine().data)
X_sub = X[:30]

Z = linkage(X_sub, method="ward")
plt.figure(figsize=(10, 4))
dendrogram(Z, truncate_mode="level", p=5)
plt.title("Dendrogramme (30 premiers individus)")
plt.tight_layout()
plt.savefig(__file__.replace(".py", "_dendro.png"), dpi=120)

ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
print("ARI Ward vs vérité cultivar:", adjusted_rand_score(load_wine().target, ward).round(3))
print("ARI K-means vs Ward:", adjusted_rand_score(ward, km).round(3))
