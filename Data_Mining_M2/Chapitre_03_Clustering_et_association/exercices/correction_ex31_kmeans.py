from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_wine().data)
for k in range(2, 9):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    print(f"K={k}  inertie={KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_:.1f}  "
          f"silhouette={silhouette_score(X, labels):.3f}")
