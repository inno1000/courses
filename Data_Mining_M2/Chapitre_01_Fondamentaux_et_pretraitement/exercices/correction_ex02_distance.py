import numpy as np
from sklearn.preprocessing import StandardScaler

# Deux clients + échantillon de référence pour fit du scaler
rng = np.random.default_rng(7)
ref = np.column_stack(
    [rng.integers(20, 70, 500), rng.lognormal(5, 0.7, 500)]
)
a = np.array([[30, 100]])
b = np.array([[35, 5000]])

d_brut = np.linalg.norm(a - b)
scaler = StandardScaler().fit(ref)
a_s, b_s = scaler.transform(a), scaler.transform(b)
d_std = np.linalg.norm(a_s - b_s)

print("Client A: age=30, CA=100 k€  |  Client B: age=35, CA=5000 k€")
print(f"Distance euclidienne brute     : {d_brut:.2f}")
print(f"Distance après standardisation : {d_std:.2f}")
