"""Génère credit_sim.csv et entraîne une régression logistique."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(1)
n = 800
debt_ratio = rng.uniform(0, 0.9, n)
income = rng.lognormal(10, 0.35, n)
score = -2 + 3.5 * debt_ratio - 0.0003 * income + rng.normal(0, 0.5, n)
p = 1 / (1 + np.exp(-score))
default = rng.binomial(1, p)

df = pd.DataFrame(
    {"debt_ratio": debt_ratio, "income": income, "default": default}
)
df.to_csv(ROOT / "credit_sim.csv", index=False)

X = df[["debt_ratio", "income"]]
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000)),
    ]
)
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"AUC test = {auc:.3f}")
coef = pipe.named_steps["clf"].coef_[0]
print("Coefs (standardisés): debt_ratio=", round(coef[0], 3), " income=", round(coef[1], 3))
