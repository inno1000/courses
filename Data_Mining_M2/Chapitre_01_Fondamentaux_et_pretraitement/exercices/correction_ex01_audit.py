"""Corrigé exercice 1.1 — génère clients.csv et exécute l'audit."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "clients.csv"

rng = np.random.default_rng(0)
n = 150
noms = [f"Client_{i}" for i in range(n)]
emails = [f"u{i}@mail.fr" for i in range(n)]
emails[5] = emails[4]  # doublon volontaire
noms[5] = noms[4]

df = pd.DataFrame(
    {
        "nom": noms,
        "email": emails,
        "montant_mensuel": rng.lognormal(5, 0.6, n),
        "ville": rng.choice(["Paris", "Lyon", "Marseille"], n),
    }
)
df.loc[rng.choice(n, 12, replace=False), "montant_mensuel"] = np.nan
df.to_csv(OUT, index=False)

print("Fichier écrit:", OUT)
print("\n--- Manquants (%) ---")
print((df.isna().mean() * 100).round(2))
print("\n--- Doublons nom+email ---", df.duplicated(subset=["nom", "email"]).sum())

col = df["montant_mensuel"].dropna()
q1, q3 = col.quantile(0.25), col.quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df["montant_mensuel"].apply(lambda x: x < low or x > high if pd.notna(x) else False)
print(f"\n--- Outliers IQR ({low:.2f}, {high:.2f}) ---", outliers.sum())
