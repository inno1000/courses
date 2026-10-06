import pandas as pd

df = pd.DataFrame(
    {
        "ROI": [0.12, 0.18, 0.10],
        "Risque": [0.3, 0.6, 0.2],
        "Delai": [6, 4, 9],
    },
    index=["A", "B", "C"],
)


def minmax(s, invert=False):
    lo, hi = s.min(), s.max()
    if hi == lo:
        z = s * 0 + 0.5
    else:
        z = (s - lo) / (hi - lo)
    return 1 - z if invert else z


n = df.copy()
n["ROI_n"] = minmax(df["ROI"])
n["Risque_n"] = minmax(df["Risque"], invert=True)
n["Delai_n"] = minmax(df["Delai"], invert=True)
n["U"] = 0.5 * n["ROI_n"] + 0.3 * n["Risque_n"] + 0.2 * n["Delai_n"]
print(n[["ROI_n", "Risque_n", "Delai_n", "U"]])
print("\nClassement:", n["U"].sort_values(ascending=False).index.tolist())
