import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
y_bin = (y == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.3, random_state=0, stratify=y_bin
)

depths = range(1, 11)
cv_scores, train_scores = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    cv_scores.append(cross_val_score(clf, X_train, y_train, cv=5).mean())
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))

plt.figure(figsize=(7, 4))
plt.plot(list(depths), train_scores, "o-", label="Train accuracy")
plt.plot(list(depths), cv_scores, "s-", label="CV accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Arbre — sur-apprentissage si écart train >> CV")
plt.tight_layout()
plt.savefig(__file__.replace(".py", ".png"), dpi=120)
print("Graphique enregistré.")
