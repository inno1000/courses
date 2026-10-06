# Support Master 2 — Data Mining (distribution étudiants)

Université de Ngaoundéré — **Pr Ndam Njoya Arouna**

## Par chapitre

| Élément | Fichier / dossier |
|--------|-------------------|
| Cours | `Chapitre_XX/Ch01_…pdf` … `Ch04_…pdf` (un PDF nommé par chapitre) |
| Exemples | `exemples/*.ipynb` **ou** `exemples/*.py` (au choix) |
| TD | `exercices/enonce.pdf`, `exercices/corriges.pdf` |
| Données / scripts TD | `exercices/correction_*.py` lorsque l’énoncé le demande |

## Environnement Python

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

(`requirements.txt` à la racine du projet lorsque vous recevez l’archive complète.)

## Regénération (enseignants)

Depuis la racine du dépôt source : `scripts\build_cours_pdf.ps1`
