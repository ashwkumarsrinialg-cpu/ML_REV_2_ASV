# Git plan — meaningful commit history (rubric C3)
Do these as **separate commits over several days / by different members** (each member commits their own track from their own account). Never upload everything in one commit.

```bash
git init && git branch -M main
git remote add origin https://github.com/<user>/<repo>.git

# 1 — scaffolding
git add .gitignore requirements.txt README.md && git commit -m "Add project scaffold, requirements and README skeleton"
git add data/ && git commit -m "Add raw dataset"

# 2 — classification Part B (owner: member 2)
git add notebooks/classification.ipynb && git commit -m "Add classification notebook: preprocessing + all 10 algorithms"
```
Realistic sequence (edit the notebook between commits, or stage cells progressively):
```
Add Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP (Part B)
Add consolidated 10-algorithm comparison table
Add GridSearchCV tuning for 7 models with before/after CV scores
Add 5-fold CV for top-2 models and final model selection
Save best classifier with joblib
Add confusion-matrix grid, ROC (OvR) and feature-importance plots

# clustering (owner: member 3)
Add clustering notebook: scaling + K-Means elbow & silhouette
Add hierarchical clustering: dendrograms and linkage comparison
Add Silhouette / Davies-Bouldin / Calinski-Harabasz evaluation table
Add PCA and t-SNE cluster visualisations
Add post-hoc interpretation (algorithm agreement, gender mix) and cluster profiles

# docs / deployment (any member)
Add Streamlit GUI app
Update README with results tables and run instructions
Add deployment instructions and live demo link
Add figures/ and results CSVs
```
**Tip:** use `git add -p` or work in small increments on feature branches (`feat/clustering`, `feat/classification`) and merge with PRs — it shows genuine collaboration.
Check with `git log --oneline --graph --all --format="%h %an %ad %s" --date=short`.
