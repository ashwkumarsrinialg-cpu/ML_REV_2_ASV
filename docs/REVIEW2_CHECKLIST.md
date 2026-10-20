# Review 2 — Complete Guidelines & Submission Checklist
*Source: 23CSE301 Capstone Guidelines 2026-27. Review 2 = **Classification Part B + full Clustering track**, 25 marks (+ up to 2 bonus; display capped at 20 unless the instructor says otherwise). In-person presentation; **every member must be able to answer on any part.***

## A. Deliverables (D1–D8)
| # | Deliverable | Where it is in this repo | Done |
|---|---|---|---|
| D1 | Jupyter notebook(s) — clean, fully run, outputs visible | `notebooks/classification.ipynb`, `notebooks/clustering.ipynb` | ☐ |
| D2 | GitHub repo with meaningful commit history | see `docs/GIT_COMMIT_PLAN.md` | ☐ |
| D3 | Comparative results table (algorithms vs metrics) | classification §5/§8, clustering §4; CSVs in `figures/` | ☐ |
| D4 | All required visualisations | `figures/*.png` (see §C below) | ☐ |
| D5 | README (dataset description, how to run) | `README.md` | ☐ |
| D6 | Consolidated 10-algorithm classification comparison | classification notebook §5 + §8 | ☐ |
| D7 | Cluster visualisations (Elbow, Dendrogram, PCA) | clustering notebook §2, §3, §5 (+ t-SNE §6) | ☐ |
| D8 | GUI / deployed app (bonus) | `app/app.py`, `app/README_DEPLOY.md` | ☐ |

## B. Rubric self-audit (25 marks)
**Section A — Classification Part B (6)**
- ☐ A1 (3) All 5 Part-B algorithms (Random Forest, AdaBoost, Gradient Boosting, Bagging with DT base, MLP) trained on the **same dataset/split as Part A**, no errors
- ☐ A2 (1) ONE table: all 10 algorithms × Accuracy, Precision, Recall, F1, ROC-AUC
- ☐ A3 (2) Best model named with justification; hyperparameter tuning applied; improvement in ≥1 metric documented (before vs after)

**Section B — Clustering (8)**
- ☐ B1 (4) K-Means and Agglomerative both fitted; labels obtained; no errors; **no labels used in fitting**
- ☐ B2 (2) Silhouette, Davies-Bouldin, Calinski-Harabasz for **all** algorithms; Elbow curve (K-Means); Dendrogram (Hierarchical)
- ☐ B3 (2) PCA 2-D scatter, colour-coded, for **every** algorithm; t-SNE for ≥1

**Section C — Pipeline & quality (3)**
- ☐ C1 (1) Cells run top-to-bottom, modular functions, key steps commented
- ☐ C2 (1) README: dataset description, problem statement, results summary table, environment setup, how-to-run
- ☐ C3 (1) Meaningful commit history (NOT one bulk upload) + `requirements.txt`/`environment.yml`

**Section D — Presentation & viva (8)**
- ☐ D1 (3) Story arc problem → data → method → results → conclusion; clear visuals; organised
- ☐ D2 (5) All members show understanding: algorithm choice, metric interpretation, trade-offs

**Bonus (+2):** ☐ +1 working Streamlit/Gradio GUI · ☐ +1 public URL deployment

## C. Algorithm requirements (must match exactly)
**Part B classifiers:** Random Forest (feature importance; tune n_estimators) · AdaBoost (tune n_estimators & learning_rate) · Gradient Boosting (sklearn GBM or XGBoost/LightGBM) · Bagging (**Decision Tree as base estimator**) · MLP (tune hidden-layer sizes & activation).
**Metrics (all mandatory):** Accuracy, Precision, Recall, F1 (**weighted**), Confusion matrix, ROC-AUC (**One-vs-Rest** if multi-class).
**Clustering:** K-Means (Elbow: inertia vs k) · Agglomerative (Dendrogram; compare linkages). Metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz for all. PCA 2-D mandatory; t-SNE encouraged.

## D. General rules (mark-deduction risks)
**Code & notebooks**
1. One notebook per track (or one with clearly labelled sections); **Restart & Run All** before submitting; no execution errors.
2. Markdown cell explaining every major code block; no walls of code.
3. `random_state=42` everywhere applicable.
4. **Scalers/encoders fitted on TRAIN only** — fitting on the full dataset = data-leakage → deductions. (Clustering has no split; say so explicitly.)

**Evaluation**
5. Same train/test split (≈80:20, stratified) for every algorithm in a track.
6. Cross-validation for ≥ the two best models.
7. Results in ONE summary table (DataFrame / Markdown), not scattered prints.

**Visualisations**
8. Every plot: title, labelled axes, legend where applicable.
9. `plt.tight_layout()` or `savefig(..., bbox_inches='tight')`.
10. Colourblind-friendly palettes (`tab10`, `Set2`, `colorblind`).

**Team**
11. Each member owns ≥1 track and can explain every line of it; split **by track, not by step**. The teacher may ask anyone about anything.

**Academic integrity**
12. Cite external code (StackOverflow/blogs) in a Markdown cell.
13. Analysis, interpretation & feature-engineering decisions must be **original**; identical notebooks across teams = zero for both.
14. GenAI is allowed for code **scaffolding only**, not analysis/interpretation — **cite it in the README**.

**GitHub** — minimum structure: `README.md`, `requirements.txt` (with versions), `data/`, `notebooks/`, `models/` (optional), `app/` (if bonus). Commit at each milestone; a single bulk commit = **zero** for the GitHub criterion.

## E. Before-you-submit actions for THIS package
- ☑ Datasets set: Adult Income (classification, `data/adult.csv`, target `income`) and Mall Customers (clustering, `data/Mall_Customers.csv`). `app/app.py` updated for the new features.
- ☐ Use the **same dataset & split as your Part A (Review 1)** results.
- ☐ Re-run both notebooks (Restart & Run All) on your data; re-check every number quoted in markdown.
- ☐ **Rewrite every "DRAFT observation" cell in your own words** (rule 14) and fill team/problem-statement placeholders.
- ☐ Update README results tables + team contribution table + AI-usage line.
- ☐ Commit progressively (see `GIT_COMMIT_PLAN.md`), push, deploy (optional), paste URL.
- ☐ Rehearse the viva (`VIVA_PREP.md`) — all three members.
