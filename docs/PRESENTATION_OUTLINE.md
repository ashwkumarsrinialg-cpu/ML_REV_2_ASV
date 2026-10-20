# Review 2 presentation outline (≈10–12 min + viva) — story arc: problem → data → method → results → conclusion
| # | Slide | Content / visual | Speaker |
|---|---|---|---|
| 1 | Title & problem | Team, dataset, 2 tracks covered, objective | M1 |
| 2 | Data recap | Shape, classes, key EDA insight (`clf_eda.png`) | M2 |
| 3 | Preprocessing | Stratified 80:20; scaler fit on train only; engineered feature & why | M2 |
| 4 | Classification — 10 algorithms | Part A (R1) + Part B (new); same split | M2 |
| 5 | Consolidated table | Full 10-row table; rank by F1; highlight ties & test-set size caveat | M2 |
| 6 | Visual evidence | Confusion-matrix grid / ROC-OvR / feature importance | M2 |
| 7 | Tuning & CV | Before/after table; 5-fold CV top-2 | M2 |
| 8 | Final model | Gradient Boosting choice, *selection on CV not test*, final confusion matrix; GUI demo (live) | M2/M1 |
| 9 | Clustering setup | Unsupervised; scaling; Elbow + silhouette → k=6 | M3 |
| 10 | Hierarchical | Dendrogram; linkage comparison (ward vs chaining) | M3 |
| 11 | Cluster metrics | Sil / DB / CH table; DB-metric trap | M3 |
| 12 | Visualisation | PCA (2-D) for both + t-SNE; cluster profiles & business segments | M3 |
| 13 | Conclusions | Winners, trade-offs, limitations, future work | M1 |
| 14 | Q&A | — | all |
**Rules of thumb:** one message per slide; every plot has a takeaway line; each member presents their own track; keep the notebook open as backup.
