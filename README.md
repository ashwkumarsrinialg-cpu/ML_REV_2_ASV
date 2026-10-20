# 23CSE301 Machine Learning: Capstone Project (Review 2)

**Team ASV_INTELLIGENCE** · Academic year 2026-27
**Live demo (bonus):** _http://localhost:8501/_**
| Member | Roll number |
|:--|:--|
| Ashwin Kumar Srinivasan | CB.SC.U4CSE24607 |
| Salla Sai Sumallik | CB.SC.U4CSE24647 |
| Tallada Vamshidhar | CB.SC.U4CSE24660 |

---

## 1. What this project is about

We worked on two questions for this review.

**Classification.** Given a person's census record (age, education, occupation, marital status, hours worked, capital gains and so on), can we predict whether they earn more than $50K a year? We trained and compared ten different classifiers to find out which one does it best.

**Clustering.** Given what we know about customers of a shopping mall (their age, annual income and spending score), are there natural groups of shoppers that a marketing team could target differently? There are no labels here, so the algorithms have to find the structure by themselves.

## 2. The data

| Track | Dataset | File | Size | Source |
|:--|:--|:--|:--|:--|
| Classification | Adult Income Census | `data/adult.csv` | 48,842 rows × 15 columns. About 76% earn ≤50K and 24% earn >50K | [Kaggle: uciml/adult-census-income](https://www.kaggle.com/datasets/uciml/adult-census-income) |
| Clustering | Mall Customer Segmentation | `data/Mall_Customers.csv` | 200 rows × 5 columns, no missing values or duplicates | [Kaggle: vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |

A few things we did to the census data before modelling:
- Missing values are written as `?` in `workclass`, `occupation` and `native_country`. We fill them with the most common category, and we do this inside the pipeline so it only learns from training data.
- We dropped `fnlwgt` (a survey sampling weight, not a property of the person) and `education` (it repeats `educational_num`).
- We added one feature of our own, `net_capital = capital_gain − capital_loss`.
- To keep tuning quick on a normal laptop, the notebook works on a stratified 8,000-row sample. If you want the full data, set `SAMPLE_N = None` in the first cells of `classification.ipynb`.

The copy of `adult.csv` in this repo has 48,842 rows. The Kaggle version has 32,561 rows and dotted column names; the notebook renames columns automatically, so either file works.

## 3. What is in this repository

```
README.md             you are here
requirements.txt      pinned library versions
data/                 adult.csv and Mall_Customers.csv
notebooks/            classification.ipynb and clustering.ipynb (fully executed)
models/               best_classifier.joblib (the saved final model)
figures/              all plots and result tables (CSV)
app/                  Streamlit demo (optional bonus) and deployment notes
```

## 4. How to run it

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                                           # open notebooks/ and choose Restart & Run All
streamlit run app/app.py                              # optional demo; run classification.ipynb first
```

Everything random uses `random_state=42`, so you should get the same numbers we did. We tested with Python 3.12 and scikit-learn 1.8. The classification notebook takes roughly 5 to 10 minutes on a laptop CPU (mostly the grid searches); the clustering one takes under a minute.

## 5. How we approached it

**Classification**
- We split the data 80:20 *before* doing anything else, stratified so both sets keep the same 76/24 balance.
- Imputing, scaling and one-hot encoding happen inside a pipeline, so they are learned from training data only, including inside every cross-validation fold. This avoids data leakage.
- Ten algorithms were trained on the same split. Part A: Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC. Part B: Random Forest, AdaBoost, Gradient Boosting, Bagging (decision-tree base), MLP.
- We report accuracy, precision, recall, F1 (weighted), ROC-AUC and confusion matrices. Because the classes are imbalanced, we look at weighted metrics and not just accuracy.
- Seven models were tuned with 5-fold GridSearchCV, the top two were cross-validated again, and the final model was chosen on cross-validation score, never on the test set.

**Clustering**
- The features are Age, Annual Income and Spending Score, standardised first because K-Means and hierarchical clustering are sensitive to scale. Gender is not used for fitting; we only use it afterwards to describe the clusters.
- K-Means: elbow curve and silhouette scores to choose k.
- Agglomerative: dendrograms and four linkage methods compared (ward, complete, average, single).
- Both are scored with Silhouette, Davies-Bouldin and Calinski-Harabasz, and visualised with PCA (both algorithms) and t-SNE.

## 6. What we found

### 6.1 Consolidated classification comparison (10 algorithms, held-out test set, n = 1,600)
| Algorithm | Part | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC | F1 after tuning |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Gradient Boosting | B | 0.8581 | 0.8518 | 0.8581 | 0.8515 | 0.9190 | 0.8540 |
| SVM (SVC) | A | 0.8512 | 0.8441 | 0.8512 | 0.8436 | 0.9007 | 0.8436 |
| Random Forest | B | 0.8406 | 0.8330 | 0.8406 | 0.8346 | 0.8950 | 0.8457 |
| Bagging (DT base) | B | 0.8356 | 0.8306 | 0.8356 | 0.8325 | 0.8876 | — |
| KNN | A | 0.8350 | 0.8301 | 0.8350 | 0.8320 | 0.8655 | 0.8357 |
| AdaBoost | B | 0.8394 | 0.8306 | 0.8394 | 0.8310 | 0.9031 | 0.8342 |
| Logistic Regression | A | 0.8362 | 0.8285 | 0.8362 | 0.8303 | 0.9015 | 0.8325 |
| MLP (Neural Net) | B | 0.8256 | 0.8231 | 0.8256 | 0.8243 | 0.8780 | 0.8478 |
| Decision Tree | A | 0.7994 | 0.8018 | 0.7994 | 0.8005 | 0.7379 | — |
| Gaussian Naive Bayes | A | 0.3175 | 0.7521 | 0.3175 | 0.2509 | 0.5888 | — |

**Final model:** Gradient Boosting (200 trees, depth 3, lr 0.1) — highest 5-fold CV weighted-F1 on the training set (0.850); test accuracy 0.859, weighted F1 0.854, AUC 0.921. Recall of the >50K minority class is only ≈ 0.61. Gaussian Naive Bayes fails (accuracy 0.32) because one-hot features violate its independence assumption.

### 6.2 Clustering (k = 6, chosen by max silhouette; elbow agrees at ≈ 5–6)
| Algorithm | k | Silhouette (↑) | Davies-Bouldin (↓) | Calinski-Harabasz (↑) | Cluster sizes |
|:--|--:|--:|--:|--:|:--|
| K-Means | 6 | 0.4284 | 0.8254 | 135.10 | [45, 39, 33, 39, 23, 21] |
| Agglomerative (ward) | 6 | 0.4201 | 0.8521 | 127.99 | [45, 45, 39, 28, 22, 21] |

K-Means and Ward agree strongly (ARI 0.89, NMI 0.90). Ward was the best linkage (single linkage chained into one giant cluster). Segments are described by income vs spending (e.g. high-income/high-spending, high-income/low-spending, low-income/high-spending).

## 7. Figures

All saved in `figures/`: `clf_confusion_all.png`, `clf_metric_bars.png`, `clf_roc_ovr.png`, `clf_rf_importance_tree.png`, `clu_elbow_silhouette.png`, `clu_dendrograms.png`, `clu_pca.png`, `clu_tsne.png`, `clu_profile.png`.

## 8. Limitations and what we would do next

- The classifier misses a lot of high earners (recall about 0.61 for >50K). Lowering the decision threshold or using class weights would be the first thing to try.
- The census data includes gender and race, so any real use of a model like this would need a careful fairness review.
- We used an 8,000-row sample for speed; results on the full data may shift slightly.
- The Mall dataset has only 200 customers, so the segments are indicative rather than definitive.
- We only used two clustering algorithms. DBSCAN or Gaussian Mixture Models would be natural additions.

## 9. Who did what

| Member | Main track |
|:--|:--|
| Ashwin Kumar Srinivasan (CB.SC.U4CSE24607) | Classification |
| Salla Sai Sumallik (CB.SC.U4CSE24647) | Clustering |
| Tallada Vamshidhar (CB.SC.U4CSE24660) | Regression (Review 1), app and documentation |

All three of us can explain any part of the project.

## 10. Credits and AI-usage declaration

- Libraries: scikit-learn, pandas, NumPy, SciPy, Matplotlib, Seaborn and Streamlit documentation.
- Datasets: UCI Adult / Kaggle (uciml) and the Mall Customer Segmentation dataset on Kaggle (vjchoudhary7).
- **AI disclosure (course rule 7.5):** an AI assistant (Claude, by Anthropic) was used for code scaffolding of the notebooks, the app and documentation templates. The analysis and interpretation in the notebooks are our own.
