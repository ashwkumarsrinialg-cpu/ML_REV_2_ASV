"""Generates notebooks/classification.ipynb and notebooks/clustering.ipynb (run once; then execute)."""
import nbformat as nbf, json, sys, os
from nbformat.v4 import new_notebook, new_markdown_cell as M, new_code_cell as C

OBS = json.load(open(os.path.join(os.path.dirname(__file__), "observations.json"))) if os.path.exists(os.path.join(os.path.dirname(__file__), "observations.json")) else {}
def obs(k): return M(OBS.get(k, "**Observation:** _(write your own interpretation here)_"))

SETUP = '''import warnings, time, joblib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12, "axes.labelsize": 10})
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)'''

CLF_CONFIG = '''# ---- DATASET CONFIG ----
# Adult Income Census (Kaggle: uciml/adult-census-income) -> predict whether income is >50K or <=50K
DATA_PATH = "../data/adult.csv"
TARGET    = "income"
SAMPLE_N  = 8000      # stratified subsample so GridSearch/SVC/MLP finish quickly on a laptop. Set to None to use ALL rows.
df = pd.read_csv(DATA_PATH, skipinitialspace=True)
# Kaggle's version uses dots in names (marital.status); other copies use hyphens -> normalise to underscores
df.columns = [c.strip().replace(".", "_").replace("-", "_") for c in df.columns]
df = df.replace("?", np.nan)                       # '?' marks missing values in this dataset
df[TARGET] = df[TARGET].astype(str).str.strip().str.rstrip(".").map({"<=50K": 0, ">50K": 1}).astype(int)   # 0 = <=50K, 1 = >50K
df = df.drop(columns=[c for c in ["fnlwgt"] if c in df.columns])   # census sampling weight: not a person attribute -> dropped
df = df.drop(columns=[c for c in ["education"] if c in df.columns]) # duplicate of educational_num (ordinal version kept)
print("Loaded:", DATA_PATH, "| full shape:", df.shape)
if SAMPLE_N and SAMPLE_N < len(df):
    from sklearn.model_selection import train_test_split
    df, _ = train_test_split(df, train_size=SAMPLE_N, stratify=df[TARGET], random_state=RANDOM_STATE); df = df.reset_index(drop=True)
    print("Stratified subsample used:", df.shape)'''

CLU_CONFIG = '''# ---- DATASET CONFIG ----
# Mall Customer Segmentation (Kaggle: vjchoudhary7/customer-segmentation-tutorial-in-python) -> unsupervised, NO label column
DATA_PATH = "../data/Mall_Customers.csv"
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"Annual Income (k$)": "Annual_Income_k", "Spending Score (1-100)": "Spending_Score"})
df = df.drop(columns=["CustomerID"])               # identifier, carries no information
print("Loaded:", DATA_PATH, "| shape:", df.shape)
display(df.head())'''

# =====================================================================  CLASSIFICATION
cl = []
cl.append(M("""# 23CSE301 Machine Learning — Capstone, Review 2
## Classification Track (Part A + Part B) — Consolidated 10-Algorithm Comparison

**Team:** ASV_INTELLIGENCE  \n**Members:** Ashwin Kumar Srinivasan (CB.SC.U4CSE24607), Salla Sai Sumallik (CB.SC.U4CSE24647), Tallada Vamshidhar (CB.SC.U4CSE24660)  \n**Track owner:** <owner>

**Dataset:** Adult Income Census (UCI / Kaggle `uciml/adult-census-income`).\n\n**Problem statement:** Predict whether a person earns **more than $50K per year** (binary classification) from census attributes such as age, education, occupation, marital status, capital gain/loss and hours worked per week. Such a model can support income-bracket analysis; it must be used carefully because features like gender and race may encode societal bias.

**Pipeline:** Audit → EDA → Preprocessing (leak-free) → Train all 10 algorithms (Part A: 1–5, Part B: 6–10) → Consolidated metrics table → Hyperparameter tuning (GridSearchCV) → 5-fold CV on top-2 → Final model selection → Save model.

> *Citations:* scikit-learn documentation used for API reference. If code from external sources / GenAI scaffolding was used, cite it here and in the README (course rule 7.5)."""))
cl.append(C(SETUP))
cl.append(M("## 1. Dataset loading & audit"))
cl.append(C(CLF_CONFIG))
cl.append(C('''print("Shape:", df.shape)
display(df.dtypes.to_frame("dtype").T)
audit = pd.DataFrame({"missing": df.isna().sum(), "missing_%": (df.isna().mean()*100).round(2), "n_unique": df.nunique()})
display(audit.T)
print("Duplicate rows:", df.duplicated().sum())
print("\\nClass distribution:")
display(pd.concat([df[TARGET].value_counts().rename("count"), (df[TARGET].value_counts(normalize=True)*100).round(1).rename("%")], axis=1))'''))
cl.append(M("## 2. Exploratory Data Analysis (summary — full EDA was presented in Review 1)"))
cl.append(C('''fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.countplot(x=TARGET, data=df, ax=axes[0], hue=TARGET, legend=False)
axes[0].set_title("Target (class) distribution"); axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
corr = df.drop(columns=TARGET).select_dtypes("number").corr()
sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[1], cbar_kws={"label": "Pearson r"})
axes[1].set_title("Feature correlation heatmap")
plt.tight_layout(); plt.savefig("../figures/clf_eda.png", bbox_inches="tight"); plt.show()'''))
cl.append(obs("clf_eda"))
cl.append(M("## 3. Preprocessing (no data leakage)\nStratified 80:20 split **first**. Imputation, scaling and one-hot encoding live in a `ColumnTransformer` that is fitted on the **training set only** (inside a `Pipeline`, so every CV fold is leak-free too). Missing values (`?` in workclass, occupation, native_country) are imputed with the most frequent category.\n\n**Engineered feature:** `net_capital = capital_gain - capital_loss` (a single signed measure of investment income; derived row-wise so it cannot leak)."))
cl.append(C('''from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df["net_capital"] = df["capital_gain"] - df["capital_loss"]          # engineered feature (row-wise)
X = df.drop(columns=TARGET); y = df[TARGET]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("Numeric:", num_cols); print("Categorical:", cat_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)])
def pipe(est): return Pipeline([("prep", make_preprocessor()), ("model", est)])     # preprocessing is re-fitted inside every fit/CV fold

classes = np.sort(y.unique()); n_classes = len(classes)
print(f"Train: {X_train.shape}  Test: {X_test.shape}  Classes: {n_classes}")
print("Train class %:", (y_train.value_counts(normalize=True).sort_index()*100).round(1).to_dict())
print("Test  class %:", (y_test.value_counts(normalize=True).sort_index()*100).round(1).to_dict())'''))
cl.append(M("## 4. Define all 10 classification algorithms\nPart A = models 1–5 (Review 1), Part B = models 6–10 (Review 2). All use the **same** split and `random_state=42`."))
cl.append(C('''from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier

models = {
    # ---- Part A ----
    "Logistic Regression":   ("A", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    "KNN":                   ("A", KNeighborsClassifier(n_neighbors=5)),
    "Gaussian Naive Bayes":  ("A", GaussianNB()),
    "Decision Tree":         ("A", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    "SVM (SVC)":             ("A", SVC(probability=True, random_state=RANDOM_STATE)),
    # ---- Part B ----
    "Random Forest":         ("B", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    "AdaBoost":              ("B", AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    "Gradient Boosting":     ("B", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    "Bagging (DT base)":     ("B", BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=100, random_state=RANDOM_STATE)),
    "MLP (Neural Net)":      ("B", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=RANDOM_STATE)),
}
print(len(models), "algorithms defined")'''))
cl.append(M("## 5. Train & evaluate all 10 algorithms (same test set)\nMetrics: Accuracy, Precision / Recall / F1 (**weighted**), ROC-AUC (**One-vs-Rest** for multi-class), confusion matrix."))
cl.append(C('''from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate(model, Xtr, ytr, Xte, yte):
    t0 = time.time(); model.fit(Xtr, ytr); fit_s = time.time() - t0
    pred = model.predict(Xte); proba = model.predict_proba(Xte)
    auc = roc_auc_score(yte, proba[:, 1]) if n_classes == 2 else roc_auc_score(yte, proba, multi_class="ovr", average="weighted")
    return {"Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, average="weighted", zero_division=0),
            "Recall": recall_score(yte, pred, average="weighted", zero_division=0),
            "F1": f1_score(yte, pred, average="weighted"),
            "ROC-AUC (OvR)": auc, "Fit time (s)": fit_s}, confusion_matrix(yte, pred, labels=classes)

rows, cms = [], {}
for name, (part, mdl) in list(models.items()):
    mdl = pipe(mdl); models[name] = (part, mdl)   # wrap estimator with the leak-free preprocessor
    res, cm = evaluate(mdl, X_train, y_train, X_test, y_test)
    rows.append({"Algorithm": name, "Part": part, **res}); cms[name] = cm
results = pd.DataFrame(rows).set_index("Algorithm")
results_ranked = results.sort_values(["F1", "ROC-AUC (OvR)"], ascending=False)
results_ranked.insert(0, "Rank", range(1, len(results_ranked) + 1))
print("CONSOLIDATED 10-ALGORITHM COMPARISON (untuned, default / baseline hyperparameters, same 80:20 test split)")
display(results_ranked.round(4).style.background_gradient(cmap="Greens", subset=["Accuracy","Precision","Recall","F1","ROC-AUC (OvR)"]).format(precision=4))
results_ranked.round(4).to_csv("../figures/classification_comparison_untuned.csv")'''))
cl.append(obs("clf_table"))
cl.append(C('''fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
for ax, (name, cm) in zip(axes.ravel(), cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=classes, yticklabels=classes)
    ax.set_title(f"{name} (Part {models[name][0]})"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.suptitle("Confusion matrices — all 10 algorithms (test set)", y=1.02, fontsize=14)
plt.tight_layout(); plt.savefig("../figures/clf_confusion_all.png", bbox_inches="tight"); plt.show()'''))
cl.append(C('''ax = results_ranked[["Accuracy","Precision","Recall","F1","ROC-AUC (OvR)"]].plot(kind="bar", figsize=(14, 5), width=0.85)
ax.set_title("Metric comparison across 10 classifiers (test set)"); ax.set_ylabel("Score"); ax.set_xlabel("Algorithm")
ax.set_ylim(min(0.8, results_ranked[["Accuracy","F1"]].min().min() - 0.05), 1.02)
plt.xticks(rotation=35, ha="right"); plt.legend(loc="lower right", ncol=5, title="Metric")
plt.tight_layout(); plt.savefig("../figures/clf_metric_bars.png", bbox_inches="tight"); plt.show()'''))
cl.append(obs("clf_bars"))
cl.append(M("### 5.1 Model-specific interpretation visuals (required by the algorithm notes)"))
cl.append(C('''fig, axes = plt.subplots(1, 2, figsize=(15, 5))
rf_pipe = models["Random Forest"][1]; feat_names = rf_pipe.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=feat_names).sort_values().tail(15)   # top 15 of the encoded features
imp.plot(kind="barh", ax=axes[0], color=sns.color_palette("colorblind")[0])
axes[0].set_title("Random Forest — top-15 feature importances"); axes[0].set_xlabel("Importance"); axes[0].set_ylabel("Feature")
dt = models["Decision Tree"][1].named_steps["model"]
plot_tree(dt, feature_names=feat_names, class_names=[str(c) for c in classes], filled=True, max_depth=3, fontsize=6, ax=axes[1])
axes[1].set_title("Decision Tree (top 3 levels shown)")
plt.tight_layout(); plt.savefig("../figures/clf_rf_importance_tree.png", bbox_inches="tight"); plt.show()'''))
cl.append(C('''# ROC curves of ALL 10 models (binary problem: positive class = income >50K)
from sklearn.metrics import roc_curve, auc
fig, ax = plt.subplots(figsize=(7, 6))
for name, (part, mdl) in models.items():
    fpr, tpr, _ = roc_curve(y_test, mdl.predict_proba(X_test)[:, 1]); ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Chance"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves — all classifiers (positive class: income >50K)"); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig("../figures/clf_roc_ovr.png", bbox_inches="tight"); plt.show()'''))
cl.append(M("## 6. Hyperparameter tuning (GridSearchCV, 5-fold stratified, scoring = weighted F1)\nTuning is done **inside a Pipeline** (scaler + model) on the **training set only**, so CV folds never see leaked scaling statistics. The test set is used once, for reporting."))
cl.append(C('''from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grids = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), {"model__C": [0.1, 1, 10]}),
    "KNN":                 (KNeighborsClassifier(), {"model__n_neighbors": [5, 11, 21], "model__metric": ["euclidean", "manhattan"]}),
    "SVM (SVC)":           (SVC(probability=True, random_state=RANDOM_STATE), {"model__C": [1, 10], "model__kernel": ["rbf"]}),
    "Random Forest":       (RandomForestClassifier(random_state=RANDOM_STATE), {"model__n_estimators": [100, 300], "model__max_depth": [None, 10, 20]}),
    "AdaBoost":            (AdaBoostClassifier(random_state=RANDOM_STATE), {"model__n_estimators": [50, 200], "model__learning_rate": [0.5, 1.0]}),
    "Gradient Boosting":   (GradientBoostingClassifier(random_state=RANDOM_STATE), {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}),
    "MLP (Neural Net)":    (MLPClassifier(max_iter=500, early_stopping=True, random_state=RANDOM_STATE), {"model__hidden_layer_sizes": [(32,), (64, 32)], "model__activation": ["relu", "tanh"], "model__alpha": [1e-4, 1e-2]}),
}
tuned, tune_rows = {}, []
for name, (est, grid) in grids.items():
    base_cv = cross_val_score(pipe(est), X_train, y_train, cv=skf, scoring="f1_weighted").mean()   # default hyperparameters
    gs = GridSearchCV(pipe(est), grid, cv=skf, scoring="f1_weighted", n_jobs=-1).fit(X_train, y_train)
    pred = gs.predict(X_test); tuned[name] = gs
    tune_rows.append({"Algorithm": name, "CV F1 (default)": base_cv, "CV F1 (tuned)": gs.best_score_, "CV gain": gs.best_score_ - base_cv,
                      "Test F1 (tuned)": f1_score(y_test, pred, average="weighted"), "Test Acc (tuned)": accuracy_score(y_test, pred),
                      "Best params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}})
tune_df = pd.DataFrame(tune_rows).set_index("Algorithm").sort_values("CV F1 (tuned)", ascending=False)
display(tune_df.round(4)); tune_df.round(4).to_csv("../figures/classification_tuning.csv")'''))
cl.append(obs("clf_tuning"))
cl.append(M("## 7. 5-fold cross-validation for the two best models"))
cl.append(C('''top2 = tune_df.index[:2].tolist()
cv_rows = []
for name in top2:
    best_est = tuned[name].best_estimator_
    for metric in ["accuracy", "f1_weighted"]:
        s = cross_val_score(best_est, X_train, y_train, cv=skf, scoring=metric)
        cv_rows.append({"Algorithm": name, "Metric": metric, "Fold scores": np.round(s, 3).tolist(), "Mean": s.mean(), "Std": s.std()})
cv_df = pd.DataFrame(cv_rows); display(cv_df.round(4)); cv_df.round(4).to_csv("../figures/classification_cv_top2.csv", index=False)'''))
cl.append(M("## 8. Final consolidated table — untuned vs tuned (all 10 algorithms)\nAlgorithms with no tuning grid (Naive Bayes, Decision Tree, Bagging) keep their baseline results."))
cl.append(C('''final = results[["Part","Accuracy","Precision","Recall","F1","ROC-AUC (OvR)"]].copy()
final.columns = ["Part"] + [f"{c} (base)" for c in final.columns[1:]]
tuned_metrics = {}
for name, gs in tuned.items():
    pred, proba = gs.predict(X_test), gs.predict_proba(X_test)
    tuned_metrics[name] = {"Accuracy (tuned)": accuracy_score(y_test, pred), "Precision (tuned)": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall (tuned)": recall_score(y_test, pred, average="weighted", zero_division=0), "F1 (tuned)": f1_score(y_test, pred, average="weighted"),
        "ROC-AUC (tuned)": roc_auc_score(y_test, proba[:, 1]) if n_classes == 2 else roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")}
final = final.join(pd.DataFrame(tuned_metrics).T)
final = final.sort_values("F1 (base)", ascending=False)
display(final.round(4)); final.round(4).to_csv("../figures/classification_final_table.csv")'''))
cl.append(M("## 9. Final model selection\nSelection criterion: **highest 5-fold CV weighted-F1 on the training set** (never the test set, to avoid selecting on the data we report on). The test set is then used once for the final report."))
cl.append(C('''best_name = tune_df["CV F1 (tuned)"].idxmax()
best_model = tuned[best_name].best_estimator_
pred = best_model.predict(X_test)
from sklearn.metrics import classification_report
print(f"FINAL MODEL: {best_name}\\nBest params: {tune_df.loc[best_name, 'Best params']}")
print(f"CV F1 (tuned) = {tune_df.loc[best_name,'CV F1 (tuned)']:.4f}  |  CV gain vs default = {tune_df.loc[best_name,'CV gain']:+.4f}")
print("\\nHeld-out test report:\\n", classification_report(y_test, pred, digits=3))
fig, ax = plt.subplots(figsize=(4.8, 4))
sns.heatmap(confusion_matrix(y_test, pred, labels=classes), annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_title(f"Final model: {best_name}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("../figures/clf_final_confusion.png", bbox_inches="tight"); plt.show()

# Save for the GUI / deployment
joblib.dump({"model": best_model, "features": X_train.columns.tolist(), "classes": classes.tolist(), "class_labels": {0: "<=50K", 1: ">50K"},
             "model_name": best_name, "num_cols": num_cols, "cat_cols": cat_cols,
             "num_stats": X_train[num_cols].describe().loc[["min", "mean", "max"]].to_dict(),
             "cat_levels": {c: sorted(X_train[c].dropna().unique().tolist()) for c in cat_cols}}, "../models/best_classifier.joblib")
print("Saved ../models/best_classifier.joblib")'''))
cl.append(obs("clf_final"))
cl.append(M("""## 10. Conclusions, limitations and trade-offs
_(Write/confirm in your own words — see the viva-prep notes in `docs/VIVA_PREP.md` for prompts.)_ Cover: which model won and why; interpretability vs accuracy trade-off; test-set size (one misclassification ≈ a fixed % change in accuracy, so small gaps between models are not statistically meaningful); cost of training; what you would do next."""))

# =====================================================================  CLUSTERING
cu = []
cu.append(M("""# 23CSE301 Machine Learning — Capstone, Review 2
## Clustering Track — K-Means & Agglomerative Hierarchical Clustering

**Team:** ASV_INTELLIGENCE  \n**Members:** Ashwin Kumar Srinivasan (CB.SC.U4CSE24607), Salla Sai Sumallik (CB.SC.U4CSE24647), Tallada Vamshidhar (CB.SC.U4CSE24660)  \n**Track owner:** <owner>

**Dataset:** Mall Customer Segmentation (Kaggle `vjchoudhary7/customer-segmentation-tutorial-in-python`) — 200 customers with Age, Annual Income (k$), Spending Score (1–100) and Gender.

**Goal:** discover natural customer segments for targeted marketing. Clustering is **unsupervised** and this dataset has **no ground-truth label**, so quality is judged with internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and by interpreting the cluster profiles. `Gender` is excluded from fitting and used only afterwards to describe the clusters."""))
cu.append(C(SETUP))
cu.append(M("## 1. Load data & scale\nDistance-based clustering is scale-sensitive, so features are standardised. There is no train/test split in clustering, so the scaler is fitted on the feature matrix once."))
cu.append(C(CLU_CONFIG))
cu.append(C('''from sklearn.preprocessing import StandardScaler
FEATURES = ["Age", "Annual_Income_k", "Spending_Score"]       # numeric behavioural features used for clustering
gender = df["Gender"].values                                    # kept aside for post-hoc profiling only — NOT used for fitting
X = df[FEATURES]
Xs = StandardScaler().fit_transform(X)
print("Feature matrix:", Xs.shape, "| missing:", int(X.isna().sum().sum()), "| duplicates:", int(df.duplicated().sum()))
display(X.describe().T.round(2))'''))
cu.append(M("## 2. K-Means — choosing k (Elbow curve + silhouette)"))
cu.append(C('''from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score

ks = range(2, 11); inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(list(ks), inertia, "o-"); axes[0].set_title("K-Means Elbow curve"); axes[0].set_xlabel("Number of clusters k"); axes[0].set_ylabel("Inertia (WCSS)")
axes[1].plot(list(ks), sil, "s-", color=sns.color_palette("colorblind")[1]); axes[1].set_title("Silhouette score vs k"); axes[1].set_xlabel("Number of clusters k"); axes[1].set_ylabel("Mean silhouette")
K = list(ks)[int(np.argmax(sil))]
axes[1].axvline(K, ls="--", color="grey", label=f"best k = {K}"); axes[1].legend()
plt.tight_layout(); plt.savefig("../figures/clu_elbow_silhouette.png", bbox_inches="tight"); plt.show()
print("k with max silhouette:", K)'''))
cu.append(obs("clu_elbow"))
cu.append(M("## 3. Agglomerative Hierarchical Clustering — dendrogram & linkage comparison"))
cu.append(C('''from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
d = pdist(Xs)
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for ax, method in zip(axes.ravel(), ["ward", "complete", "average", "single"]):
    Z = linkage(Xs, method=method); c, _ = cophenet(Z, d)
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, no_labels=True, color_threshold=0.7 * Z[-1, 2])
    ax.set_title(f"Dendrogram — {method} linkage (cophenetic r = {c:.3f})"); ax.set_xlabel("Samples (truncated to 30 leaves)"); ax.set_ylabel("Merge distance")
plt.tight_layout(); plt.savefig("../figures/clu_dendrograms.png", bbox_inches="tight"); plt.show()'''))
cu.append(C('''# Compare linkage strategies at the same k (internal metrics only — no labels)
link_rows = []
for link in ["ward", "complete", "average", "single"]:
    lab = AgglomerativeClustering(n_clusters=K, linkage=link).fit_predict(Xs)
    link_rows.append({"Linkage": link, "Silhouette": silhouette_score(Xs, lab), "Davies-Bouldin": davies_bouldin_score(Xs, lab),
                      "Calinski-Harabasz": calinski_harabasz_score(Xs, lab), "Cluster sizes": np.bincount(lab).tolist()})
link_df = pd.DataFrame(link_rows).set_index("Linkage"); display(link_df.round(4))
best_link = link_df["Silhouette"].idxmax(); print("Best linkage by silhouette:", best_link)'''))
cu.append(obs("clu_linkage"))
cu.append(M("## 4. Fit final models & evaluate with all three internal metrics\nSilhouette ↑ better · Davies-Bouldin ↓ better · Calinski-Harabasz ↑ better."))
cu.append(C('''kmeans = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Xs)
agg = AgglomerativeClustering(n_clusters=K, linkage=best_link).fit(Xs)
labels = {"K-Means": kmeans.labels_, f"Agglomerative ({best_link})": agg.labels_}
rows = []
for name, lab in labels.items():
    rows.append({"Algorithm": name, "k": len(np.unique(lab)), "Silhouette (↑)": silhouette_score(Xs, lab),
                 "Davies-Bouldin (↓)": davies_bouldin_score(Xs, lab), "Calinski-Harabasz (↑)": calinski_harabasz_score(Xs, lab),
                 "Cluster sizes": np.bincount(lab).tolist()})
clu_df = pd.DataFrame(rows).set_index("Algorithm"); display(clu_df.round(4)); clu_df.round(4).to_csv("../figures/clustering_metrics.csv")'''))
cu.append(obs("clu_metrics"))
cu.append(M("## 5. Cluster visualisation — PCA (2 components) for every algorithm"))
cu.append(C('''from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE); P = pca.fit_transform(Xs)
ev = pca.explained_variance_ratio_ * 100
panels = {**labels, "Gender (post-hoc reference, not used in fitting)": gender}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, lab) in zip(axes, panels.items()):
    sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=lab, palette="colorblind", s=45, ax=ax, edgecolor="white", linewidth=0.4, legend="full")
    ax.set_title(name); ax.set_xlabel(f"PC1 ({ev[0]:.1f}% var.)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}% var.)"); ax.legend(title="Group")
if hasattr(kmeans, "cluster_centers_"):
    cp = pca.transform(kmeans.cluster_centers_); axes[0].scatter(cp[:, 0], cp[:, 1], marker="X", s=220, c="black", label="Centroids")
    axes[0].legend(title="Cluster")
fig.suptitle(f"PCA projection (total variance explained: {ev.sum():.1f}%)", y=1.02)
plt.tight_layout(); plt.savefig("../figures/clu_pca.png", bbox_inches="tight"); plt.show()'''))
cu.append(obs("clu_pca"))
cu.append(M("## 6. t-SNE visualisation (encouraged; shown for both algorithms)"))
cu.append(C('''from sklearn.manifold import TSNE
perp = min(30, max(5, (len(Xs) - 1) // 3))
T = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_STATE, init="pca", learning_rate="auto").fit_transform(Xs)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, lab) in zip(axes, labels.items()):
    sns.scatterplot(x=T[:, 0], y=T[:, 1], hue=lab, palette="colorblind", s=45, ax=ax, edgecolor="white", linewidth=0.4, legend="full")
    ax.set_title(f"t-SNE — {name}"); ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2"); ax.legend(title="Cluster")
plt.tight_layout(); plt.savefig("../figures/clu_tsne.png", bbox_inches="tight"); plt.show()'''))
cu.append(M("## 7. Post-hoc interpretation (no ground truth exists)\nThere is no label to compute ARI/NMI against, so we (a) check how much the two algorithms **agree with each other** (ARI between their labelings) and (b) look at the gender mix of each cluster — gender was *not* used for fitting."))
cu.append(C('''agree_ari = adjusted_rand_score(kmeans.labels_, agg.labels_); agree_nmi = normalized_mutual_info_score(kmeans.labels_, agg.labels_)
val = pd.DataFrame({"K-Means vs Agglomerative": {"Adjusted Rand Index": agree_ari, "Normalized Mutual Info": agree_nmi}}).T
display(val.round(4)); val.round(4).to_csv("../figures/clustering_posthoc_validation.csv")
print("Gender mix per K-Means cluster (row %):")
display((pd.crosstab(pd.Series(kmeans.labels_, name="cluster"), pd.Series(gender, name="gender"), normalize="index")*100).round(1))'''))
cu.append(M("### Cluster profiles (mean of original features per K-Means cluster)"))
cu.append(C('''prof = X.assign(cluster=kmeans.labels_).groupby("cluster").mean().round(2); prof["size"] = np.bincount(kmeans.labels_); display(prof)
fig, ax = plt.subplots(figsize=(7, 3.5))
sns.heatmap(pd.DataFrame(Xs, columns=FEATURES).assign(cluster=kmeans.labels_).groupby("cluster").mean().T, cmap="coolwarm", center=0, annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "Mean z-score"})
ax.set_title("K-Means cluster profile (standardised feature means)"); ax.set_xlabel("Cluster"); ax.set_ylabel("Feature")
plt.tight_layout(); plt.savefig("../figures/clu_profile.png", bbox_inches="tight"); plt.show()'''))
cu.append(obs("clu_profile"))
cu.append(M("## 8. Conclusions\n_(Write in your own words.)_ Cover: chosen k and why (elbow + silhouette agree/disagree?); which algorithm wins on each metric; effect of linkage; what PCA/t-SNE reveal; agreement between the two algorithms (ARI) and the business meaning of each segment (income vs spending); limitations (scale sensitivity, spherical-cluster assumption of K-Means, O(n²) cost of hierarchical)."))

def save(cells, path):
    nb = new_notebook(cells=cells); nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, path)
here = os.path.dirname(os.path.abspath(__file__))
save(cl, os.path.join(here, "..", "notebooks", "classification.ipynb"))
save(cu, os.path.join(here, "..", "notebooks", "clustering.ipynb"))
print("notebooks written")
