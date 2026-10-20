"""Streamlit GUI for the Review-2 classifier (bonus +1). Run:  streamlit run app/app.py"""
from pathlib import Path
import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="ML Capstone — Income Classifier", page_icon="💰", layout="wide")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_classifier.joblib"

@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)

bundle = load()
model, features = bundle["model"], bundle["features"]
num_cols, cat_cols = bundle["num_cols"], bundle["cat_cols"]
stats, levels, labels = bundle["num_stats"], bundle["cat_levels"], bundle["class_labels"]
DERIVED = {"net_capital"}            # computed from capital_gain - capital_loss, not asked from the user

st.title("Adult Income prediction")
st.caption(f"Model: **{bundle['model_name']}** (selected by 5-fold CV weighted-F1 in `classification.ipynb`). Predicts whether annual income is >50K or <=50K.")

st.sidebar.header("Person's attributes")
values = {}
for f in num_cols:
    if f in DERIVED: continue
    lo, mean, hi = stats[f]["min"], stats[f]["mean"], stats[f]["max"]
    values[f] = st.sidebar.slider(f, int(lo), int(hi), int(round(mean)))
for f in cat_cols:
    values[f] = st.sidebar.selectbox(f, levels[f])
values["net_capital"] = values["capital_gain"] - values["capital_loss"]

row = pd.DataFrame([values])[features]
pred = int(model.predict(row)[0])
proba = pd.Series(model.predict_proba(row)[0], index=[labels[c] for c in bundle["classes"]], name="Probability")

c1, c2 = st.columns(2)
c1.metric("Predicted income", labels[pred])
c1.bar_chart(proba)
c2.subheader("Input summary"); c2.dataframe(row.T.rename(columns={0: "value"}), width="stretch")
