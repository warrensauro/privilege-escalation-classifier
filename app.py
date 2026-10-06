from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_PATH = Path("priv_escalation_dataset.csv")
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]
SEVERITY_COLORS = {
    "Critical": "#d62728",
    "High": "#ff7f0e",
    "Medium": "#f2c12e",
    "Low": "#2ca02c",
}
TEXT_COLUMNS = [
    "platform", "command", "description", "category",
    "severity", "mapped_technique", "reference",
]
REQUIRED_COLUMNS = ["platform", "command", "description", "category", "severity"]
FEATURE_COLUMNS = [
    "platform", "category", "text",
    "command_len", "description_len", "text_word_count",
]

st.set_page_config(
    page_title="Privilege Escalation Risk Classifier",
    page_icon="🛡️",
    layout="wide",
)

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering identical to the notebook (section 3)."""
    out = frame.copy()
    out["text"] = (out["command"] + " " + out["description"]).str.strip()
    out["command_len"] = out["command"].str.len()
    out["description_len"] = out["description"].str.len()
    out["text_word_count"] = out["text"].str.split().str.len()
    return out


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning steps from the notebook (section 2)."""
    clean = df.drop_duplicates().copy()
    for col in TEXT_COLUMNS:
        if col in clean.columns:
            clean[col] = clean[col].fillna("").astype(str).str.strip()
    clean = clean[clean["severity"].isin(SEVERITY_ORDER)].copy()
    return add_features(clean)


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(data))


@st.cache_data(show_spinner=False)
def read_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline(n_estimators: int, random_state: int) -> Pipeline:
    """ColumnTransformer + RandomForest pipeline (sections 5 and 6)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             ["platform", "category"]),
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=5000),
             "text"),
            ("numeric", "passthrough",
             ["command_len", "description_len", "text_word_count"]),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]
    )


@st.cache_resource(show_spinner=False)
def train_model(clean_df: pd.DataFrame, n_estimators: int,
                test_size: float, random_state: int):
    """Train/test split (stratified), fit, and evaluate (sections 4, 6, 7, 8)."""
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["severity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_pipeline(n_estimators, random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    report = pd.DataFrame(
        classification_report(
            y_test, y_pred, labels=SEVERITY_ORDER,
            zero_division=0, output_dict=True,
        )
    ).T
    cm = confusion_matrix(y_test, y_pred, labels=SEVERITY_ORDER)

    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "metrics": metrics,
        "report": report,
        "cm": cm,
        "importance": importance_df,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_train": y_train,
        "y_test": y_test,
    }


def predict_records(model: Pipeline, records: pd.DataFrame) -> pd.DataFrame:
    """Predict severity + class probabilities for raw records (text only)."""
    feats = add_features(records)[FEATURE_COLUMNS]
    preds = model.predict(feats)
    proba = pd.DataFrame(
        model.predict_proba(feats),
        columns=model.named_steps["classifier"].classes_,
        index=records.index,
    )
    proba = proba.reindex(columns=SEVERITY_ORDER, fill_value=0.0)
    result = records.copy()
    result["predicted_severity"] = preds
    result["confidence"] = proba.max(axis=1).round(4)
    for sev in SEVERITY_ORDER:
        result[f"p_{sev}"] = proba[sev].round(4)
    return result

st.sidebar.title("🛡️ Settings")
st.sidebar.subheader("1. Dataset")

uploaded = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])

raw_df = None
source_label = ""
if uploaded is not None:
    raw_df = read_csv_bytes(uploaded.getvalue())
    source_label = f"Uploaded: {uploaded.name}"
elif DEFAULT_PATH.exists():
    raw_df = read_csv_path(str(DEFAULT_PATH))
    source_label = f"Local file: {DEFAULT_PATH}"

st.sidebar.subheader("2. Model")
n_estimators = st.sidebar.slider("Number of trees", 50, 600, 300, step=50)
test_size = st.sidebar.slider("Test split", 0.10, 0.40, 0.20, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

st.sidebar.divider()
st.sidebar.caption(
    "Defensive analytics only. Dataset commands are processed as text and "
    "are never executed."
)

st.title("Privilege Escalation Risk Classifier")
st.markdown(
    "Random Forest multiclass classification of privilege-escalation "
    "technique severity: **Critical, High, Medium, Low**."
)

if raw_df is None:
    st.info(
        "Upload `priv_escalation_dataset.csv` in the sidebar, or place it in "
        "the same folder as this app, to get started."
    )
    st.stop()

missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
if missing:
    st.error(f"The dataset is missing required columns: {', '.join(missing)}")
    st.stop()

clean_df = clean_dataset(raw_df)
if clean_df["severity"].nunique() < 2 or len(clean_df) < 50:
    st.error("Not enough valid labelled records after cleaning to train a model.")
    st.stop()

st.caption(f"{source_label} • {len(raw_df):,} raw rows → {len(clean_df):,} after cleaning")

with st.spinner("Training Random Forest pipeline..."):
    result = train_model(clean_df, n_estimators, float(test_size), int(random_state))
model = result["model"]

tab_data, tab_eval, tab_predict, tab_batch, tab_imp = st.tabs(
    ["📊 Data", "📈 Evaluation", "🔎 Predict", "📁 Batch Predict", "⭐ Feature Importance"]
)

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(clean_df):,}")
    c2.metric("Categories", clean_df["category"].nunique())
    c3.metric("Unique commands", f"{clean_df['command'].nunique():,}")
    c4.metric("Duplicates removed", len(raw_df) - len(raw_df.drop_duplicates()))

    left, right = st.columns(2)
    with left:
        st.subheader("Severity distribution")
        counts = clean_df["severity"].value_counts().reindex(SEVERITY_ORDER).fillna(0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(counts.index, counts.values,
               color=[SEVERITY_COLORS[s] for s in counts.index])
        ax.set_ylabel("Number of records")
        ax.set_xlabel("Severity")
        for i, v in enumerate(counts.values):
            ax.text(i, v, int(v), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Platform × severity")
        ct = pd.crosstab(clean_df["platform"], clean_df["severity"])
        ct = ct.reindex(columns=SEVERITY_ORDER, fill_value=0)
        st.dataframe(ct, width="stretch")
        st.subheader("Top categories")
        st.dataframe(
            clean_df["category"].value_counts().head(10).rename("records"),
            width="stretch",
        )

    st.subheader("Browse records")
    f1, f2, f3 = st.columns(3)
    plat_f = f1.multiselect("Platform", sorted(clean_df["platform"].unique()))
    sev_f = f2.multiselect("Severity", SEVERITY_ORDER)
    cat_f = f3.multiselect("Category", sorted(clean_df["category"].unique()))
    view = clean_df
    if plat_f:
        view = view[view["platform"].isin(plat_f)]
    if sev_f:
        view = view[view["severity"].isin(sev_f)]
    if cat_f:
        view = view[view["category"].isin(cat_f)]
    show_cols = [c for c in ["id", "platform", "category", "severity",
                             "command", "description"] if c in view.columns]
    st.dataframe(view[show_cols], width="stretch", height=350)
    st.caption(f"Showing {len(view):,} of {len(clean_df):,} records (text only; nothing is executed).")

with tab_eval:
    st.caption(
        f"Stratified split • {result['n_train']:,} training / {result['n_test']:,} "
        f"test records • {n_estimators} trees • balanced class weights"
    )
    cols = st.columns(4)
    for col, (name, val) in zip(cols, result["metrics"].items()):
        col.metric(name, f"{val:.2%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Classification report")
        rep = result["report"].copy()
        rep[["precision", "recall", "f1-score"]] = rep[["precision", "recall", "f1-score"]].round(3)
        rep["support"] = rep["support"].round(0).astype(int)
        st.dataframe(rep, width="stretch")
    with right:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ConfusionMatrixDisplay(
            confusion_matrix=result["cm"], display_labels=SEVERITY_ORDER
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Random Forest")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "Results describe performance on this held-out split only and should "
        "not be taken as proof of performance on new, external datasets."
    )

with tab_predict:
    st.subheader("Classify a technique")
    st.caption("The command is analysed purely as text. It is never run.")

    c1, c2 = st.columns(2)
    platform = c1.selectbox("Platform", sorted(clean_df["platform"].unique()))
    cats = sorted(clean_df.loc[clean_df["platform"] == platform, "category"].unique())
    category = c2.selectbox("Category", cats)
    command = st.text_area("Command (text only)", height=80,
                           placeholder="e.g. R --no-save -e 'system(\"/bin/sh\")'")
    description = st.text_area("Description", height=80,
                               placeholder="e.g. Abuse the 'R' binary (shell) in a sudo context")

    if st.button("Predict severity", type="primary"):
        if not (command.strip() or description.strip()):
            st.warning("Enter a command and/or description first.")
        else:
            rec = pd.DataFrame([{
                "platform": platform,
                "category": category,
                "command": command.strip(),
                "description": description.strip(),
            }])
            out = predict_records(model, rec).iloc[0]
            sev = out["predicted_severity"]
            color = SEVERITY_COLORS[sev]
            st.markdown(
                f"<div style='padding:14px 18px;border-radius:8px;"
                f"background:{color}22;border-left:6px solid {color};'>"
                f"<span style='font-size:0.9rem;'>Predicted severity</span><br>"
                f"<span style='font-size:2rem;font-weight:700;color:{color};'>{sev}</span>"
                f"<span style='margin-left:12px;'>confidence {out['confidence']:.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            probs = pd.Series({s: out[f"p_{s}"] for s in SEVERITY_ORDER})
            fig, ax = plt.subplots(figsize=(6, 2.6))
            ax.barh(probs.index[::-1], probs.values[::-1],
                    color=[SEVERITY_COLORS[s] for s in probs.index[::-1]])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("Or try a random record from the dataset"):
        if st.button("Pick random record"):
            st.session_state["sample"] = clean_df.sample(1).iloc[0].to_dict()
        s = st.session_state.get("sample")
        if s:
            st.write(f"**Platform:** {s['platform']}  |  **Category:** {s['category']}")
            st.code(s["command"], language="text")
            st.write(s["description"])
            pred = predict_records(model, pd.DataFrame([{
                k: s[k] for k in ["platform", "category", "command", "description"]
            }])).iloc[0]
            st.write(f"**Actual:** {s['severity']}  →  **Predicted:** {pred['predicted_severity']} "
                     f"({pred['confidence']:.1%})")

with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.caption("Required columns: platform, category, command, description.")
    batch_file = st.file_uploader("Upload CSV to classify", type=["csv"], key="batch")
    if batch_file is not None:
        bdf = pd.read_csv(batch_file)
        need = ["platform", "category", "command", "description"]
        miss = [c for c in need if c not in bdf.columns]
        if miss:
            st.error(f"Missing columns: {', '.join(miss)}")
        else:
            bdf = bdf.copy()
            for c in need:
                bdf[c] = bdf[c].fillna("").astype(str).str.strip()
            scored = predict_records(model, bdf)
            st.success(f"Classified {len(scored):,} records.")
            st.bar_chart(scored["predicted_severity"].value_counts().reindex(SEVERITY_ORDER).fillna(0))
            st.dataframe(scored, width="stretch", height=350)
            st.download_button(
                "Download predictions (CSV)",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="severity_predictions.csv",
                mime="text/csv",
            )

with tab_imp:
    st.subheader("Top Random Forest feature importances")
    top_n = st.slider("Features to show", 5, 40, 20)
    imp = result["importance"].head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, top_n * 0.28)))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#1f77b4")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(imp, width="stretch")
    st.caption(
        "Text features are TF-IDF unigrams/bigrams, so individual words or "
        "n-grams can appear among the most important features."
    )
