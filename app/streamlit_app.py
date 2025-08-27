import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="NSL-KDD Streamlit", layout="wide")
st.title("NSL-KDD Intrusion Detection — Streamlit")

# ---------------------------------------------------------------------
# Paths & model
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model" / "nslkdd_pipeline.joblib"
FEATURE_COLS = [f"f{i}" for i in range(1, 42)]
CATEGORICAL = ["f2", "f3", "f4"]
NUMERIC = [c for c in FEATURE_COLS if c not in CATEGORICAL]

if not MODEL_PATH.exists():
    st.error(f"Model not found at: {MODEL_PATH}\n"
             f"Make sure you saved the trained pipeline there.")
    st.stop()

pipe = joblib.load(MODEL_PATH)
st.caption(f"Loaded model: `{MODEL_PATH.name}`")

# Small set of known raw labels to auto-trim single-row inputs that include label/difficulty
KNOWN_LABELS = {
    "normal","back","land","neptune","pod","smurf","teardrop",
    "apache2","udpstorm","processtable","worm",
    "satan","ipsweep","nmap","portsweep","mscan","saint",
    "ftp_write","guess_passwd","imap","multihop","phf","spy",
    "warezclient","warezmaster","sendmail","named","snmpgetattack",
    "snmpguess","xlock","xsnoop",
    "buffer_overflow","loadmodule","perl","rootkit",
    "httptunnel","ps","sqlattack","xterm"
}

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📈 Batch Evaluation"])

# =========================
# Single-row prediction
# =========================
with tab1:
    st.subheader("Single row inference")
    st.write("Paste a CSV row. The app will auto-trim `label`/`difficulty` if you forgot to remove them.")

    example = "0,tcp,http,SF,232,8153,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,4,4,0,0,0,0,1,0,0,71,255,1,0,0.01,0.04,0,0,0,0,0"
    raw = st.text_area("CSV row (f1..f41 or with trailing label/difficulty)", example, height=100)

    if st.button("Predict", type="primary"):
        vals = [x.strip() for x in raw.strip().split(",") if x.strip() != ""]

        # Auto-handle NSL-KDD suffixes
        # 43 tokens and token[-2] is a known label -> strip label + difficulty
        if len(vals) == 43 and vals[-2].lower() in KNOWN_LABELS:
            vals = vals[:-2]
        # 42 tokens and token[-1] looks like difficulty 1..21 -> strip difficulty
        elif len(vals) == 42 and vals[-1].isdigit() and 1 <= int(vals[-1]) <= 21:
            vals = vals[:-1]

        if len(vals) != 41:
            st.error(f"Expected 41 values after cleanup, got {len(vals)}. "
                     f"Remove extra fields (label/difficulty) or stray commas.")
        else:
            # Build DF and cast numerics
            df1 = pd.DataFrame([vals], columns=FEATURE_COLS)
            # cast numeric cols strictly
            for c in NUMERIC:
                df1[c] = pd.to_numeric(df1[c], errors="raise")

            try:
                pred = pipe.predict(df1)[0]
                st.success(f"Prediction: **{pred}**")
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(df1)[0]
                    classes = getattr(pipe, "classes_", None)
                    if classes is not None:
                        st.markdown("**Class probabilities:**")
                        st.dataframe(pd.DataFrame([proba], columns=classes))
            except Exception as e:
                st.error(f"Inference error: {e}")

# =========================
# Batch evaluation
# =========================
# =========================
# Batch evaluation
# =========================
with tab2:
    st.subheader("Batch evaluation")
    st.write("Upload a CSV with columns **f1..f41**. Optional: `category` for metrics.")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is None:
        st.info("Waiting for CSV upload...")
    else:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # 0) Catch accidental extra columns (e.g., trailing values -> 'Unnamed: 41')
        extra_cols = [c for c in df.columns if c not in FEATURE_COLS + ["category"]]
        if extra_cols:
            st.error(f"Your CSV has unexpected columns: {extra_cols}. "
                     f"Each row must have exactly 41 features. "
                     f"Remove any extra trailing values (no difficulty/label here).")
            st.stop()

        # 1) Ensure required columns exist
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # 2) Trim whitespace
        df = df.replace(r"^\s+$", "", regex=True)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # 3) Dtypes: strings for cats, numerics for the rest
        for c in CATEGORICAL:
            df[c] = df[c].astype(str)
        df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")

        # 4) Pinpoint bad cells if coercion failed
        bad_mask = df[NUMERIC].isna()
        if bad_mask.any().any():
            bad_locs = []
            for col in NUMERIC:
                rows = bad_mask.index[bad_mask[col]].tolist()
                if rows:
                    bad_locs.append(f"{col}: rows {', '.join(str(r+1) for r in rows[:5])}")
            st.error("Your CSV has non-numeric values in numeric columns.\n\n"
                     "Fix these cells (1-based rows shown):\n- " + "\n- ".join(bad_locs))
            st.stop()

        # 5) Predict
        preds = pipe.predict(df[FEATURE_COLS])
        df["prediction"] = preds
        st.markdown("### Preview")
        st.dataframe(df.head(30))

        # Optional metrics
        if "category" in df.columns:
            st.markdown("### Metrics")
            st.text(classification_report(df["category"], df["prediction"], zero_division=0))

            st.markdown("### Confusion Matrix")
            fig = plt.figure()
            ConfusionMatrixDisplay.from_predictions(df["category"], df["prediction"])
            st.pyplot(fig)
        else:
            st.info("No `category` column found. Metrics skipped.")
