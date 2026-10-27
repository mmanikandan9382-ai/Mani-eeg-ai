from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from generate_sample_data import generate
FEATURES=["mean","std","variance","rms","line_length","zero_crossings","delta_power","theta_power","alpha_power","beta_power","gamma_power","dominant_frequency"]

st.set_page_config(page_title="EEG Pattern Research Demo",page_icon="🧠",layout="wide")
st.title("🧠 EEG Seizure Pattern Detection — Research Demo")
st.warning("Educational prototype only. Sample data is synthetic, not patient EEG. Not a diagnostic tool.")
uploaded=st.file_uploader("Upload a feature-table CSV",type=["csv"])
if uploaded:
    try: df=pd.read_csv(uploaded)
    except Exception as e: st.error(f"Could not read CSV: {e}"); st.stop()
else:
    st.caption("Showing synthetic demonstration data. Upload a licensed, appropriately prepared feature table for research.")
    df=generate()
missing=[c for c in FEATURES if c not in df.columns]
if missing:
    st.error("Missing feature columns: "+", ".join(missing)); st.stop()
a,b=st.columns(2)
a.metric("Segments",f"{len(df):,}")
b.metric("Feature count",len(FEATURES))
if "label" in df:
    st.subheader("Label distribution")
    st.bar_chart(df.label.astype(str).value_counts())
feature=st.selectbox("Inspect a feature",FEATURES,index=1)
fig,ax=plt.subplots(figsize=(9,3.5))
if "label" in df:
    for label,part in df.groupby("label"):
        ax.hist(pd.to_numeric(part[feature],errors="coerce").dropna(),bins=30,alpha=.5,label=str(label))
    ax.legend()
else:
    ax.hist(pd.to_numeric(df[feature],errors="coerce").dropna(),bins=30)
ax.set_title(feature); ax.set_xlabel("Value"); ax.set_ylabel("Count")
st.pyplot(fig); plt.close(fig)
model_path=ROOT/"outputs/eeg_seizure_classifier.joblib"
if model_path.exists():
    import joblib
    model=joblib.load(model_path)
    df["predicted_label"]=model.predict(df[FEATURES].astype(float))
    st.subheader("Model predictions")
    st.dataframe(df.head(100),use_container_width=True)
    st.download_button("Download predictions CSV",df.to_csv(index=False).encode(),"eeg_predictions.csv","text/csv")
else:
    st.info("No trained model found. Run `python src/train.py` in the project terminal.")
st.subheader("Preview")
st.dataframe(df.head(25),use_container_width=True)
