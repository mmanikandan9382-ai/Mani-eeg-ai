from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["mean","std","variance","rms","line_length","zero_crossings",
            "delta_power","theta_power","alpha_power","beta_power","gamma_power","dominant_frequency"]

def metric_summary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["non_seizure","seizure"])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true,y_pred)),
        "precision": float(precision_score(y_true,y_pred,pos_label="seizure",zero_division=0)),
        "sensitivity_recall": float(recall_score(y_true,y_pred,pos_label="seizure",zero_division=0)),
        "specificity": float(tn/(tn+fp)) if tn+fp else 0.0,
        "f1_score": float(f1_score(y_true,y_pred,pos_label="seizure",zero_division=0)),
        "confusion_matrix_labels": ["non_seizure","seizure"],
        "confusion_matrix": cm.tolist(),
        "true_negative": int(tn), "false_alarms_false_positives": int(fp),
        "missed_detections_false_negatives": int(fn), "true_positive": int(tp),
        "classification_report": classification_report(y_true,y_pred,output_dict=True,zero_division=0)
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default=str(ROOT/"data/sample/eeg_segments.csv"))
    p.add_argument("--test-size",type=float,default=.25)
    p.add_argument("--subject-independent",action="store_true",help="Split by subject_id groups if available")
    a=p.parse_args()
    df=pd.read_csv(a.data)
    required=FEATURES+["label"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    df=df.dropna(subset=required).copy()
    df=df[df.label.isin(["non_seizure","seizure"])]
    X=df[FEATURES].astype(float); y=df.label.astype(str)
    if a.subject_independent:
        if "subject_id" not in df: raise ValueError("subject_id column is required for subject-independent split.")
        splitter=GroupShuffleSplit(n_splits=1,test_size=a.test_size,random_state=42)
        train_idx,test_idx=next(splitter.split(X,y,groups=df.subject_id))
        Xtr,Xte=X.iloc[train_idx],X.iloc[test_idx]; ytr,yte=y.iloc[train_idx],y.iloc[test_idx]
        split_note="GroupShuffleSplit by subject_id"
    else:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=a.test_size,random_state=42,stratify=y)
        split_note="Stratified random split (not patient-independent)"
    model=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1,min_samples_leaf=2)
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    metrics=metric_summary(yte,pred)
    metrics.update({"train_rows":len(Xtr),"test_rows":len(Xte),"split_method":split_note,
                    "warning":"Metrics on synthetic sample data are illustrative only; not clinical performance."})
    out=ROOT/"outputs"; out.mkdir(exist_ok=True)
    joblib.dump(model,out/"eeg_seizure_classifier.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k not in ("classification_report",)},indent=2))
    print(f"Saved model and metrics under {out}")

if __name__=="__main__": main()
