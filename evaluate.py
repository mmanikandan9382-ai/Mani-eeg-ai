from pathlib import Path
import argparse, json
import joblib, pandas as pd
from train import FEATURES, metric_summary

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default=str(ROOT/"data/sample/eeg_segments.csv"))
    a=p.parse_args()
    model_path=ROOT/"outputs/eeg_seizure_classifier.joblib"
    if not model_path.exists(): raise FileNotFoundError("Run python src/train.py first.")
    df=pd.read_csv(a.data).dropna(subset=FEATURES+["label"])
    df=df[df.label.isin(["non_seizure","seizure"])]
    model=joblib.load(model_path)
    result=metric_summary(df.label.astype(str),model.predict(df[FEATURES].astype(float)))
    result["warning"]="This evaluates every row supplied and is not an independent test if these rows were used for training. Use a separate held-out dataset for final claims."
    out=ROOT/"outputs"; (out/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="classification_report"},indent=2))
if __name__=="__main__": main()
