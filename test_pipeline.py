from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from generate_sample_data import generate, FEATURES

def test_sample_schema():
    df=generate(n_per_class=10)
    assert len(df)==20
    assert set(FEATURES).issubset(df.columns)
    assert set(df.label)=={"seizure","non_seizure"}
