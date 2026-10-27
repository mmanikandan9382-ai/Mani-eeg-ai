from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "eeg_segments.csv"
FEATURES = ["mean","std","variance","rms","line_length","zero_crossings",
            "delta_power","theta_power","alpha_power","beta_power","gamma_power","dominant_frequency"]

def generate(n_per_class=400, seed=17):
    rng = np.random.default_rng(seed)
    rows = []
    # Feature patterns are deliberately simulated and do not represent physiological distributions.
    for label, cls in [("non_seizure", 0), ("seizure", 1)]:
        for i in range(n_per_class):
            scale = rng.uniform(0.7, 1.3)
            row = {
                "mean": rng.normal(0, .05),
                "std": scale * (1.0 if cls == 0 else 1.65) + rng.normal(0,.15),
                "variance": scale**2 * (1.0 if cls == 0 else 2.7) + rng.normal(0,.2),
                "rms": scale * (1.0 if cls == 0 else 1.6) + rng.normal(0,.15),
                "line_length": scale * (0.7 if cls == 0 else 1.2) + rng.normal(0,.12),
                "zero_crossings": rng.uniform(.1,.5) if cls == 0 else rng.uniform(.03,.25),
                "delta_power": rng.lognormal(0.0 if cls == 0 else .45, .45),
                "theta_power": rng.lognormal(0.1 if cls == 0 else .35, .4),
                "alpha_power": rng.lognormal(.4 if cls == 0 else .0, .4),
                "beta_power": rng.lognormal(.1 if cls == 0 else .3, .4),
                "gamma_power": rng.lognormal(-.2 if cls == 0 else .2, .45),
                "dominant_frequency": rng.normal(10 if cls == 0 else 6.5, 2.5 if cls == 0 else 2.0),
                "label": label,
                "subject_id": f"S{(i % 20)+1:02d}"
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # Shuffle rows for the sample only. Real datasets should preserve subject/session metadata.
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(OUT, index=False)
    print(f"Created synthetic illustrative feature table: {OUT}")
