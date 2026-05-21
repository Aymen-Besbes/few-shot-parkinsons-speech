"""Paired Wilcoxon signed-rank tests for fusion vs. baselines.

Reads per_seed_scores.pkl produced by 3_benchmark.py and tests whether
HuBERT+Whisper early fusion significantly outperforms each baseline at
k = 5, 10, 15, 20. Uses the one-sided alternative ('fusion > baseline')
and reports rank-biserial effect sizes alongside p-values.
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import Config

CHALLENGER = ('hubert_base+whisper_base', 'EarlyFusion')

BASELINES = [
    ('hubert_base',                'EarlyFusion'),
    ('wav2vec2_base',              'EarlyFusion'),
    ('whisper_base',               'EarlyFusion'),

]

K_REPORT = [5, 10, 15, 20]


def rank_biserial(differences):
    """Rank-biserial effect size: (#positive - #negative) / #non-zero."""
    diffs = np.asarray(differences)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    pos = int(np.sum(diffs > 0))
    neg = int(np.sum(diffs < 0))
    return (pos - neg) / (pos + neg)


def main():
    path = os.path.join(Config.RESULTS_DIR, "per_seed_scores.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run 3_benchmark.py first to produce per-seed scores."
        )
    with open(path, "rb") as f:
        per_seed = pickle.load(f)

    rows = []
    for k in K_REPORT:
        key = (CHALLENGER[0], CHALLENGER[1], k)
        if key not in per_seed:
            print(f"Missing per-seed scores for {key}")
            continue
        chal = np.array(per_seed[key])

        for base_method, base_type in BASELINES:
            base_key = (base_method, base_type, k)
            if base_key not in per_seed:
                continue
            base = np.array(per_seed[base_key])

            n = min(len(chal), len(base))
            a, b = chal[:n], base[:n]
            diff = a - b
            try:
                W, p = wilcoxon(a, b, alternative='greater', zero_method='wilcox')
            except ValueError:
                W, p = np.nan, np.nan
            r = rank_biserial(diff)

            rows.append({
                'k': k,
                'Challenger': f"{CHALLENGER[0]} ({CHALLENGER[1]})",
                'Baseline':   f"{base_method} ({base_type})",
                'Mean_diff_pct': diff.mean() * 100,
                'W': W,
                'p_value': p,
                'Effect_r': r,
                'Significant_p<0.05': (p < 0.05) if not np.isnan(p) else False,
            })

    df = pd.DataFrame(rows)
    out = os.path.join(Config.RESULTS_DIR, "wilcoxon_results.csv")
    df.to_csv(out, index=False)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
