"""Held-out validation of layer selection.

Partitions the 30 Monte Carlo seeds into a selection set (0-14) and a disjoint
evaluation set (15-29). The stability-aware criterion S(l) = mu - sigma,
averaged over k = 1..20, is re-applied using only the selection seeds; the
resulting nested L* is then evaluated on the held-out seeds at k = 20.
Neighboring layers L*-1 and L*+1 are evaluated on the same held-out seeds,
and paired Wilcoxon signed-rank tests assess local stability.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import Config

K_VALUES_FOR_SELECTION = list(range(1, 21))
K_FOR_REPORTING = 10
SELECTION_SEEDS = list(range(0, 15))
EVALUATION_SEEDS = list(range(15, 30))


def per_seed_balanced_acc(X, y, k, seeds):
    """Return list of per-seed balanced accuracies for the given (X, y, k, seeds)."""
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    accs = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        try:
            train_ix = np.concatenate([
                rng.choice(idx_0, k, replace=False),
                rng.choice(idx_1, k, replace=False),
            ])
            test_ix = np.setdiff1d(np.arange(len(y)), train_ix)
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel='linear', class_weight='balanced'),
            )
            clf.fit(X[train_ix], y[train_ix])
            pred = clf.predict(X[test_ix])
            sens = recall_score(y[test_ix], pred, pos_label=1)
            spec = recall_score(y[test_ix], pred, pos_label=0)
            accs.append((sens + spec) / 2)
        except Exception:
            continue
    return accs


def stability_score(per_layer_accs_by_k):
    """S(l) = mean over k of mean_seed(BAcc) minus mean over k of std_seed(BAcc)."""
    means, stds = [], []
    for accs in per_layer_accs_by_k.values():
        if len(accs) == 0:
            continue
        means.append(np.mean(accs))
        stds.append(np.std(accs))
    if not means:
        return -np.inf
    return np.mean(means) - np.mean(stds)


def main():
    print(f"Held-out validation of layer selection")
    print(f"  Selection seeds:        {SELECTION_SEEDS}")
    print(f"  Evaluation seeds:       {EVALUATION_SEEDS}")
    print(f"  k for selection score:  {K_VALUES_FOR_SELECTION[0]}..{K_VALUES_FOR_SELECTION[-1]}")
    print(f"  k for reporting:        {K_FOR_REPORTING}\n")

    with open(os.path.join(Config.RESULTS_DIR, "best_layers.json")) as f:
        original_selection = json.load(f)

    rows = []
    summary_lines = []

    for model in Config.MODELS.keys():
        emb_path = os.path.join(Config.EMB_DIR, f"{model}_embeddings.pkl")
        if not os.path.exists(emb_path):
            print(f"  Skipping {model}: embeddings not found")
            continue
        df = pd.read_pickle(emb_path)
        layers = sorted(df['layer'].unique())

        print(f"Processing {model} ({len(layers)} layers)...")

        per_layer_sel = {L: {} for L in layers}
        per_layer_eval = {}

        for L in layers:
            sub = df[df['layer'] == L].sort_values('file_id')
            X = np.stack(sub['embedding'].values)
            y = sub['label'].values

            for k in K_VALUES_FOR_SELECTION:
                per_layer_sel[L][k] = per_seed_balanced_acc(X, y, k, SELECTION_SEEDS)
            per_layer_eval[L] = per_seed_balanced_acc(X, y, K_FOR_REPORTING, EVALUATION_SEEDS)

        scores_sel = {L: stability_score(per_layer_sel[L]) for L in layers}
        L_star_nested = max(scores_sel, key=scores_sel.get)
        L_star_original = original_selection.get(model)

        eval_nested = np.array(per_layer_eval[L_star_nested])
        rows.append({
            'Model': model, 'Scheme': 'nested_held_out',
            'Layer': L_star_nested,
            'BAcc_eval_mean': eval_nested.mean(),
            'BAcc_eval_std': eval_nested.std(),
            'p_value': np.nan,
            'n_seeds': len(eval_nested),
        })

        eval_original = None
        if L_star_original is not None and L_star_original in layers:
            eval_original = np.array(per_layer_eval[L_star_original])
            rows.append({
                'Model': model, 'Scheme': 'original_circular',
                'Layer': L_star_original,
                'BAcc_eval_mean': eval_original.mean(),
                'BAcc_eval_std': eval_original.std(),
                'p_value': np.nan,
                'n_seeds': len(eval_original),
            })

        neighbor_results = {}
        for offset in (-1, 0, +1):
            L = L_star_nested + offset
            if L not in layers:
                continue
            accs = np.array(per_layer_eval[L])
            neighbor_results[offset] = accs
            rows.append({
                'Model': model, 'Scheme': f'neighbor_L*{offset:+d}',
                'Layer': L,
                'BAcc_eval_mean': accs.mean(),
                'BAcc_eval_std': accs.std(),
                'p_value': np.nan,
                'n_seeds': len(accs),
            })

        center = neighbor_results.get(0)
        if center is not None:
            for offset in (-1, +1):
                if offset not in neighbor_results:
                    continue
                neighbor = neighbor_results[offset]
                n = min(len(center), len(neighbor))
                if n < 3:
                    continue
                try:
                    _, p = wilcoxon(center[:n], neighbor[:n])
                except ValueError:
                    p = np.nan
                rows.append({
                    'Model': model,
                    'Scheme': f'wilcoxon_L*_vs_L*{offset:+d}',
                    'Layer': f"{L_star_nested} vs {L_star_nested + offset}",
                    'BAcc_eval_mean': np.nan,
                    'BAcc_eval_std': np.nan,
                    'p_value': p,
                    'n_seeds': n,
                })

        gap_str = ""
        if eval_original is not None:
            gap = (eval_original.mean() - eval_nested.mean()) * 100
            gap_str = (f" | original L={L_star_original}: "
                       f"{eval_original.mean() * 100:.1f}% (gap {gap:+.2f}%)")
        same = " (same as original)" if L_star_nested == L_star_original else ""
        summary_lines.append(
            f"- {model}: nested L*={L_star_nested}{same}, "
            f"held-out BAcc = {eval_nested.mean() * 100:.1f}% +/- "
            f"{eval_nested.std() * 100:.1f}%{gap_str}"
        )

    res = pd.DataFrame(rows)
    out_csv = os.path.join(Config.RESULTS_DIR, "circularity_analysis.csv")
    res.to_csv(out_csv, index=False)

    out_md = os.path.join(Config.RESULTS_DIR, "circularity_summary.md")
    with open(out_md, "w") as f:
        f.write("# Held-out validation of layer selection\n\n")
        f.write(f"Selection criterion: S(l) = mean(mu_k) - mean(std_k) "
                f"averaged over k = {K_VALUES_FOR_SELECTION[0]}..{K_VALUES_FOR_SELECTION[-1]}.\n\n")
        f.write(f"- Selection seeds: {SELECTION_SEEDS}\n")
        f.write(f"- Evaluation seeds: {EVALUATION_SEEDS}\n")
        f.write(f"- Reporting k = {K_FOR_REPORTING}\n\n")
        f.write("## Per-model summary\n\n")
        f.write("\n".join(summary_lines))
        f.write("\n\n## Full table\n\n")
        f.write(res.to_markdown(index=False, floatfmt=".4f"))

    print("\nResults:\n")
    print(res.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nSaved CSV -> {out_csv}")
    print(f"Saved summary -> {out_md}")


if __name__ == "__main__":
    main()
