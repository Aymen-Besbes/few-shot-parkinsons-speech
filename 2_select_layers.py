"""Stability-aware layer selection: pick L* maximizing S(l) = mu - sigma."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import Config


def safe_mean_std(values):
    """Return mean and sample std (ddof=1). Std is 0 if only one value."""
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(np.mean(values)), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def evaluate_layer(X, y):
    """Run the seeds x k Monte Carlo loop for one (model, layer) and return per-k metrics."""
    metrics = []
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    if min(len(idx_0), len(idx_1)) < min(Config.K_VALUES):
        return []

    for k in Config.K_VALUES:
        baccs, sens_list, spec_list = [], [], []
        for seed in range(Config.SEEDS):
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
                bacc = (sens + spec) / 2

                baccs.append(bacc)
                sens_list.append(sens)
                spec_list.append(spec)
            except Exception:
                continue

        if baccs:
            bacc_m, bacc_s = safe_mean_std(baccs)
            sens_m, sens_s = safe_mean_std(sens_list)
            spec_m, spec_s = safe_mean_std(spec_list)
            metrics.append({
                'k': k,
                'BAcc_Mean': bacc_m, 'BAcc_Std': bacc_s,
                'Sens_Mean': sens_m, 'Sens_Std': sens_s,
                'Spec_Mean': spec_m, 'Spec_Std': spec_s,
            })
    return metrics


def select_best():
    best_layers = {}
    all_results = []

    for model_name in Config.MODELS.keys():
        print(f"Analyzing {model_name}...")
        df_path = os.path.join(Config.EMB_DIR, f"{model_name}_embeddings.pkl")
        if not os.path.exists(df_path):
            print(f"  Embeddings file not found: {df_path}")
            continue

        df = pd.read_pickle(df_path)
        layer_scores = []

        for layer in sorted(df['layer'].unique()):
            sub = df[df['layer'] == layer].sort_values('file_id')
            X = np.stack(sub['embedding'].values)
            y = sub['label'].values

            res = evaluate_layer(X, y)
            if not res:
                continue

            for r in res:
                all_results.append({
                    'Model': model_name,
                    'Layer': int(layer),
                    'k': r['k'],
                    'BAcc_Mean': r['BAcc_Mean'], 'BAcc_Std': r['BAcc_Std'],
                    'Sens_Mean': r['Sens_Mean'], 'Sens_Std': r['Sens_Std'],
                    'Spec_Mean': r['Spec_Mean'], 'Spec_Std': r['Spec_Std'],
                })

            # Stability-aware score: S(l) = mean(mu_k) - mean(sigma_k)
            avg_acc = np.mean([r['BAcc_Mean'] for r in res])
            avg_std = np.mean([r['BAcc_Std'] for r in res])
            score = avg_acc - avg_std
            layer_scores.append((int(layer), score, avg_acc, avg_std))

        if not layer_scores:
            print(f"  No valid layers for {model_name}")
            continue

        best = max(layer_scores, key=lambda x: x[1])
        best_layers[model_name] = best[0]
        print(f"  Best layer: {best[0]} "
              f"(score={best[1]:.3f}, acc={best[2]:.3f}, std={best[3]:.3f})")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(Config.RESULTS_DIR, "best_layers.json")
    with open(json_path, "w") as f:
        json.dump(best_layers, f, indent=4)
    print(f"Saved {json_path}")

    csv_path = os.path.join(Config.RESULTS_DIR, "layer_metrics.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    select_best()