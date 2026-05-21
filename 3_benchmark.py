"""Few-shot fusion benchmark: early fusion (concat) and late fusion (majority vote).

Saves aggregated results (final_results.csv) and per-seed balanced accuracies
(per_seed_scores.pkl) for downstream statistical analysis.
"""
import itertools
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import Config


def load_data_map():
    """Load embeddings at the best layer for each model."""
    with open(os.path.join(Config.RESULTS_DIR, "best_layers.json"), "r") as f:
        best_layers = json.load(f)

    data_map = {}
    for m, l in best_layers.items():
        df = pd.read_pickle(os.path.join(Config.EMB_DIR, f"{m}_embeddings.pkl"))
        sub = df[df['layer'] == l].sort_values('file_id')
        data_map[m] = {
            'X': np.stack(sub['embedding'].values),
            'y': sub['label'].values,
            'ids': sub['file_id'].values,
        }
    return data_map


def evaluate_combo(combo, data_map):
    """Evaluate one model combination across all k and all seeds."""
    y = data_map[combo[0]]['y']
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    X_concat = np.hstack([data_map[m]['X'] for m in combo])

    aggregated = []
    per_seed = {}

    for k in Config.K_VALUES:
        accs_early, sens_early, spec_early = [], [], []
        accs_late, sens_late, spec_late = [], [], []

        for seed in range(Config.SEEDS):
            rng = np.random.RandomState(seed)
            try:
                train_ix = np.concatenate([
                    rng.choice(idx_0, k, replace=False),
                    rng.choice(idx_1, k, replace=False),
                ])
                test_ix = np.setdiff1d(np.arange(len(y)), train_ix)

                # Early fusion: concatenated embeddings, single classifier
                clf = make_pipeline(
                    StandardScaler(),
                    SVC(kernel='linear', class_weight='balanced'),
                )
                clf.fit(X_concat[train_ix], y[train_ix])
                pred = clf.predict(X_concat[test_ix])
                s = recall_score(y[test_ix], pred, pos_label=1)
                sp = recall_score(y[test_ix], pred, pos_label=0)
                sens_early.append(s)
                spec_early.append(sp)
                accs_early.append((s + sp) / 2)

                # Late fusion: per-model classifiers, majority vote
                if len(combo) > 1:
                    preds = []
                    for m in combo:
                        clf_m = make_pipeline(
                            StandardScaler(),
                            SVC(kernel='linear', class_weight='balanced'),
                        )
                        clf_m.fit(data_map[m]['X'][train_ix], y[train_ix])
                        preds.append(clf_m.predict(data_map[m]['X'][test_ix]))
                    pred_vote = mode(np.array(preds), axis=0)[0].flatten()
                    s_v = recall_score(y[test_ix], pred_vote, pos_label=1)
                    sp_v = recall_score(y[test_ix], pred_vote, pos_label=0)
                    sens_late.append(s_v)
                    spec_late.append(sp_v)
                    accs_late.append((s_v + sp_v) / 2)
            except Exception:
                continue

        name = "+".join(combo)

        if accs_early:
            aggregated.append({
                'Method': name, 'k': k, 'Type': 'EarlyFusion',
                'Acc': np.mean(accs_early), 'Std': np.std(accs_early),
                'Sens': np.mean(sens_early), 'Sens_Std': np.std(sens_early),
                'Spec': np.mean(spec_early), 'Spec_Std': np.std(spec_early),
            })
            per_seed[(name, 'EarlyFusion', k)] = accs_early
        if accs_late:
            aggregated.append({
                'Method': name, 'k': k, 'Type': 'LateFusion',
                'Acc': np.mean(accs_late), 'Std': np.std(accs_late),
                'Sens': np.mean(sens_late), 'Sens_Std': np.std(sens_late),
                'Spec': np.mean(spec_late), 'Spec_Std': np.std(spec_late),
            })
            per_seed[(name, 'LateFusion', k)] = accs_late

    return aggregated, per_seed


def run_benchmark():
    print("Running fusion benchmark (early + late)...")
    data_map = load_data_map()
    models = list(data_map.keys())

    combos = []
    for r in range(1, len(models) + 1):
        combos.extend(itertools.combinations(models, r))

    all_results = []
    all_per_seed = {}
    for combo in combos:
        results, per_seed = evaluate_combo(combo, data_map)
        all_results.extend(results)
        all_per_seed.update(per_seed)

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(Config.RESULTS_DIR, "final_results.csv"), index=False)
    with open(os.path.join(Config.RESULTS_DIR, "per_seed_scores.pkl"), "wb") as f:
        pickle.dump(all_per_seed, f)

    print("\nTop results at k=20:")
    final = res_df[res_df['k'] == 20].sort_values('Acc', ascending=False)
    print(final[['Method', 'Type', 'Acc', 'Std', 'Sens', 'Spec']].head(10).to_string(index=False))


if __name__ == "__main__":
    run_benchmark()