# Few-Shot Detection of Early-Stage Parkinson's Disease with Speech Foundation Models

This repository accompanies the paper:

> **Data-Efficient Detection of Early-Stage Parkinson’s Disease from 
Spontaneous Speech via Stability-Aware Foundation Model Fusion**
> [Authors]
> [Venue], [Year]
> [DOI or URL]

We benchmark three speech foundation models (HuBERT base, Wav2Vec2 base, Whisper base encoder) for few-shot detection of early-stage Parkinson's disease from spontaneous monologue, using linear-SVM classifiers on mean-pooled hidden-state embeddings. A stability-aware criterion `S(l) = mean(mu_k) - mean(sigma_k)` selects discriminative layers per model, and early/late fusion strategies are compared. The pipeline also includes a held-out validation of the layer selection (to address selection-evaluation seed overlap) and paired Wilcoxon significance tests on the fusion gains.

## Repository structure

| File | Purpose |
|---|---|
| `config.py` | Paths, model checkpoints, hyperparameters |
| `1_extract.py` | Extract per-layer mean-pooled embeddings for each subject |
| `2_select_layers.py` | Stability-aware (mu - sigma) layer selection |
| `3_benchmark.py` | Few-shot fusion benchmark (single + early + late) with per-seed score logging |
| `circularity_analysis.py` | Held-out validation of the layer selection |
| `wilcoxon_test.py` | Paired Wilcoxon signed-rank tests for best fusion vs. baselines |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |

## Data

This work uses the PC-GITA corpus, which is not redistributed in this repository. Request access from the corpus authors. The pipeline expects audio at:

```
data/PC-GITA-Monologue/hc/<filename>.wav      # Healthy controls
data/PC-GITA-Monologue/pd/<filename>.wav      # Parkinson's disease
```

Together with two metadata CSVs at `data/PC-GITA-Monologue/`:

```
df_healthy.csv         columns include: RECODING ORIGINAL NAME, Monologue
df_parkinsonian.csv    columns include: RECODING ORIGINAL NAME, Monologue, UPDRS-speech
```

The `Monologue` column gives the audio filename; `UPDRS-speech` is the per-subject UPDRS-III speech subscore used for early-stage filtering (subjects with score > 2 are excluded).

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU is required for embedding extraction. The remaining steps (layer selection, benchmark, statistical analyses) run on CPU.

## Reproducing the results

Run the scripts in order:

```bash
python 1_extract.py              # extract embeddings for all three models
python 2_select_layers.py        # select L* per model and save layer_metrics.csv
python 3_benchmark.py            # fusion benchmark + per-seed scores
python circularity_analysis.py   # held-out validation of layer selection
python wilcoxon_test.py          # paired Wilcoxon significance tests
```

Outputs are written to `./results/results/`:

- `best_layers.json` — selected layer per model
- `layer_metrics.csv` — full per-(model, layer, k) metrics
- `final_results.csv` — aggregated fusion benchmark
- `per_seed_scores.pkl` — per-seed balanced accuracies (input to Wilcoxon)
- `circularity_analysis.csv` and `circularity_summary.md` — held-out validation
- `wilcoxon_results.csv` — paired-test statistics

## Implementation details

- **Foundation models:** `facebook/hubert-base-ls960` (12 transformer layers), `facebook/wav2vec2-base-960h` (12), `openai/whisper-base` (6 encoder layers). All frozen; no fine-tuning.
- **Hidden states:** for each model, `N+1` hidden states are extracted (Layer 0 = convolutional feature extractor output; Layers 1..N = transformer block outputs).
- **Pooling:** mean over the time dimension.
- **Audio:** loaded with librosa at 16 kHz mono. No voice activity detection, silence trimming, or duration truncation.
- **Classifier:** `sklearn.svm.SVC` with `kernel='linear'`, `class_weight='balanced'`, default `C=1`, wrapped with `StandardScaler` in a `Pipeline`.
- **Few-shot protocol:** for each `k` in 1..20, draw `k` HC and `k` PD subjects without replacement using `numpy.random.RandomState(seed)` for seeds 0..29. Test on the remainder. Repeat for all 30 seeds.
- **Metric:** balanced accuracy = (sensitivity + specificity) / 2.

## Citation

If you use this code, please cite:

```bibtex
@article{[citation-key],
  title   = {[Few-Shot Detection of Early-Stage Parkinson's Disease with Speech Foundation Models]},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {[Year]}
}
```

## License

MIT (see `LICENSE`).
