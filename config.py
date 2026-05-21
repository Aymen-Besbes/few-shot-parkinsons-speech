"""Configuration for the few-shot Parkinson's disease detection pipeline."""
import os
import torch


class Config:
    # Paths
    DATA_DIR = "./data/PC-GITA-Monologue"
    BASE_OUTPUT_DIR = "./results"
    EMB_DIR = os.path.join(BASE_OUTPUT_DIR, "embeddings")
    RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "results")
    PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")
    for d in [EMB_DIR, RESULTS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Metadata CSV filenames (expected under DATA_DIR)
    CSV_HC = "df_healthy.csv"
    CSV_PD = "df_parkinsonian.csv"

    # Speech foundation models: {name: (HuggingFace ID, number of transformer layers)}
    MODELS = {
        'hubert_base':   ('facebook/hubert-base-ls960',   12),
        'wav2vec2_base': ('facebook/wav2vec2-base-960h',  12),
        'whisper_base':  ('openai/whisper-base',           6),
    }

    # Experimental setup
    SR = 16000
    SEEDS = 30
    K_VALUES = list(range(1, 21))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Plotting
    FONT_SCALE = 1.2
    DPI = 300
