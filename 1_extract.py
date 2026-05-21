"""Extract per-layer mean-pooled embeddings for each subject and model."""
import gc
import os

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    HubertModel,
    Wav2Vec2Model,
    WhisperModel,
)

from config import Config


def load_metadata():
    """Load HC and PD metadata, filter PD by UPDRS-III speech subscore, verify files exist."""
    df_hc = pd.read_csv(os.path.join(Config.DATA_DIR, Config.CSV_HC))
    df_pd = pd.read_csv(os.path.join(Config.DATA_DIR, Config.CSV_PD))

    df_hc['path'] = df_hc['Monologue'].apply(
        lambda x: os.path.join(Config.DATA_DIR, 'hc', x))
    df_pd['path'] = df_pd['Monologue'].apply(
        lambda x: os.path.join(Config.DATA_DIR, 'pd', x))

    df_hc['label'] = 0
    df_pd['label'] = 1

    if 'UPDRS-speech' in df_pd.columns:
        n_before = len(df_pd)
        df_pd = df_pd[df_pd['UPDRS-speech'] <= 2].copy()
        print(f"Filtered PD subjects (UPDRS-speech <= 2): {n_before} -> {len(df_pd)}")

    df = pd.concat([df_hc, df_pd], ignore_index=True)
    df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"Final cohort: {len(df)} subjects "
          f"(HC={int((df['label'] == 0).sum())}, "
          f"PD={int((df['label'] == 1).sum())})")
    return df


def load_model(name, checkpoint):
    """Return (processor, model) for a given foundation model."""
    if 'whisper' in name:
        proc = AutoProcessor.from_pretrained(checkpoint)
        model = WhisperModel.from_pretrained(checkpoint).to(Config.DEVICE)
    else:
        proc = AutoFeatureExtractor.from_pretrained(checkpoint)
        cls = HubertModel if 'hubert' in name else Wav2Vec2Model
        model = cls.from_pretrained(checkpoint).to(Config.DEVICE)
    model.eval()
    return proc, model


def extract_hidden_states(name, proc, model, waveform):
    """Return tuple of hidden states (Layer 0 through Layer N) for one waveform."""
    with torch.no_grad():
        if 'whisper' in name:
            inputs = proc(waveform, sampling_rate=Config.SR,
                          return_tensors="pt").input_features.to(Config.DEVICE)
            out = model.encoder(inputs, output_hidden_states=True)
        else:
            inputs = proc(waveform, sampling_rate=Config.SR,
                          return_tensors="pt", padding=True).input_values.to(Config.DEVICE)
            out = model(inputs, output_hidden_states=True)
    return out.hidden_states


def extract():
    df = load_metadata()

    for name, (checkpoint, _) in Config.MODELS.items():
        save_path = os.path.join(Config.EMB_DIR, f"{name}_embeddings.pkl")
        if os.path.exists(save_path):
            print(f"Skipping {name}: embeddings already exist.")
            continue

        print(f"Extracting embeddings with {name}...")
        proc, model = load_model(name, checkpoint)

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            waveform, _ = librosa.load(row['path'], sr=Config.SR)
            hidden_states = extract_hidden_states(name, proc, model, waveform)
            for i, h in enumerate(hidden_states):
                emb = torch.mean(h, dim=1).squeeze().cpu().numpy()
                results.append({
                    'file_id': row.get('RECODING ORIGINAL NAME',
                                       os.path.basename(row['path'])),
                    'label': row['label'],
                    'model': name,
                    'layer': i,
                    'embedding': emb,
                })

        pd.DataFrame(results).to_pickle(save_path)
        print(f"Saved {save_path}")

        del model, proc
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    extract()
