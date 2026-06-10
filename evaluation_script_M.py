#!/usr/bin/env python3
"""
Evaluation script for DNA regulatory activity prediction.

Usage: python evaluation_script.py <path_to_model> <path_to_test_data>

Input: TSV file with columns: id    sequence

Output: Prints predictions to stdout and saves them to predictions.tsv:
        id    predicted_is_active    predicted_rna_dna_ratio
"""

from __future__ import annotations
import argparse
import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn


SEQ_COL = "sequence"

def clean_sequence(seq):
    """Normalize DNA sequence by uppercasing and replacing invalid bases with 'N'."""
    seq = str(seq).upper().strip()
    return "".join(base if base in "ACGT" else "N" for base in seq)


def one_hot_encode(seq, max_len):
    """Convert DNA sequence into one-hot encoded matrix. """
    seq = clean_sequence(seq)
    encoded = np.zeros((4, max_len), dtype=np.float32)
    base_to_channel = {"A": 0, "C": 1, "G": 2, "T": 3}

    for position, base in enumerate(seq[:max_len]):
        channel = base_to_channel.get(base)
        if channel is not None:
            encoded[channel, position] = 1.0

    return encoded


class DNARegulatoryCNN(nn.Module):
    """CNN model with shared representation and two prediction heads."""

    def __init__(self, model_params: dict) -> None:
        super().__init__()

        num_filters1 = model_params["num_filters1"]
        num_filters2 = model_params["num_filters2"]
        kernel_size1 = model_params["kernel_size1"]
        kernel_size2 = model_params["kernel_size2"]
        kernel_size3 = model_params.get("kernel_size3", 5)
        hidden_dim = model_params["hidden_dim"]
        dropout = model_params["dropout"]

        self.features = nn.Sequential(
            nn.Conv1d(4, num_filters1, kernel_size=kernel_size1, padding=kernel_size1 // 2),
            nn.BatchNorm1d(num_filters1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters1, num_filters2, kernel_size=kernel_size2, padding=kernel_size2 // 2),
            nn.BatchNorm1d(num_filters2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(num_filters2, num_filters2, kernel_size=kernel_size3, padding=kernel_size3 // 2),
            nn.BatchNorm1d(num_filters2),
            nn.ReLU(),

            nn.AdaptiveMaxPool1d(1),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.class_head = nn.Linear(hidden_dim, 1)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)

        class_logit = self.class_head(x).squeeze(1)
        reg_scaled = self.reg_head(x).squeeze(1)

        return class_logit, reg_scaled

def get_model_params(checkpoint):
    """Extract architecture parameters from saved checkpoint."""
    if "model_params" in checkpoint:
        params = checkpoint["model_params"].copy()
    elif "model_config" in checkpoint:
        cfg = checkpoint["model_config"]
        params = {
            "num_filters1": cfg["num_filters1"],
            "num_filters2": cfg["num_filters2"],
            "kernel_size1": cfg["kernel_size1"],
            "kernel_size2": cfg["kernel_size2"],
            "kernel_size3": cfg.get("kernel_size3", 5),
            "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"],
        }
    else:
        raise KeyError("Checkpoint missing model parameters.")

    params.setdefault("kernel_size3", 5)
    return params

def read_test_data(path):
    """Load test dataset and detect identifier column."""
    df = pd.read_csv(path, sep="\t")

    if "id" in df.columns:
        id_col = "id"
    elif "seq_id" in df.columns:
        id_col = "seq_id"
    else:
        raise ValueError("Missing ID column ('id' or 'seq_id').")

    if SEQ_COL not in df.columns:
        raise ValueError("Missing 'sequence' column.")

    return df, id_col

def main():
    parser = argparse.ArgumentParser(description="Run model inference on DNA sequences.")
    parser.add_argument("model_path")
    parser.add_argument("test_data")
    args = parser.parse_args()

    # Load trained model checkpoint
    checkpoint = torch.load(args.model_path, map_location="cpu")

    max_len = int(checkpoint["max_len"])
    reg_mean = float(checkpoint["reg_mean"])
    reg_std = float(checkpoint["reg_std"])
    threshold = float(checkpoint.get("threshold", 0.5))

    model_params = get_model_params(checkpoint)

    model = DNARegulatoryCNN(model_params)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load test data
    # test_df, id_col = read_test_data(args.test_data)

    with open(args.test_data, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        header = lines[0][1:] if lines[0].startswith('>') else ''
        sequence = ''.join(lines[1:]) if len(lines) > 1 else ''
    df = pd.DataFrame([{'seq_id': header, 'sequence': sequence}])

    # Prepare output file (named after model)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    output_path = f"{model_name}_predictions.tsv"
    output_file = open(output_path, "w")

    header = "id\tpredicted_is_active\tpredicted_rna_dna_ratio"
    print(header)
    output_file.write(header + "\n")

    # Run inference
    with torch.no_grad():
        for _, row in df.iterrows():
            x = one_hot_encode(row[SEQ_COL], max_len=max_len)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

            class_logit, reg_scaled = model(x)

            # Classification prediction
            prob = torch.sigmoid(class_logit).item()
            predicted_is_active = int(prob >= threshold)

            # Regression prediction (reverse normalization)
            predicted_ratio = float(reg_scaled.item() * reg_std + reg_mean)

            line = f"{row['seq_id']}\t{predicted_is_active}\t{predicted_ratio:.6f}"

            print(line)
            output_file.write(line + "\n")

    output_file.close()

if __name__ == "__main__":
    main()
