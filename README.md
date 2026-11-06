# DL4LS Project 2

This repository contains a sequence-optimization and evaluation workflow for DNA regulatory activity prediction. The project compares several deep learning models that score DNA sequences for activity and RNA/DNA ratio, then applies stochastic mutation search to improve candidate sequences based on model gradients and local saliency.

## Overview

The codebase includes:

- A set of model evaluation scripts for different architectures:
  - `evaluation_script_A.py` — DeepSTARR-style model
  - `evaluation_script_J.py` — lighter DeepSTARR variant
  - `evaluation_script_M.py` — CNN-based regulatory model
- A stochastic optimization pipeline in `b.py` that:
  - reads a DNA sequence from a FASTA file
  - evaluates it with one or more trained models
  - computes gradient saliency
  - selects high-impact mutation windows
  - iteratively applies improvements and writes optimization logs
- A mutation aggregation script `combine.py` that summarizes mutation counts by genomic position across runs
- A notebook `dl4ls_MAJ.ipynb` for exploratory analysis and model evaluation

## Repository Structure

- `evaluation_script_A.py` — inference for model A
- `evaluation_script_J.py` — inference for model J
- `evaluation_script_M.py` — inference for model M
- `b.py` — sequence optimization and trajectory generation
- `combine.py` — combine stochastic run outputs into position-wise mutation tables
- `dl4ls_MAJ.ipynb` — project notebook
- `seq1.fa`, `seq2.fa`, `seq3.fa`, `seq4.fa`, `sequences.fa` — example DNA sequences
- `model_A.pth`, `model_J.pth`, `model_M.pth` — trained model checkpoints
- `runs_stochastic/` — generated optimization logs and mutation outputs
- `mutacje.xlsx` — mutation data spreadsheet

## Model and Data Format

The sequence files are FASTA files with one header and a DNA sequence. Example:

```text
>example_sequence
ACGTGCTTAGCA...
```

The evaluation scripts expect a FASTA-style input and output predictions in TSV form:

```text
id	predicted_is_active	predicted_rna_dna_ratio
```

## Running inference

You can run each model on a sequence file directly.

Example for the DeepSTARR model:

```bash
python evaluation_script_A.py model_A.pth seq1.fa
```

Example for the lightweight model:

```bash
python evaluation_script_J.py model_J.pth seq2.fa
```

Example for the CNN model:

```bash
python evaluation_script_M.py model_M.pth seq3.fa
```

## Running the stochastic optimization workflow

The optimization pipeline is started via:

```bash
python b.py
```

This script loads one of the input FASTA sequences, evaluates it against the A/J/M models, and generates per-iteration mutation logs saved in the `runs_stochastic/` directory. The script is configured for the sequence and model paths used in this project and may need small path adjustments depending on your local environment.

## Aggregating mutation results

After running the optimization workflow, you can aggregate the generated TSV files into position-level mutation counts:

```bash
python combine.py
```

This produces CSV summaries in the `combined/` directory, grouped by sequence position and mutation type.

## Requirements

This project uses:

- Python 3
- PyTorch
- NumPy
- pandas
- matplotlib

Install dependencies with:

```bash
pip install torch numpy pandas matplotlib
```

## Project goal

The main objective is to predict whether a DNA regulatory sequence is active and to optimize sequence variants that improve model scoring. The repository combines model inference, gradient-based mutation search, and post-hoc mutation statistics to analyze sequence-function relationships.

## Notes

- The project is strongly centered on sequence analysis and model-driven optimization for biological/regulatory DNA.
- Several scripts are tailored to the repository's local paths and dataset conventions and may require minor edits when reused in a different environment.
- The notebook (`dl4ls_MAJ.ipynb`) captures the broader analysis and experimentation flow used during the project.

## License

This repository does not currently declare a license.
