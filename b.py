import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# 1. SETUP AND UTILITIES
# -------------------------------------------------------------------------
NUCLEOTIDES = ['A', 'C', 'G', 'T']
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def one_hot_encode(seq: str) -> torch.Tensor:
    """Encodes DNA sequence to match DeepSTARR dimension expectations (4 x L)."""
    ohe = torch.zeros((4, len(seq)), dtype=torch.float32)
    for i, char in enumerate(seq.upper()):
        if char in NUC_TO_IDX:
            ohe[NUC_TO_IDX[char], i] = 1.0
    return ohe.to(device)

def one_hot_decode(tensor: torch.Tensor) -> str:
    """Decodes a 4 x L tensor back into standard DNA string representation."""
    indices = torch.argmax(tensor, dim=0)
    return "".join([NUCLEOTIDES[idx.item()] for idx in indices])

# -------------------------------------------------------------------------
# 2. SINGLE-MODEL EVALUATION FUNCTION
# -------------------------------------------------------------------------
def evaluate_single_model(input_tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Evaluates the input_tensor using the appropriate evaluation strategy
    based on the model architecture, mimicking the output structure and normalization
    logic of either evaluation_script_A.py, evaluation_script_J.py, or evaluation_script_M.py.
    Always returns the *raw regression output* (rna_dna_ratio) as used for optimization.
    """

    model_name = type(model).__name__
    model_device = next(model.parameters()).device
    x = input_tensor.unsqueeze(0).to(model_device)

    with torch.set_grad_enabled(True):
        predicted = model(x)

    # --- Model from evaluation_script_A.py (DeepSTARR) ---
    if model_name == "DeepSTARR":
        # predicted = (is_active, ratio)
        _, ratio_norm = predicted if isinstance(predicted, tuple) else (None, predicted)
        # Use train mean and std from script A, as in their printout
        train_mean = -0.22148310042248945
        train_std = 0.7843937844681367
        output_real = ratio_norm * train_std + train_mean
        return output_real.squeeze()

    # --- Model from evaluation_script_J.py (DeepSTARR_Lite) ---
    elif model_name == "DeepSTARR_Lite":
        # predicted = (out_class, out_reg)
        _, out_reg = predicted if isinstance(predicted, tuple) else (None, predicted)
        # In script J, pred_reg = max(0, out_reg.item()) for the *output*,
        # but for optimization we use the *raw* regression output:
        return out_reg.squeeze()

    # --- Model from evaluation_script_M.py (DNARegulatoryCNN) ---
    elif model_name == "DNARegulatoryCNN":
        # predicted = (class_logit, reg_scaled)
        _, reg_scaled = predicted if isinstance(predicted, tuple) else (None, predicted)
        # Extract normalization stats from model if available
        reg_mean = float(getattr(model, "reg_mean", 0.0))
        reg_std = float(getattr(model, "reg_std", 1.0))
        if hasattr(model, "_reg_stats"):
            reg_mean, reg_std = model._reg_stats
            reg_mean = float(reg_mean)
            reg_std = float(reg_std)
        # Some checkpoints may attach these as attributes
        output_real = reg_scaled * reg_std + reg_mean
        return output_real.squeeze()

    # --- Unknown/fallback: try tuple[1] or direct output ---
    else:
        pred = predicted[1] if isinstance(predicted, tuple) and len(predicted) > 1 else predicted
        return pred.squeeze()

# -------------------------------------------------------------------------
# 3. SINGLE-MODEL ITERATIVE OPTIMIZATION LOOP
# -------------------------------------------------------------------------
def optimize_sequence_single(initial_seq: str, model: nn.Module, max_iterations: int = 50, scan_window_size: int = 10, device=None):
    """
    Iteratively optimizes a sequence using gradient attribution and
    in silico scanning against a single model.
    """
    model.to(device)
    current_tensor = one_hot_encode(initial_seq)
    # print(current_tensor)
    optimization_history = []

    print(f"Initial Sequence Length: {current_tensor.shape[1]} bp")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- BEGIN ITERATION {iteration:03d} ---")
        # Step A: Compute Gradients for Saliency Mapping
        current_tensor.requires_grad = True
        current_score_tensor = evaluate_single_model(current_tensor, model)
        baseline_score = current_score_tensor.item()

        print(f"[BASE EVAL] Current Sequence Predicted Real Ratio: {baseline_score:.6f}")

        # Backpropagate to extract nucleotide sensitivity
        model.zero_grad()
        if current_tensor.grad is not None:
            current_tensor.grad.zero_()
        current_score_tensor.backward()
        gradients = current_tensor.grad

        # Calculate absolute maximum gradient across the 4 bases for each position
        position_salience = torch.max(torch.abs(gradients), dim=0)[0]

        # Isolate the top N positions most sensitive to point mutations
        top_positions = torch.topk(position_salience, k=scan_window_size).indices.tolist()

        print(f"[SALIENCY] Global Max Gradient Magnitude: {torch.max(position_salience).item():.6f}")
        print(f"[SALIENCY] Isolated Top-{scan_window_size} High-Sensitivity Coordinates: {top_positions}")

        best_mutation_score = baseline_score
        best_tensor = current_tensor.detach().clone()
        chosen_mutation = None

        # Step B: Targeted In Silico Scanning
        print(f"\n[TIER 1] Launching Localized In Silico Scan on Top Positions...")
        current_tensor_detached = current_tensor.detach().clone()
        for pos in top_positions:
            current_base_idx = torch.argmax(current_tensor_detached[:, pos]).item()

            # Diagnostic string to collect alternative scores for this position
            pos_trace = []

            for alt_idx in range(4):
                wt_type = NUCLEOTIDES[current_base_idx]
                alt_type = NUCLEOTIDES[alt_idx]
                if alt_idx == current_base_idx:
                    continue # Skip the wild-type residue

                # Propose mutation variant
                mutated_tensor = current_tensor_detached.clone()
                mutated_tensor[:, pos] = 0.0
                mutated_tensor[alt_idx, pos] = 1.0

                # Evaluate variant performance
                with torch.no_grad():
                    variant_score = evaluate_single_model(mutated_tensor, model).item()

                delta = variant_score - baseline_score
                pos_trace.append(f"{alt_type}: {variant_score:.4f} (Δ:{delta:+.4f})")

                # If variant improves the single model score, track it
                if variant_score > best_mutation_score:
                    best_mutation_score = variant_score
                    best_tensor = mutated_tensor.clone()
                    chosen_mutation = {
                        "position": pos,
                        "from": NUCLEOTIDES[current_base_idx],
                        "to": NUCLEOTIDES[alt_idx]
                    }
            # print(f"  Coord {pos:03d} [WT: {wt_type} -> Variants evaluated: {', '.join(pos_trace)}")
        # Step C: State Update and Logging
        if chosen_mutation is not None:
            current_tensor = best_tensor.clone()
            current_seq_str = one_hot_decode(current_tensor)

            optimization_history.append({
                "iteration": iteration,
                "score": best_mutation_score,
                "sequence": current_seq_str,
                "mutation": f"{chosen_mutation['from']}{chosen_mutation['position']}{chosen_mutation['to']}"
            })
            print(f"Iteration {iteration:03d} | Score: {best_mutation_score:.4f} | Mutation: {optimization_history[-1]['mutation']}")
        else:
            print(f"Iteration {iteration:03d} | Convergence reached. No single-model optimization observed.")
            break

    return optimization_history

# -------------------------------------------------------------------------
# 4. RUN PIPELINE
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Setup device (gpu if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # with open("seq1.fa", 'r') as f:
    #     # og: ~2.5 40 mutations
    #     og = 2.5
    #     limit=40
    #     lines = [line.strip() for line in f if line.strip()]
    #     header = lines[0][1:] if lines[0].startswith('>') else ''
    #     input_sequence = ''.join(lines[1:]) if len(lines) > 1 else ''


    # with open("seq2.fa", 'r') as f:
    #     # og: ~2.7 16 mutations
    #     og = 2.7
    #     limit=16
    #     lines = [line.strip() for line in f if line.strip()]
    #     header = lines[0][1:] if lines[0].startswith('>') else ''
    #     input_sequence = ''.join(lines[1:]) if len(lines) > 1 else ''

    # with open("seq3.fa", 'r') as f:
    #     # og: ~1.1 20 mutations
    #     og = 1.1
    #     limit=20
    #     lines = [line.strip() for line in f if line.strip()]
    #     header = lines[0][1:] if lines[0].startswith('>') else ''
    #     input_sequence = ''.join(lines[1:]) if len(lines) > 1 else ''
    
    with open("seq4.fa", 'r') as f:
        # none
        limit=None
        lines = [line.strip() for line in f if line.strip()]
        header = lines[0][1:] if lines[0].startswith('>') else ''
        input_sequence = ''.join(lines[1:]) if len(lines) > 1 else ''

    # Import model
    from evaluation_script_A import DeepSTARR
    # Set the sequence length according to your expected input (230 in evaluation_script)
    seq_len = 230
    my_model = DeepSTARR(seq_len)
    my_model.load_state_dict(torch.load("/Users/alicja/Downloads/Alicja-Gawron/model_A.pth", map_location=device))
    my_model.to(device)
    my_model.eval()

    # Execute optimization pipeline, making sure data is on correct device

    # Parameters
    max_iter = limit if limit is not None else 43
    for scan_window_size in [20,25,30]:

        log_data = optimize_sequence_single(input_sequence, my_model, max_iterations=max_iter, scan_window_size=scan_window_size, device=device)

        if log_data and 'sequence' in log_data[-1]:
            final_mutated_sequence = log_data[-1]['sequence']
            print("Final mutated sequence:", final_mutated_sequence)
            # You may also explicitly return it if this code is in a function:
            # return final_mutated_sequence
        else:
            final_mutated_sequence = None
            print("No mutated sequence found.")
        import datetime

        df_single_trajectory_A = pd.DataFrame(log_data)
        

        # Create a filename with short date, time and pipeline params
        now = datetime.datetime.now()
        time_stamp = now.strftime("%Y%m%d_%H%M%S")
        filename_A = f"runs/{header}_{time_stamp}_iter{max_iter}_win{scan_window_size}_A.tsv"

        from evaluation_script_J import DeepSTARR_Lite
        my_model_J = DeepSTARR_Lite(seq_len)
        my_model_J.load_state_dict(torch.load("/Users/alicja/Downloads/Alicja-Gawron/model_J.pth", map_location=device))
        my_model_J.to(device)
        my_model_J.eval()

        log_data_J = optimize_sequence_single(input_sequence, my_model_J, max_iterations=max_iter, scan_window_size=scan_window_size, device=device)

        if log_data_J and 'sequence' in log_data_J[-1]:
            final_mutated_sequence_J = log_data_J[-1]['sequence']
            print("Final mutated sequence:", final_mutated_sequence_J)
        else:
            final_mutated_sequence_J = None
            print("No mutated sequence found.")

        df_single_trajectory_J = pd.DataFrame(log_data_J)

        filename_J = f"runs/{header}_{time_stamp}_iter{max_iter}_win{scan_window_size}_J.tsv"

        from evaluation_script_M import DNARegulatoryCNN, get_model_params

        # Load model_M checkpoint
        model_m_ckpt_path = "/Users/alicja/Downloads/Alicja-Gawron/model_M.pth"
        checkpoint_M = torch.load(model_m_ckpt_path, map_location=device)
        try:
            model_params_M = get_model_params(checkpoint_M)
        except Exception as e:
            print("Could not extract model parameters from model_M checkpoint:", e)
            raise

        # Load max_len if present (useful if one-hot encoding needed elsewhere)
        seq_len_M = 230

        my_model_M = DNARegulatoryCNN(model_params_M)
        my_model_M.load_state_dict(checkpoint_M["model_state"])
        my_model_M.to(device)
        my_model_M.eval()

        log_data_M = optimize_sequence_single(input_sequence, my_model_M, max_iterations=max_iter, scan_window_size=scan_window_size, device=device)

        if log_data_M and 'sequence' in log_data_M[-1]:
            final_mutated_sequence_M = log_data_M[-1]['sequence']
            print("Final mutated sequence:", final_mutated_sequence_M)
        else:
            final_mutated_sequence_M = None
            print("No mutated sequence found.")

        df_single_trajectory_M = pd.DataFrame(log_data_M)

        filename_M = f"runs/{header}_{time_stamp}_iter{max_iter}_win{scan_window_size}_M.tsv"

        df_single_trajectory_A.to_csv(filename_A, sep='\t', index=False)
        print(f"Saved optimization log to {filename_A}")
        df_single_trajectory_J.to_csv(filename_J, sep='\t', index=False)
        print(f"Saved optimization log to {filename_J}")
        df_single_trajectory_M.to_csv(filename_M, sep='\t', index=False)
        print(f"Saved optimization log to {filename_M}")
