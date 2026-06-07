import os
import re
import glob
import csv
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

DIRECTORY = "runs_stochastic"
OUTPUT_DIRECTORY = "combined"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
seq_no = "4"

def extract_info_from_filename(filename):
    m = re.search(r'iter(\d+)_win(\d+)', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def parse_mutation(mutation_str):
    m = re.match(r'([ACGT])(\d+)([ACGT])', mutation_str)
    if m:
        ref, pos, alt = m.groups()
        return int(pos)+1, ref, alt
    else:
        return None, None, None

def collect_records(tsv_file):
    max_iteration, window = extract_info_from_filename(tsv_file)
    records = []
    with open(tsv_file, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) < 4:
                continue
            iteration = int(cols[0])
            mutation = cols[3]
            pos, ref, alt = parse_mutation(mutation)
            if pos is not None:
                record = {
                    'position': pos,
                    'from': ref,
                    'to': alt,
                    'max_iteration': max_iteration,
                    'window': window,
                    'actual_iteration': iteration,
                }
                records.append(record)
    return records

def get_original_nts_from_fasta(fasta_filename):
    """Returns a dict mapping (1-based) position to nt (e.g. {1: 'A', 2: 'C', ...})"""
    with open(fasta_filename) as f:
        lines = f.readlines()
    seq = ""
    for line in lines:
        if not line.startswith(">"):
            seq += line.strip().upper()
    # seq4.fs: positions 1-based, code expects that
    pos_to_nt = {i+1: nt for i, nt in enumerate(seq)}
    return pos_to_nt

def main():
    # Load original nts from FASTA or .fs file
    original_seq_file = f"seq{seq_no}.fa"
    pos_to_original_nt = get_original_nts_from_fasta(original_seq_file)

    
    path_pattern = f"{DIRECTORY}/seq_{seq_no}_*_A.tsv"
    tsv_files = glob.glob(path_pattern)
    
    all_records = []
    for tsv in tsv_files:
        recs = collect_records(tsv)
        all_records.extend(recs)

    # Group by position, then count (from, to) combinations
    position_to_changes = defaultdict(Counter)
    for rec in all_records:
        position = rec["position"]
        from_nt = rec["from"]
        to_nt = rec["to"]
        position_to_changes[position][(from_nt, to_nt)] += 1

    # Determine all possible from-to changes present, to determine columns of output
    all_changes = set()
    for changes_counter in position_to_changes.values():
        all_changes.update(changes_counter.keys())
    all_changes = sorted(all_changes, key=lambda ft: (ft[0], ft[1]))  # sorted for consistency

    # Prepare header for output CSV (now includes original_nt field)
    output_fields = ["position", "original_nt"] + [f"{f}_to_{t}" for (f, t) in all_changes]

    # Write grouped and counted output
    output_file = f"{OUTPUT_DIRECTORY}/seq{seq_no}_count_by_position.csv"
    with open(output_file, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_fields, delimiter=',')
        writer.writeheader()
        for position in sorted(position_to_changes.keys()):
            # Get the original nt (use .get for positions that may be missing in the .fs file)
            original_nt = pos_to_original_nt.get(position, "")
            row = {'position': position, 'original_nt': original_nt}
            for (f, t) in all_changes:
                row[f"{f}_to_{t}"] = position_to_changes[position][(f, t)]
            writer.writerow(row)
    
    # ---- Plotting section ----
    # Always plot from 0 to 230, even if data only covers a subset
    plot_min, plot_max = 0, 230
    num_positions = plot_max - plot_min + 1
    full_positions = np.arange(plot_min, plot_max + 1)

    # Build a lookup for position to row
    position_lookup = {}
    mutation_types = [f"{f}_to_{t}" for (f, t) in all_changes]
    for mut in mutation_types:
        position_lookup[mut] = np.zeros(num_positions, dtype=int)

    original_nts_list = [''] * num_positions

    # Read the output file and fill mapped positions in full arrays
    with open(output_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = int(row["position"])
            idx = pos - plot_min  # so if pos=0, idx=0; if pos=1, idx=1
            if 0 <= idx < num_positions:
                original_nts_list[idx] = row["original_nt"]
                for mut in mutation_types:
                    position_lookup[mut][idx] = int(row[mut])

    # Prepare data for heatmap (rows: positions 0..230, columns: mutation_types)
    data = np.zeros((num_positions, len(mutation_types)), dtype=int)
    for j, mut in enumerate(mutation_types):
        data[:, j] = position_lookup[mut]

    # Plot as heatmap: rows are positions 0..230
    plt.figure(figsize=(max(10, len(mutation_types)*0.7), max(5, num_positions*0.12)))
    im = plt.imshow(data, aspect='auto', interpolation='nearest', cmap='plasma', origin='upper')

    plt.colorbar(im, label='Count')
    plt.yticks(np.arange(0, num_positions, max(1, num_positions // 20)), np.arange(plot_min, plot_max + 1, max(1, num_positions // 20)))
    plt.xticks(np.arange(len(mutation_types)), mutation_types, rotation=45, ha='right')
    plt.xlabel('Mutation (from_to)')
    plt.ylabel('Position')
    plt.title(f'Mutation Counts by Position for seq{seq_no}')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIRECTORY}/seq{seq_no}_count_by_position.png", dpi=200)
    plt.close()
    print(f"Plot saved to {OUTPUT_DIRECTORY}/{seq_no}_count_by_position.png")

    # ---- Additional Plot: Stacked Bar Chart of Changes per Position ----
    # Plot a stacked bar chart showing, for each position (0..230), counts of each mutation type

    fig, ax = plt.subplots(figsize=(max(10, num_positions * 0.05), 7))
    bottom = np.zeros(num_positions, dtype=int)
    color_map = plt.get_cmap('tab20')
    colors = [color_map(i % 20) for i in range(len(mutation_types))]
    for i, mut in enumerate(mutation_types):
        counts = position_lookup[mut]
        ax.bar(full_positions, counts, bottom=bottom, label=mut, color=colors[i], width=1.0)
        bottom += counts

    ax.set_xlim(plot_min, plot_max)
    ax.set_xlabel('Position')
    ax.set_ylabel('Count')
    ax.set_title(f'Stacked Mutation Counts by Position for seq{seq_no}')
    ax.legend(title="Mutation", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small', borderaxespad=0)
    plt.tight_layout()
    barplot_file = f"{OUTPUT_DIRECTORY}/seq{seq_no}_mutation_stack_by_position.png"
    plt.savefig(barplot_file, dpi=200)
    plt.close()
    print(f"Stacked bar plot saved to {barplot_file}")

if __name__ == "__main__":
    main()