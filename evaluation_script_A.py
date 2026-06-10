import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class DeepSTARR(nn.Module):
    def __init__(self, seq_len):
        super(DeepSTARR, self).__init__()

        self.conv1 = nn.Conv1d(4, 256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(256, 60, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(60, 120, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.spatial_dropout = nn.Dropout(0.1)

        flattened_dim = self._infer_flattened_dim(seq_len)
        self.fc1 = nn.Linear(flattened_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.classifier = nn.Linear(256, 1)  # is_active
        self.regressor = nn.Linear(256, 1)   # rna_dna_ratio

    def _infer_flattened_dim(self, seq_len):
        with torch.no_grad():
            x = torch.zeros(1, 4, seq_len)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            return x.flatten(1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.spatial_dropout(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        is_active = self.classifier(x)
        ratio = self.regressor(x)

        return is_active, ratio
    
def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    ohe = torch.zeros((4, len(seq)))
    for i, char in enumerate(seq):
        if char in mapping:
            ohe[mapping[char], i] = 1
    return ohe

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluation_script.py <path_to_model> <path_to_test_data>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSTARR(230)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # For a .fs (fasta) file with only one record, isolate the header and the sequence
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        header = lines[0][1:] if lines[0].startswith('>') else ''
        sequence = ''.join(lines[1:]) if len(lines) > 1 else ''
    df = pd.DataFrame([{'seq_id': header, 'sequence': sequence}])
    
    train_mean = -0.22148310042248945 
    train_std = 0.7843937844681367   

    print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")

    output_rows = []

    with torch.no_grad():
        for _, row in df.iterrows():
            seq_id = row['seq_id']
            seq = row['sequence']
            
            x = one_hot_encode(seq).unsqueeze(0).to(device)
            
            is_active_logits, ratio_norm = model(x)
            
            is_active = 1 if is_active_logits.item() > 0 else 0
            
            ratio_real = (ratio_norm.item() * train_std) + train_mean
            
            print(f"{seq_id}\t{is_active}\t{ratio_real:.6f}")
            output_rows.append([seq_id, is_active, f"{ratio_real:.6f}"])

    output_tsv = "predictions.tsv"
    with open(output_tsv, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["id", "predicted_is_active", "predicted_rna_dna_ratio"])
        writer.writerows(output_rows)

if __name__ == "__main__":
    main()