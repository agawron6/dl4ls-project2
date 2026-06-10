import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# model
class DeepSTARR_Lite(nn.Module):
    def __init__(self, seq_len):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.head_class = nn.Linear(128, 1)
        self.head_reg   = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)

        shared = self.shared(x)

        return self.head_class(shared), self.head_reg(shared)

# slownik
def seq_to_one_hot(seq):
    mapping = {
        'A':[1,0,0,0],
        'C':[0,1,0,0],
        'G':[0,0,1,0],
        'T':[0,0,0,1],
        'N':[0,0,0,0]
    }
    return np.array([mapping.get(b.upper(), [0,0,0,0]) for b in seq], dtype=np.float32).T

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Użycie: python evaluation_script.py <model_path> <test_data.tsv>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_path  = sys.argv[2]

    device = torch.device("cpu")  

    try:
        # df = pd.read_csv(test_path, sep='\t')
        with open(test_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            header = lines[0][1:] if lines[0].startswith('>') else ''
            sequence = ''.join(lines[1:]) if len(lines) > 1 else ''
        df = pd.DataFrame([{'seq_id': header, 'sequence': sequence}])

        id_column = 'id' if 'id' in df.columns else 'seq_id'

        seq_length = len(df['sequence'].iloc[0])

        
        model = DeepSTARR_Lite(seq_length)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")

        with torch.no_grad():
            for _, row in df.iterrows():
                seq_id = row[id_column]
                seq = row['sequence']

                x = torch.tensor(seq_to_one_hot(seq)).unsqueeze(0).to(device)

                out_class, out_reg = model(x)

                prob = torch.sigmoid(out_class).item()
                pred_class = 1 if prob > 0.5 else 0

                pred_reg = out_reg.item()

                pred_reg = max(0.0, pred_reg)

                print(f"{seq_id}\t{pred_class}\t{pred_reg:.4f}")

    except Exception as e:
        print(f"Błąd w czasie ewaluacji: {e}", file=sys.stderr)
        sys.exit(1)

