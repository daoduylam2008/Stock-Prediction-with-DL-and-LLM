import torch
import torch.nn as nn


class LSTM(nn.Module): 
    def __init__(self, n_features: int, hidden_size: int = 100,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,        # input shape: (batch, seq, features)
            dropout      = dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden_size)
        out    = out[:, -1, :]          # last time-step: (batch, hidden_size)
        out    = self.dropout(out)
        out    = self.fc(out)           # (batch, 1)
        return out


class ANN(nn.Module):
    def __init__(self, n_features: int, seq_len: int = 30,
                 hidden1: int = 100, hidden2: int = 50, dropout: float = 0.2):
        super().__init__()
        input_dim = n_features * seq_len   # flatten the full window

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = x.view(x.size(0), -1)    # flatten → (batch, seq_len * n_features)
        return self.net(x)            # (batch, 1)


class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, lstm_pred, ann_pred):
        # Concatenate both predictions along feature dim
        combined = torch.cat([lstm_pred, ann_pred], dim=1)  # (batch, 2)
        return self.fc(combined)                             # (batch, 1)


if __name__ == "__main__":
    BATCH, SEQ, FEAT = 16, 30, 7

    dummy = torch.randn(BATCH, SEQ, FEAT)

    lstm_model = LSTM(n_features=FEAT)
    ann_model  = ANN(n_features=FEAT, seq_len=SEQ)
    meta_model = MetaModel()

    with torch.no_grad():
        lstm_out = lstm_model(dummy)
        ann_out  = ann_model(dummy)
        meta_out = meta_model(lstm_out, ann_out)

    print(f"LSTM output shape : {lstm_out.shape}")   # (16, 1)
    print(f"ANN  output shape : {ann_out.shape}")    # (16, 1)
    print(f"Meta output shape : {meta_out.shape}")   # (16, 1)