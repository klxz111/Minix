import torch.nn as nn
import torch.nn.functional as F


class LSTMLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=2, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout if n_layer > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        x, _ = self.lstm(x)
        x = self.ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss