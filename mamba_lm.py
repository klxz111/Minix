import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mixer(self.ln(x)))


class MambaLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layer=4,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
        tie_weights=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tie_weights = tie_weights

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if self.tie_weights:
            self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)          # [B, T, C]
        x = self.emb_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss