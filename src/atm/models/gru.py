"""A small CPU GRU over observed action tokens."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from atm.features import Prefix


class ActionGRU(nn.Module):
    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, cfg["gru_embedding"], padding_idx=0)
        self.gru = nn.GRU(cfg["gru_embedding"], cfg["gru_hidden"], num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(cfg["gru_dropout"])
        self.output = nn.Linear(cfg["gru_hidden"], 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(self.dropout(self.embedding(tokens)))
        return self.output(self.dropout(hidden[-1])).squeeze(-1)


def encode(rows: list[Prefix], vocabulary: tuple[str, ...]) -> torch.Tensor:
    token_to_id = {token: i + 1 for i, token in enumerate(vocabulary)}
    return torch.tensor([[token_to_id.get(token, 0) for token in row.tokens] for row in rows],
                        dtype=torch.int64)


@torch.no_grad()
def predict(model: ActionGRU, encoded: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    return torch.cat([torch.sigmoid(model(part)) for part in encoded.split(batch)]).numpy()


def fit(rows: torch.Tensor, labels: np.ndarray, vocab_size: int, cfg: dict, seed: int,
        epochs: int, validation: tuple[torch.Tensor, np.ndarray] | None = None) -> tuple[ActionGRU, int]:
    """Early-stop epochs only on the supplied grouped inner validation split."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = ActionGRU(vocab_size, cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["gru_lr"])
    targets = torch.tensor(labels, dtype=torch.float32)
    score, best_epoch = -1.0, epochs
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(rows), generator=rng).split(cfg["gru_batch"]):
            optim.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(rows[batch]), targets[batch])
            loss.backward()
            optim.step()
        if validation is not None:
            valid_x, valid_y = validation
            auc = roc_auc_score(valid_y, predict(model, valid_x, cfg["gru_batch"]))
            if auc > score + 1e-6:
                score, best_epoch = auc, epoch
            if epoch - best_epoch >= cfg["gru_patience"]:
                break
    return model, best_epoch
