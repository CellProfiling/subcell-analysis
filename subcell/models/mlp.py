"""Grid-searched MLP probe, trained on frozen embeddings.

Used by ``subcell.classification.nested_cv`` (GO / Reactome membership, Fig. 3d)
and ``subcell.perturbation.cm4ai_drug_prediction``. The fixed two-layer probe of
the localization and cell-cycle benchmarks is ``subcell.classification.mlp``.
"""

from torch import nn


class MLPClassifier(nn.Module):
    """Dropout on the input, ``n_layers`` hidden blocks of
    Linear-Dropout-ReLU-BatchNorm, then a linear output layer.

    ``n_layers == 1`` is the exception: no hidden block, so the probe is dropout
    followed by one linear map and ``hidden_dim`` is ignored. The stacked trainer
    in ``nested_cv`` relies on this layout.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=512, n_layers=1, dropout=0.5):
        super().__init__()

        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        if n_layers > 1:
            for i in range(n_layers):
                setattr(self, f"fc{i}", nn.Linear(input_dim, hidden_dim))
                setattr(self, f"dropout{i}", nn.Dropout(dropout))
                setattr(self, f"nonlin{i}", nn.ReLU())
                setattr(self, f"norm{i}", nn.BatchNorm1d(hidden_dim))
                input_dim = hidden_dim

        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        X = self.dropout(X)
        if self.n_layers > 1:
            for i in range(self.n_layers):
                X = getattr(self, f"fc{i}")(X)
                X = getattr(self, f"dropout{i}")(X)
                X = getattr(self, f"nonlin{i}")(X)
                X = getattr(self, f"norm{i}")(X)
        X = self.fc_out(X)
        return X
