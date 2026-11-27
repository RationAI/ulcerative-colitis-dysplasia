from collections.abc import Sequence

from torch import Tensor, nn

from ml.modeling.decode_head.base_classifier import BaseClassifier


class MlpEmbeddingClassifier(BaseClassifier):
    """Embedding head with hidden ReLU layers.

    ``EmbeddingClassifier`` is dropout plus a single ``Linear``, i.e. a linear
    probe. This head inserts hidden layers before the same output
    projection, so the two differ only in capacity and can be compared
    directly. ``hidden_dims=()`` makes it equivalent to the linear head.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.5,
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, dropout=dropout
        )
        dims = [in_features, *hidden_dims]
        self.hidden = nn.Sequential(
            *[
                module
                for i in range(len(dims) - 1)
                for module in (nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
            ]
        )
        #: Replaces the base projection so it maps from the last hidden dim.
        self.proj = nn.Linear(dims[-1], out_features)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {x.ndim}D")

        x = self.dropout(x)
        x = self.hidden(x)
        return self.proj(x)
