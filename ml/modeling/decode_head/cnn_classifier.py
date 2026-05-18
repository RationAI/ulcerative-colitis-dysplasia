from torch import Tensor, nn

from ml.modeling.decode_head.base_classifier import BaseClassifier


class CNNClassifier(BaseClassifier):
    def __init__(
        self, in_features: int, out_features: int, dropout: float = 0.5
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, dropout=dropout
        )
        self.global_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {x.ndim}D")

        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)
        x = self.dropout(x)
        x = self.proj(x)

        return x
