from torch import Tensor

from ml.modeling.decode_head.base_classifier import BaseClassifier


class EmbeddingClassifier(BaseClassifier):
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {x.ndim}D")

        x = self.dropout(x)
        x = self.proj(x)
        return x
