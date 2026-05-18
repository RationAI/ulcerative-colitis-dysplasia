from torch import Tensor

from ml.base_model import BaseModel
from ml.modeling.decode_head import EmbeddingClassifier


class EmbeddingModel(BaseModel):
    def __init__(self, decode_head: EmbeddingClassifier, lr: float) -> None:
        self.decode_head = decode_head
        super().__init__(lr=lr)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.decode_head(x)
        return logits
