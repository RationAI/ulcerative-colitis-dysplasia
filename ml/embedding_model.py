from torch import Tensor

from ml.base_model import BaseModel
from ml.modeling.decode_head import EmbeddingClassifier


class EmbeddingModel(BaseModel):
    def __init__(self, decode_head: EmbeddingClassifier, lr: float) -> None:
        super().__init__(decode_head=decode_head, lr=lr)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.decode_head(x)
        return logits.squeeze(1)
