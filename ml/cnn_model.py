from torch import Tensor, nn

from ml.base_model import BaseModel
from ml.modeling.decode_head import CNNClassifier


class CNNModel(BaseModel):
    def __init__(
        self, backbone: nn.Module, decode_head: CNNClassifier, lr: float
    ) -> None:
        self.backbone = backbone
        self.decode_head = decode_head
        super().__init__(lr=lr)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        logits = self.decode_head(features)
        return logits
