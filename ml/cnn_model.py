from torch import Tensor, nn

from ml.base_model import BaseModel
from ml.modeling.decode_head import CNNClassifier


class CNNModel(BaseModel):
    def __init__(
        self, backbone: nn.Module, decode_head: CNNClassifier, lr: float
    ) -> None:
        super().__init__(decode_head=decode_head, lr=lr)
        self.backbone = backbone

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        logits = self.decode_head(features)
        return logits.squeeze(1)
