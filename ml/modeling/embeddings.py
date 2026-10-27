"""Binary dysplasia classifier on precomputed tile embeddings (hunCRC).

Port of the IBD feat/modeling embeddings experiment, adapted to the hunCRC
label rule: one binary target (dysplasia vs no-dysplasia, the union of the
lowgrade/highgrade tiling columns -- the same target as the U-Net
experiment) instead of IBD's two percentage regressions.
"""

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import BinaryAUROC

from ml.modeling.mlp import MLP


class DysplasiaEmbeddingsClassifier(LightningModule):
    """MLP: 1536-d tile embedding -> dysplasia logit.

    forward returns LOGITS (BCEWithLogitsLoss); metrics and predictions
    sigmoid them.
    """

    def __init__(
        self,
        dims: list[int] | tuple[int, ...] = (1536, 512, 256, 1),
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.regressor = MLP(*dims)
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.val_metrics = MetricCollection(
            {"auc": BinaryAUROC(), "acc": Accuracy(task="binary")},
            prefix="validation/",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test/")

    def forward(self, x: Tensor) -> Tensor:
        return self.regressor(x)

    def training_step(self, batch: tuple) -> Tensor:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, labels.float())
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple) -> None:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        logits = self(x).squeeze(-1)
        loss = self.criterion(logits, labels.float())
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)

        probs = logits.sigmoid()
        self.val_metrics.update(probs, labels.long())
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch: tuple) -> None:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        logits = self(x).squeeze(-1)
        probs = logits.sigmoid()
        self.test_metrics.update(probs, labels.long())
        self.log_dict(self.test_metrics, on_epoch=True)

    def predict_step(self, batch: tuple) -> Tensor:
        return self(batch[0]).sigmoid()

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.lr)
