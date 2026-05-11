from copy import deepcopy

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCohenKappa,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchvision.models import vgg16, VGG16_Weights
from ulcerative_colitis.typing import (
    Output,
    SlideEmbeddingsInput,
    SlideEmbeddingsPredictInput,
)


class UlcerativeColitisModelVGG16(LightningModule):
    def __init__(self, lr: float | None = None, pretrained: bool = True) -> None:
        super().__init__()
        self.lr = lr

        # 1. Load VGG16
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.model = vgg16(weights=weights)

        # 2. Modify the classifier for Binary Classification
        # VGG16.classifier[6] is the final linear layer (4096 -> 1000)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, 1)

        self.criterion = nn.BCEWithLogitsLoss()  # More stable than Sigmoid + BCELoss

        metrics = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "kappa": BinaryCohenKappa(),
        }

        self.train_metrics = MetricCollection(deepcopy(metrics), prefix="train/")
        self.val_metrics = MetricCollection(deepcopy(metrics), prefix="validation/")
        self.test_metrics = MetricCollection(deepcopy(metrics), prefix="test/")

    def forward(self, x: Tensor) -> Output:
        # VGG16 returns logits; we apply sigmoid for the metrics/output
        return self.model(x).sigmoid()

    def training_step(self, batch: SlideEmbeddingsInput) -> Tensor:
        x, labels, _ = batch
        # For BCEWithLogitsLoss, we pass raw logits (outputs before sigmoid)
        logits = self.model(x).squeeze(1)
        labels = labels.float()

        loss = self.criterion(logits, labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        # Update metrics with sigmoid probabilities
        self.train_metrics.update(logits.sigmoid(), labels)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: SlideEmbeddingsInput) -> None:
        x, labels, _ = batch
        logits = self.model(x).squeeze(1)
        labels = labels.float()

        loss = self.criterion(logits, labels)
        self.log("validation/loss", loss, prog_bar=True)

        self.val_metrics.update(logits.sigmoid(), labels)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)

    def test_step(self, batch: SlideEmbeddingsInput) -> None:
        x, labels, _ = batch
        logits = self.model(x).squeeze(1)

        self.test_metrics.update(logits.sigmoid(), labels.float())
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False)

    def predict_step(self, batch: SlideEmbeddingsPredictInput) -> Output:
        return self.forward(batch[0])

    def configure_optimizers(self) -> Optimizer:
        if self.lr is None:
            raise ValueError("Learning rate must be set for training.")
        return Adam(self.parameters(), lr=self.lr)
