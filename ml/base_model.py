from copy import deepcopy
from typing import Any, Literal, cast

from lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Sigmoid, Softmax
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import (
    AUROC,
    Accuracy,
    CohenKappa,
    Metric,
    MetricCollection,
    Precision,
    Recall,
    Specificity,
)

from ml.modeling.decode_head import BaseClassifier
from ml.typing import Input, PredictInput


class BaseModel(LightningModule):
    decode_head: BaseClassifier

    def __init__(self, lr: float) -> None:
        super().__init__()
        self.lr = lr

        num_classes = self.decode_head.out_features
        self.criterion = BCEWithLogitsLoss() if num_classes == 1 else CrossEntropyLoss()
        self.activation = Sigmoid() if num_classes == 1 else Softmax(dim=1)

        task: Literal["binary", "multiclass"] = (
            "binary" if num_classes == 1 else "multiclass"
        )
        metrics: dict[str, Metric | MetricCollection] = {
            "AUC": AUROC(task=task, num_classes=num_classes, average="none"),
            "accuracy": Accuracy(task=task, num_classes=num_classes),
            "precision": Precision(task=task, num_classes=num_classes, average="none"),
            "recall": Recall(task=task, num_classes=num_classes, average="none"),
            "specificity": Specificity(
                task=task, num_classes=num_classes, average="none"
            ),
            "kappa": CohenKappa(task=task, num_classes=num_classes),
        }

        self.train_metrics = MetricCollection(
            metrics=deepcopy(metrics), prefix="train/"
        )

        self.val_metrics = MetricCollection(
            metrics=deepcopy(metrics), prefix="validation/"
        )

        self.test_metrics = MetricCollection(metrics=deepcopy(metrics), prefix="test/")

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def training_step(self, batch: Input) -> Tensor:
        inputs, targets, _ = batch
        logits = self(inputs)
        predictions = self.activation(logits)

        loss = self.criterion(logits, targets)
        self.log(
            "train/loss",
            loss,
            batch_size=len(inputs),
            on_step=True,
            prog_bar=True,
        )

        self.train_metrics.update(predictions, targets)
        self.log_dict(self.train_metrics, batch_size=len(inputs), on_epoch=True)

        return loss

    def validation_step(self, batch: Input) -> None:
        inputs, targets, _ = batch
        logits = self(inputs)
        predictions = self.activation(logits)

        loss = self.criterion(logits, targets)
        self.log(
            "validation/loss",
            loss,
            batch_size=len(inputs),
            on_epoch=True,
            prog_bar=True,
        )

        self.val_metrics.update(predictions, targets)
        self.log_dict(self.val_metrics, batch_size=len(inputs), on_epoch=True)

    def test_step(self, batch: Input) -> Tensor:
        inputs, targets, _ = batch
        logits = self(inputs)
        predictions = self.activation(logits)

        self.test_metrics.update(predictions, targets)
        self.log_dict(self.test_metrics, batch_size=len(inputs), on_epoch=True)

        return predictions

    def predict_step(self, batch: PredictInput) -> Tensor:
        return self.activation(self(batch[0]))

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), self.lr)

    def log_dict(self, dictionary: MetricCollection, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        for name, result in dictionary.compute().items():
            result = cast("Tensor", result)
            if result.shape:
                for i, value in enumerate(result):
                    self.log(f"{name}/{i}", value, *args, **kwargs)
            else:
                self.log(name, result, *args, **kwargs)
