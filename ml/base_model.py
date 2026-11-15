from copy import deepcopy
from typing import Any, Literal, cast

from lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import (
    AUROC,
    Accuracy,
    Metric,
    MetricCollection,
    Precision,
    Recall,
    Specificity,
)

from ml.modeling.decode_head import BaseClassifier
from ml.typing import Input, MetadataBatch, PredictInput


class BaseModel(LightningModule):
    decode_head: BaseClassifier

    def __init__(self, decode_head: BaseClassifier, lr: float) -> None:
        super().__init__()
        self.decode_head = decode_head
        self.lr = lr
        self.predict_metadata: list[MetadataBatch] = []

        num_classes = self.decode_head.out_features
        self.criterion = BCEWithLogitsLoss()
        self.activation = Sigmoid()

        task: Literal["binary", "multilabel"] = (
            "binary" if num_classes == 1 else "multilabel"
        )
        kwargs: dict[str, Any] = {"task": task}
        if task == "binary":
            kwargs["num_classes"] = num_classes
        else:
            kwargs["num_labels"] = num_classes

        metrics: dict[str, Metric | MetricCollection] = {
            "AUC": AUROC(**kwargs, average="none"),
            "accuracy": Accuracy(**kwargs),
            "precision": Precision(**kwargs, average="none"),
            "recall": Recall(**kwargs, average="none"),
            "specificity": Specificity(**kwargs, average="none"),
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

        self.train_metrics.update(predictions, targets.long())
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

        self.val_metrics.update(predictions, targets.long())
        self.log_dict(self.val_metrics, batch_size=len(inputs), on_epoch=True)

    def test_step(self, batch: Input) -> Tensor:
        inputs, targets, _ = batch
        logits = self(inputs)
        predictions = self.activation(logits)

        self.test_metrics.update(predictions, targets.long())
        self.log_dict(self.test_metrics, batch_size=len(inputs), on_epoch=True)

        return predictions

    def predict_step(self, batch: PredictInput) -> Tensor:
        predictions = self.activation(self(batch[0]))
        self.predict_metadata.append(batch[1])
        return predictions

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
