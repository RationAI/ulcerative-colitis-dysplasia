"""Compact U-Net + Lightning module for tile-level dysplasia segmentation.

Deliberately a plain conv U-Net (no backbone download, no extra deps): the
experiment's question is whether the tile crops carry dysplasia signal at all,
not backbone comparison. ~2 M params, 512 px in -> 512 px logits out.

Loss: weighted BCE + soft Dice, both restricted to `valid` pixels (white
padding outside the slide contributes nothing). Metrics on valid pixels only.
"""

from __future__ import annotations

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, JaccardIndex
from torchmetrics.collections import MetricCollection


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UNet(nn.Module):
    """4-level U-Net: 3ch RGB in, 1ch dysplasia logits out.

    `patch_size` must be divisible by 8 (four pool/upsample stages).
    """

    def __init__(self, in_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.encoder = nn.ModuleList(
            [_Block(in_channels, c[0]), _Block(c[0], c[1]), _Block(c[1], c[2]), _Block(c[2], c[3])]
        )
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(c[3], c[2], 2, stride=2),
                nn.ConvTranspose2d(c[2], c[1], 2, stride=2),
                nn.ConvTranspose2d(c[1], c[0], 2, stride=2),
            ]
        )
        self.neck = _Block(c[2] * 2, c[2])
        self.mid = _Block(c[1] * 2, c[1])
        self.head = _Block(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        e0 = self.encoder[0](x)
        p0 = self.pool(e0)
        e1 = self.encoder[1](p0)
        p1 = self.pool(e1)
        e2 = self.encoder[2](p1)
        p2 = self.pool(e2)
        e3 = self.encoder[3](p2)

        d2 = self.neck(torch.cat([self.up[0](e3), e2], dim=1))
        d1 = self.mid(torch.cat([self.up[1](d2), e1], dim=1))
        d0 = self.head(torch.cat([self.up[2](d1), e0], dim=1))
        return self.out(d0)


class UNetSeg(LightningModule):
    def __init__(
        self,
        base_channels: int = 32,
        in_channels: int = 3,
        dice_weight: float = 0.5,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet(in_channels, base_channels)
        self.val_metrics = MetricCollection(
            {
                "dice": BinaryF1Score(threshold=0.5),
                "iou": JaccardIndex(task="binary", threshold=0.5),
                "acc": BinaryAccuracy(threshold=0.5),
            },
            prefix="validation/",
        )

    def forward(self, x: Tensor) -> Tensor:
        #: keep (B, 1, H, W) -- matches target/valid; metric step squeezes.
        return self.net(x)

    def _loss(self, logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        bce = nn.BCEWithLogitsLoss(weight=valid)(logits, target)
        #: soft dice with padding zeroed out by the valid mask; the +1 smoothing
        #: makes an all-zero patch pair score 0 loss, so negatives need no
        #: special case.
        p = torch.sigmoid(logits).flatten(1) * valid.flatten(1)
        t = target.flatten(1)
        dice = 1.0 - (2.0 * (p * t).sum(1) + 1.0) / (p.sum(1) + t.sum(1) + 1.0)
        return (1.0 - self.hparams.dice_weight) * bce + self.hparams.dice_weight * dice.mean()

    def _step_metrics(self, logits: Tensor, target: Tensor, valid: Tensor) -> None:
        mask = (valid > 0.5).squeeze(1)
        pred = (logits > 0).float().squeeze(1)
        tgt = target.squeeze(1)
        for name, metric in self.val_metrics.items():
            metric.update(pred[mask].reshape(-1, 1), tgt[mask].reshape(-1, 1))
        self.log_dict(self.val_metrics, on_epoch=True)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        img, target, valid = batch
        loss = self._loss(self(img), target, valid)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        img, target, valid = batch
        logits = self(img)
        self.log("validation/loss", self._loss(logits, target, valid), on_epoch=True, prog_bar=True)
        self._step_metrics(logits, target, valid)

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.trainer.estimated_stepping_batches, 1)
        )
        return {"optimizer": opt, "lr_scheduler": sched}
