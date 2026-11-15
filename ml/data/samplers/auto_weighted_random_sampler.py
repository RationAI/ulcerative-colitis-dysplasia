from collections import Counter
from typing import Protocol

import torch
from torch.utils.data import WeightedRandomSampler


class LabeledDataset(Protocol):
    def __len__(self) -> int: ...
    @property
    def labels(self) -> torch.Tensor: ...


class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, dataset: LabeledDataset, replacement: bool = True) -> None:
        labels = dataset.labels  # Tensor (N,) or (N, num_labels)

        if labels.ndim == 1:
            list_labels = labels.tolist()
            counts = Counter(list_labels)
            weights = [1.0 / counts[label] for label in list_labels]
        else:
            weights = self._multilabel_weights(labels)

        super().__init__(weights, len(dataset), replacement)

    @staticmethod
    def _multilabel_weights(labels: torch.Tensor) -> list[float]:
        normal_mask = (labels == 0).all(dim=1)  # (N,)
        lgd_mask = labels[:, 0] == 1  # (N,)
        hgd_mask = labels[:, 1] == 1  # (N,)

        weights = torch.zeros(len(labels))
        weights[normal_mask] += 1.0 / normal_mask.sum().item()
        weights[lgd_mask] += 1.0 / lgd_mask.sum().item()
        weights[hgd_mask] += 1.0 / hgd_mask.sum().item()

        return weights.tolist()
