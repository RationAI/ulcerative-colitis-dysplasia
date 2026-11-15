from collections.abc import Iterable
from typing import Any, cast

import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset

from ml.data.datasets.labels import LabelMode, get_label, process_slides
from ml.data.datasets.utils import filter_negative_origin, filter_tiles
from ml.typing import Metadata, PredictSample, Sample


class _Embeddings[T: Sample | PredictSample](Dataset[T]):
    def __init__(
        self,
        slide_metadata: dict[str, Any],
        tiles: HFDataset,
        mode: LabelMode | str | None = None,
        include_labels: bool = True,
    ) -> None:
        self.slide_metadata = slide_metadata
        self.tiles = tiles
        self.mode = LabelMode(mode) if mode is not None else None
        self.include_labels = include_labels

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified if labels are included.")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> T:
        tile = self.tiles[idx]
        metadata = Metadata(
            slide_name=self.slide_metadata["name"],
            x=tile["x"],
            y=tile["y"],
        )

        embedding = torch.tensor(tile["embedding"])
        if not self.include_labels:
            return cast("T", (embedding, metadata))

        assert self.mode is not None, "Mode must be specified for labels."
        label = get_label(tile, self.mode)
        return cast("T", (embedding, label, metadata))


class Embeddings(MetaTiledSlides[Sample]):
    def _filter_tiles(self, tiles: HFDataset) -> HFDataset:
        return filter_negative_origin(
            filter_tiles(tiles, self.thresholds, self.min_thresholds),
            self.negative_slides,
        )

    @property
    def labels(self) -> torch.Tensor:
        return torch.stack(
            [
                get_label(cast("dict[str, Any]", tile), self.mode)
                for slide in self.slides
                for tile in self._filter_tiles(
                    self.filter_tiles_by_slide(dict(slide)["id"])
                )
            ]
        )

    def __init__(
        self,
        uris: Iterable[str] | str,
        mode: LabelMode | str,
        thresholds: dict[str, float] | None = None,
        min_thresholds: dict[str, float] | None = None,
        negative_slides: bool = False,
        val_fold: int | None = None,
        is_val: bool = False,
    ) -> None:
        self.mode = LabelMode(mode)
        self.thresholds = thresholds or {}
        self.min_thresholds = min_thresholds or {}
        self.negative_slides = negative_slides
        self.val_fold = val_fold
        self.is_val = is_val
        super().__init__(uris=(uris,) if isinstance(uris, str) else uris)

    def generate_datasets(self) -> Iterable[_Embeddings[Sample]]:
        self.slides = process_slides(self.slides, self.val_fold, self.is_val)
        return (
            _Embeddings(
                slide_metadata=dict(slide),
                tiles=self._filter_tiles(self.filter_tiles_by_slide(dict(slide)["id"])),
                mode=self.mode,
                include_labels=True,
            )
            for slide in self.slides
        )


class EmbeddingsPredict(MetaTiledSlides[PredictSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        thresholds: dict[str, float] | None = None,
        min_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.thresholds = thresholds or {}
        self.min_thresholds = min_thresholds or {}
        super().__init__(uris=(uris,) if isinstance(uris, str) else uris)

    def generate_datasets(self) -> Iterable[_Embeddings[PredictSample]]:
        self.slides = process_slides(self.slides)
        return (
            _Embeddings(
                slide_metadata=dict(slide),
                tiles=filter_tiles(
                    self.filter_tiles_by_slide(dict(slide)["id"]),
                    self.thresholds,
                    self.min_thresholds,
                ),
                include_labels=False,
            )
            for slide in self.slides
        )
