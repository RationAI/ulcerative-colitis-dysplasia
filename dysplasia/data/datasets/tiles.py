from collections.abc import Iterable
from typing import Generic, TypeVar

import pandas as pd
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import Dataset

from dysplasia.data.datasets.labels import process_slides
from dysplasia.typing import MetadataTiles, TilesPredictSample, TilesSample


T = TypeVar("T", bound=TilesSample | TilesPredictSample)


class _Tiles(Dataset[T], Generic[T]):
    def __init__(
        self,
        slide_metadata: pd.Series,
        tiles: HFDataset,
        include_labels: bool = True,
        transforms: TransformType | None = None,
    ) -> None:
        super().__init__()
        self.slide_tiles = OpenSlideTilesDataset(
            slide_path=slide_metadata["path"],
            level=slide_metadata["level"],
            tile_extent_x=slide_metadata["tile_extent_x"],
            tile_extent_y=slide_metadata["tile_extent_y"],
            tiles=tiles,
        )
        self.slide_metadata = slide_metadata
        self.include_labels = include_labels
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

        if self.include_labels and self.mode is None:
            raise ValueError("Mode must be specified if labels are included.")

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> TilesSample | TilesPredictSample:
        image = self.slide_tiles[idx]
        tile = self.slide_tiles.tiles[idx]
        metadata = MetadataTiles(
            slide_id=self.slide_tiles.slide_path.stem,
            x=tile["x"],
            y=tile["y"],
        )

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        image = self.to_tensor(image=image)["image"]
        return image, metadata


class TilesPredict(MetaTiledSlides[TilesPredictSample]):
    def __init__(
        self,
        uris: Iterable[str] | str,
        transforms: TransformType | None = None,
    ) -> None:
        self.transforms = transforms
        super().__init__(uris=(uris,) if isinstance(uris, str) else uris)

    def generate_datasets(self) -> Iterable[_Tiles[TilesPredictSample]]:
        self.slides = process_slides(self.slides)
        return (
            _Tiles(
                slide_metadata=slide,
                tiles=self.filter_tiles_by_slide(slide["id"]),
                include_labels=False,
                transforms=self.transforms,
            )
            for _, slide in self.slides.iterrows()
        )
