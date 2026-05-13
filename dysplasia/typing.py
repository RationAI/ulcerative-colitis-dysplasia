from pathlib import Path
from typing import TypedDict

import pandas as pd
from torch import Tensor


class Metadata(TypedDict):
    slide_id: str


class MetadataTiles(Metadata):
    slide_id: str
    x: int
    y: int


type TilesSample = tuple[Tensor, Tensor, MetadataTiles]
type TilesPredictSample = tuple[Tensor, MetadataTiles]


class MetadataTileEmbeddings(Metadata):
    slide_id: str
    slide_name: str
    slide_path: Path
    level: int
    tile_extent_x: int
    tile_extent_y: int
    tiles: pd.DataFrame
    x: Tensor  # Tensor[int]
    y: Tensor  # Tensor[int]


type TileEmbeddingsSample = tuple[Tensor, Tensor, MetadataTileEmbeddings]
type TileEmbeddingsPredictSample = tuple[Tensor, MetadataTileEmbeddings]

type TileEmbeddingsInput = tuple[Tensor, Tensor, list[MetadataTileEmbeddings]]
type TileEmbeddingsPredictInput = tuple[Tensor, list[MetadataTileEmbeddings]]


class MetadataSlideEmbeddings(Metadata):
    slide_id: str
    slide_name: str
    slide_path: Path


type SlideEmbeddingsSample = tuple[Tensor, Tensor, MetadataSlideEmbeddings]
type SlideEmbeddingsPredictSample = tuple[Tensor, MetadataSlideEmbeddings]

type SlideEmbeddingsInput = tuple[Tensor, Tensor, list[MetadataSlideEmbeddings]]
type SlideEmbeddingsPredictInput = tuple[Tensor, list[MetadataSlideEmbeddings]]

type Output = Tensor
