from typing import TypedDict

from torch import Tensor


class Metadata(TypedDict):
    slide_name: str


class MetadataBatch(TypedDict):
    slide_name: list[str]


type Sample = tuple[Tensor, Tensor, Metadata]
type PredictSample = tuple[Tensor, Metadata]


class MetadataTiles(Metadata):
    x: int
    y: int


class MetadataTilesBatch(MetadataBatch):
    x: Tensor
    y: Tensor


type TilesSample = tuple[Tensor, Tensor, MetadataTiles]
type TilesPredictSample = tuple[Tensor, MetadataTiles]

type TilesInput = tuple[Tensor, Tensor, MetadataTilesBatch]
type TilesPredictInput = tuple[Tensor, MetadataTilesBatch]
