from typing import TypedDict

from torch import Tensor


class Metadata(TypedDict):
    slide_name: str
    x: int
    y: int


class MetadataBatch(TypedDict):
    slide_name: list[str]
    x: Tensor
    y: Tensor


type Sample = tuple[Tensor, Tensor, Metadata]
type PredictSample = tuple[Tensor, Metadata]

type Input = tuple[Tensor, Tensor, MetadataBatch]
type PredictInput = tuple[Tensor, MetadataBatch]
