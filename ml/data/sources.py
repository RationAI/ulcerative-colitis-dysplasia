"""Resolving a dataset config's embedding source.

``mlkit``'s ``SlidesTilesLoader`` reads slides/tiles parquets either from
MLflow artifact URIs or from local directories. The IKEM embeddings live in
MLflow (``dataset.mlflow_uris``); the HunCRC ones live on the mounted share
(``dataset.paths``), so the stages that build a dataset by hand have to take
whichever the active dataset config provides.
"""

from typing import Any


def embedding_source(dataset: Any, split: str) -> dict[str, list[str]]:
    """Return the ``Embeddings`` source kwargs for one split.

    Args:
        dataset: The resolved ``dataset`` config node.
        split: ``train``, ``test_preliminary`` or ``test_final``.

    Returns:
        ``{"paths": [...]}`` when the dataset config declares
        ``paths.embeddings``, otherwise ``{"uris": [...]}`` from
        ``mlflow_uris.embeddings``.
    """
    paths = getattr(dataset, "paths", None)
    if paths is not None and "embeddings" in paths:
        return {"paths": [str(paths["embeddings"][split])]}

    return {"uris": [str(dataset.mlflow_uris.embeddings[split])]}
