import json
import ray
import pandas as pd
import pyvips
import numpy as np
from pathlib import Path

import hydra
from omegaconf import DictConfig
from openslide import OpenSlide
from PIL import Image, ImageDraw

from rationai.masks import slide_resolution, write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


class JSONPolygonMask:
    """Parses JSON annotations and renders them into a multi-class mask array."""

    def __init__(
        self,
        json_path: Path,
        mask_size: tuple[int, int],
        scale_factor: float,
        target_groups: dict[str, int],
    ):
        self.json_path = json_path
        self.mask_size = mask_size  # (width, height)
        self.scale = scale_factor
        self.target_groups = target_groups

    def __call__(self) -> np.ndarray:
        mask_img = Image.new("L", self.mask_size, 0)
        draw = ImageDraw.Draw(mask_img)

        if not self.json_path.exists():
            return np.array(mask_img)

        with open(self.json_path, "r") as f:
            data = json.load(f)

        for item in data.get("items", []):
            class_name = item.get("name")

            if class_name in self.target_groups:
                coords = item.get("coordinates", [])
                if not coords:
                    continue

                fill_value = self.target_groups[class_name]
                scaled_coords = [(c[0] * self.scale, c[1] * self.scale) for c in coords]
                draw.polygon(scaled_coords, fill=fill_value)

        return np.array(mask_img)


@ray.remote
def process_slide(
    slide_path: str,
    annot_dir: str,
    level: int,
    output_base: Path,
    target_groups: list[str],
) -> str:
    slide_p = Path(slide_path)
    json_path = Path(annot_dir) / f"{slide_p.stem}.json"

    if not json_path.exists():
        return f"Skipped: No JSON for {slide_p.name}"

    with OpenSlide(slide_path) as slide:
        mpp_x, mpp_y = slide_resolution(slide, level=level)
        full_dim = slide.level_dimensions[0]
        mask_dim = slide.level_dimensions[level]
        scale = mask_dim[0] / full_dim[0]

    parser = JSONPolygonMask(json_path, mask_dim, scale, target_groups)
    mask_array = parser()

    if mask_array.max() == 0:
        print(f"Empty mask for {slide_p.name}")

    vips_mask = pyvips.Image.new_from_array(mask_array)

    save_path = output_base / f"{slide_p.stem}.tiff"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    write_big_tiff(vips_mask, str(save_path), mpp_x=mpp_x, mpp_y=mpp_y)


@with_cli_args([])
@hydra.main(config_path="conf", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = pd.read_csv(config.dataset.local_path)

    output_path = Path(config.annotation_mask.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    process_items(
        df["slide_path"].tolist(),
        process_item=process_slide,
        fn_kwargs={
            "annot_dir": config.annotation_mask.dir,
            "level": config.annotation_mask.level,
            "output_base": output_path,
            "target_groups": config.annotation_mask.target_groups,
        },
        max_concurrent=config.annotation_mask.max_concurrent,
    )

    # if logger:
    #     logger.log_artifacts(
    #         local_dir=str(output_path), artifact_path="annotation_masks"
    #     )


if __name__ == "__main__":
    ray.init()
    main()
