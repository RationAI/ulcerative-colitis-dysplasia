#!/usr/bin/env python
"""Build an xOpat viewer link: one hunCRC slide + one finished epithelium mask.

No ReportFast report -- just the redirect.php session the viewer uses, with a
single heatmap layer on top of the slide. The epi-200 job only uploads masks to
MLflow when ALL 200 are done, so this uploads one finished mask itself (same
public run make_dysplasia_masks.py publishes to, so the /public_mlflow DataID
prefix holds).

The raw epi BigTIFF is accepted by the tile server as-is: the proven tissue
masks (verified in the viewer) have the identical layout -- bigtiff, 512x512
deflate tiles, no SubfileType, XResolution px/cm. The epi top level is slide
level 2 (147200x136384 at 20615.6 px/cm = 10000/0.485068 um/px), so no
conversion is needed.

Usage (from the notebook pod):
    .venv/bin/python scripts/make_epi_viewer_link.py 169
"""
import json
import sys
import urllib.parse

from mlflow.tracking import MlflowClient

EXPERIMENT_ID = 168
#: The report/publish run -- the artifact host make_dysplasia_masks.py already
#: uses, so DataIDs under it resolve in the viewer.
PUBLISH_RUN = "45a957c5a79d4560b2b3efb68e19eed6"
BASE_URL = "https://xopat.rationai.cloud.trusted.e-infra.cz/xopat/redirect.php"
SLIDE_ROOT = "data/Public/Public/colorectum/TCIA_HunCRC/PKG - Hungarian-Colorectal-Screening_v2"
MASK_DIR = "/mnt/projects/huncrc_dysplasia/epithelium_masks/raw"
COLOR = "#00cc44"


def main(slide: str) -> None:
    src = f"{MASK_DIR}/{slide}.tiff"
    progress = json.load(open(f"{src}.progress.json"))
    if progress["status"] != "done" or progress["strips_done"] != progress["strips_total"]:
        raise SystemExit(f"{slide}: mask not done ({progress['status']}, "
                         f"{progress['strips_done']}/{progress['strips_total']} strips)")

    key = open("/home/jovyan/.mlflow_key").read().strip()
    client = MlflowClient(
        tracking_uri="https://mlflow.rationai.cloud.e-infra.cz",
        username="519192@mail.muni.cz",
        password=key,
    )
    client.get_run(PUBLISH_RUN)  # raises if the run is unreachable
    print(f"uploading {src} (~{src.stat().st_size / 1e6:.0f} MB) to run {PUBLISH_RUN} ...",
          flush=True)
    client.log_artifact(PUBLISH_RUN, src, artifact_path=f"epithelium_masks/{slide}.tiff")
    print("uploaded")

    slide_id = f"{SLIDE_ROOT}/{slide}.mrxs"
    mask_id = (f"/public_mlflow/{EXPERIMENT_ID}/{PUBLISH_RUN}"
               f"/artifacts/epithelium_masks/{slide}.tiff")
    doc = {
        "params": {},
        "data": [slide_id, mask_id],
        "background": [{
            "dataReference": 0,
            "lossless": False,
            "protocol": f"`/wsi-service/v3/files/info?paths={slide_id}`",
        }],
        "visualizations": [{
            "name": f"{slide} episeg",
            "lossless": True,
            "shaders": {
                "layer_shader_0": {
                    "name": "Epithelium",
                    "type": "heatmap",
                    "dataReferences": [1],
                    "fixed": False,
                    "color": COLOR,
                    "visible": True,
                    "params": {"use_channel0": "r", "color": COLOR, "opacity": 0.6},
                }
            },
        }],
    }
    link = BASE_URL + "#" + urllib.parse.quote(
        json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    print("LINK:")
    print(link)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "169")
