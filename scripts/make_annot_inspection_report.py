#!/usr/bin/env python
"""7-slide hunCRC annotation inspection report (all nine classes).

Companion to scripts/restamp_annot_tiles.py, which re-derived per-tile
coverage for all nine annotation classes on the 7 inspect slides (tiling_200
carries only the two dysplasia classes) and rasterized per-class level-3
masks to preprocessing/tile_masks_allclasses/{class}/{stem}.tiff. This script
uploads those 63 masks into the report run and builds the xOpat report in the
shape of make_mlp_prediction_report.py -- the proven byte shape for this
deployment (v2 redirect.php fallback + v3 with wsi_service background /
iipimage mask protocols; reportfast-feedback.md sessions 2-3).

Layers per slide (data-pool order after the .mrxs background): the annotation
classes present on the slide (tiles with coverage > 0), specific classes
first and `normal` last. The seven non-dysplasia classes are visible by
default; the two dysplasia classes are hidden by default (reference layers,
re-derived from the current annotations -- the tiling_200 columns are stale
and not used here).

Upload is ~500 MB (63 tiffs, ~8 MB each) and is fine on the pod; --report and
--publish are light (artifact listing + two small html uploads).

Usage:
    python scripts/make_annot_inspection_report.py --upload
    python scripts/make_annot_inspection_report.py --report OUT.html --publish
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
T7 = PROJECT / "preprocessing" / "tiling_7annot"
MASKS = PROJECT / "preprocessing" / "tile_masks_allclasses"

EXPERIMENT_ID = 168
#: Same artifact host as the 186 tile-masks report and the MLP prediction
#: report (their layers already live there).
RUN = "45a957c5a79d4560b2b3efb68e19eed6"
VIEWER = "https://xopat.rationai.cloud.trusted.e-infra.cz"
V2 = f"{VIEWER}/xopat/redirect.php"
V3 = f"{VIEWER}/v3/"
MLFLOW_URL = "https://mlflow.rationai.cloud.e-infra.cz"
MLFLOW_USER = "519192@mail.muni.cz"

#: class -> (color, visible by default). Layer/stack order: specific classes
#: first, the broad `normal` reference last, the two dysplasia classes (off
#: by default) at the end.
CLASSES = {
    "adenocarcinoma": ("#ff8800", True),
    "suspicious_for_invasion": ("#ff55cc", True),
    "tumor_necrosis": ("#aa4422", True),
    "inflammation": ("#00ccff", True),
    "resection_edge": ("#00cc44", True),
    "artifact": ("#9955ff", True),
    "normal": ("#4488ff", True),
    "lowgrade_dysplasia": ("#ffcc00", False),
    "highgrade_dysplasia": ("#ff2200", False),
}

#: Cohort category (hunCRC diagnosis) for the card label -- the 7-slide
#: cohort is hand-picked, so the labels are hand-picked with it.
CATEGORY = {
    "090": "adenoma",
    "106": "CRC",
    "112": "adenoma",
    "121": "CRC",
    "124": "negative",
    "133": "adenoma",
    "139": "CRC",
}


def _slides() -> pd.DataFrame:
    s = pd.read_parquet(T7 / "slides" / "slides.parquet")
    s["stem"] = s["path"].map(lambda p: Path(p).stem)
    return s.sort_values("stem").reset_index(drop=True)


def tile_counts(stem: str) -> dict[str, int]:
    """Tiles with coverage > 0 per class, from the restamped table."""
    t = pd.read_parquet(
        T7 / "tiles" / f"{stem}.parquet",
        columns=[f"annot_{c}" for c in CLASSES],
    )
    return {c: int((t[f"annot_{c}"] > 0).sum()) for c in CLASSES}


def _client():
    from mlflow.tracking import MlflowClient

    #: 3.x clients take no credentials in the constructor -- auth rides on the
    #: environment. Pod mode: ~/.mlflow_key (same as make_mlp_prediction_report).
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        key = Path.home().joinpath(".mlflow_key").read_text().strip()
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = key
    return MlflowClient(tracking_uri=MLFLOW_URL)


def _remote(artifact_dir: str, stem: str) -> str:
    return f"/public_mlflow/{EXPERIMENT_ID}/{RUN}/artifacts/{artifact_dir}/{stem}.tiff"


def upload() -> None:
    client = _client()
    client.get_run(RUN)  #: fail fast on auth
    stems = set(_slides()["stem"])
    total = sum(
        len([p for p in (MASKS / cls).glob("*.tiff") if p.stem in stems])
        for cls in CLASSES
    )
    done = 0
    for cls in CLASSES:
        artifact_dir = f"tile_masks_allclasses/{cls}"
        try:
            have = {
                i.path.rsplit("/", 1)[-1]
                for i in client.list_artifacts(RUN, path=artifact_dir)
            }
        except Exception:
            have = set()
        for src in sorted((MASKS / cls).glob("*.tiff")):
            if src.stem not in stems:
                continue
            done += 1
            if src.name in have:
                continue
            client.log_artifact(RUN, str(src), artifact_path=artifact_dir)
            have.add(src.name)
            print(f"[{done}/{total}] {artifact_dir}/{src.name}", flush=True)
    print(f"upload done: {done}/{total} files accounted for in the run")


def _bg_data_id(path: str) -> str:
    """mrxs path on the DATA share -> DataID (path relative to /mnt, raw
    spaces -- the proven rule from make_epi_viewer_link.py)."""
    p = str(path)
    assert p.startswith("/mnt/"), p
    return p[len("/mnt/"):]


def session_doc(stem: str, bg_id: str, layers: list, shape: str) -> dict:
    """Same byte shape as make_mlp_prediction_report.session_doc (proven on
    this deployment): v2 = redirect.php with the inline protocol patched back
    in; v3 = registered per-data-entry protocols (wsi_service background,
    iipimage masks) and the required `order` on the visualization.
    `layers` is a list of (class, color, visible) tuples."""
    mask_ids = [_remote(f"tile_masks_allclasses/{cls}", stem) for cls, _, _ in layers]
    shaders = {
        f"layer{i}": {
            "name": name,
            "type": "heatmap",
            "dataReferences": [i + 1],
            "fixed": False,
            "color": color,
            "visible": 1 if vis else 0,
            "params": {"use_channel0": "r", "color": color, "opacity": 0.6},
        }
        for i, (name, color, vis) in enumerate(layers)
    }
    if shape == "v2":
        return {
            "params": {"sessionName": stem},
            "data": [bg_id, *mask_ids],
            "background": [{
                "dataReference": 0,
                "lossless": False,
                "protocol": f"`/wsi-service/v3/files/info?paths={bg_id}`",
            }],
            "visualizations": [{
                "name": f"{stem} annotations",
                "lossless": True,
                "shaders": {
                    f"layer_shader_{i}": {
                        "name": name,
                        "type": "heatmap",
                        "dataReferences": [i + 1],
                        "fixed": False,
                        "color": color,
                        "visible": vis,
                        "params": {"use_channel0": "r", "color": color,
                                   "opacity": 0.6},
                    }
                    for i, (name, color, vis) in enumerate(layers)
                },
            }],
        }
    return {
        "params": {"sessionName": stem, "bypassCookies": True, "theme": "auto",
                   "activeBackgroundIndex": [0], "bypassCacheLoadTime": True},
        "data": [
            {"dataID": bg_id, "protocol": "wsi_service"},
            *[{"dataID": m, "protocol": "iipimage"} for m in mask_ids],
        ],
        "background": [{
            "id": f"bg{stem}",
            "name": stem,
            "dataReference": 0,
            "visualizationIndex": 0,
            "lossless": False,
        }],
        "visualizations": [{
            "name": f"{stem} annotations",
            "order": list(shaders),
            "shaders": shaders,
        }],
    }


def build_report(out: Path) -> list[Path]:
    """Writes OUT.v3.html (primary) and OUT.v2.html (proven fallback)."""
    from report_fast import Report, SlideGrid, XopatEndpoint, XopatSession

    client = _client()
    uploaded = {}
    for cls in CLASSES:
        artifact_dir = f"tile_masks_allclasses/{cls}"
        try:
            infos = client.list_artifacts(RUN, path=artifact_dir)
        except Exception:
            infos = []
        uploaded[artifact_dir] = {i.path.rsplit("/", 1)[-1] for i in infos}

    slides = _slides()
    counts = {row.stem: tile_counts(row.stem) for row in slides.itertuples()}

    #: single-dot filenames: the mlflow-artifacts proxy 500s on "name.v3.html"
    v3 = out.with_name(f"{out.stem}_v3.html")
    v2 = out.with_name(f"{out.stem}_v2.html")
    for shape, page in (("v3", v3), ("v2", v2)):
        base = V2 if shape == "v2" else V3
        endpoint = XopatEndpoint(base_url=base)
        grid = SlideGrid(title="Annotation classes", endpoint=endpoint)
        for _, row in slides.iterrows():
            stem = row["stem"]
            c = counts[stem]
            #: a class appears on a card only when the slide has tiles of it
            #: -- the layer list itself shows which annotations the slide has
            layers = [
                (cls, color, vis)
                for cls, (color, vis) in CLASSES.items()
                if c[cls] and f"{stem}.tiff" in uploaded[f"tile_masks_allclasses/{cls}"]
            ]
            doc = session_doc(stem, _bg_data_id(row["path"]), layers, shape)
            if shape == "v2":
                proto = doc["background"][0].pop("protocol")
                s = XopatSession.from_config(json.dumps(doc))
                s.background[0]["protocol"] = proto
            else:
                s = XopatSession.from_config(json.dumps(doc))
            present = sorted((cl for cl in CLASSES if c[cl]), key=lambda cl: -c[cl])
            note = ", ".join(f"{cl} {c[cl]:,}" for cl in present)
            grid.add(
                s,
                name=f"{stem} ({CATEGORY.get(stem, '?')})",
                note=f"{len(layers)} layers -- {note}",
            )
        report = Report(
            title="HunCRC -- non-dysplasia annotation classes (7 slides)",
            subtitle=(
                "7 slides; each card shows only the annotation classes the "
                "slide actually has (a layer is present iff the slide has "
                "tiles of that class). Layers re-derived from the current "
                "EMPIA JSONs (preprocessing/annotations), level-3 masks, "
                "2x2-tile mean. Non-dysplasia classes on by default; LGD/HGD "
                "off (reference)."
            ),
            preamble=(
                "One card per slide. Only the classes with tiles on that "
                "slide appear as layers, so the layer list itself tells you "
                "what the slide is annotated with. The non-dysplasia classes "
                "are visible by default (orange = adenocarcinoma, pink = "
                "suspicious_for_invasion, brown = tumor_necrosis, cyan = "
                "inflammation, green = resection_edge, purple = artifact, "
                "blue = normal); the two dysplasia classes (yellow = LGD, red "
                "= HGD) are reference layers, hidden by default. The card "
                "note lists the per-class tile counts. All layers were "
                "re-stamped from preprocessing/annotations -- the current "
                "ground truth, not the tiling_200 columns. Mask geometry "
                "matches preprocessing/tile_masks (level 3, tile 224 -> 112 "
                "px, stride 112 -> 56 px). Links unverified from the build "
                "pod -- report blank cards."
            ),
        )
        report.add(grid)
        report.write(str(page))
        if shape == "v2":
            #: ReportFast 0.1.0 joins base with '/#'; redirect.php white-screens
            #: on it (the /v3/ mount wants '/#', so v3 needs no fix).
            html = page.read_text()
            fixed = html.replace(f"{V2}/#", f"{V2}#")
            if fixed != html:
                page.write_text(fixed)
        print(f"report -> {page}")
    return [v3, v2]


def publish(html_path: Path, artifact_dir: str = "report") -> str:
    client = _client()
    client.log_artifact(RUN, str(html_path), artifact_path=artifact_dir)
    return (
        f"{MLFLOW_URL}/api/2.0/mlflow-artifacts/artifacts/"
        f"{EXPERIMENT_ID}/{RUN}/artifacts/{artifact_dir}/{html_path.name}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upload", action="store_true",
                    help="upload the 63 class masks into the run (resumable)")
    ap.add_argument("--report", type=Path, metavar="OUT.html",
                    help="build the 7-slide report (v3 + v2 pages)")
    ap.add_argument("--publish", action="store_true",
                    help="log the built pages to the run and print the URLs")
    args = ap.parse_args()

    if args.upload:
        upload()
    pages = []
    if args.report:
        pages = build_report(args.report)
    if pages and args.publish:
        for p in pages:
            print("PUBLISHED:", publish(p))
