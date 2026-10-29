#!/usr/bin/env python
"""MLP tile-prediction masks + xOpat report for the hunCRC val set.

Predicts per-tile dysplasia probability with the MLP from this morning's run
(training/embeddings_huncrc/last-v2.ckpt, val AUC on the test_preliminary
split), rasterizes each slide's tile field into a level-3 big-TIFF (same
canvas/geometry as the tile_masks tiffs: 2x coarser than the tiling level,
tile extent 224 -> 112 px, stride 112 -> 56 px, 2x2-tile overlap averaged),
and builds the xOpat report in the shape of make_tile_masks_report.py -- the
proven byte-shape for this deployment (v2 redirect.php fallback + v3 with
wsi_service/iipimage per-data-entry protocols).

Layers per slide (in data-pool order after the .mrxs background):
  * Dysplasia prediction (MLP) -- the new mask, magenta, visible
  * LGD / HGD annotation       -- the original pathologist masks, annotated
                                  slides only (all-zero canvases elsewhere)
  * Epithelium                 -- raw epi mask shrunk 2x to the level-3 canvas
  * Tile outlines              -- the existing tile_masks/outlines grid

Mask TIFFs go to preprocessing/mlp_predictions/ (a NEW dir; nothing existing
is touched) and are uploaded into the report run alongside the previous
report's artifacts. The annotation-mask and outline layers are already in the
run from earlier reports, so only the prediction + epi layers need uploading.

Memory note: the embedding parquets hold list<double> columns; the script
reads the dense values buffer directly (no Python-float materialization) --
the naive pd.read_parquet path OOM'd the 32 GB pod on slide 126.

Usage (as one cluster job, or step by step):
    python scripts/make_mlp_prediction_report.py --eval      # checkpoint val AUC
    python scripts/make_mlp_prediction_report.py --predict   # per-tile masks
    python scripts/make_mlp_prediction_report.py --epi       # level-3 epi masks
    python scripts/make_mlp_prediction_report.py --upload    # masks -> the run
    python scripts/make_mlp_prediction_report.py --report OUT.html --publish

The heavy steps (eval/predict/epi/upload) belong in a kube-job: the notebook
pod is 32 GB and the per-slide 1.5 GB embedding reads OOM'd it when run
interactively. --report alone is light and can run on the pod.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
#: sys.path[0] is scripts/, where `preprocessing/` holds the JOB SUBMITTERS --
#: the repo root (the real `preprocessing` package + `ml`) must win.
sys.path.insert(0, str(REPO_ROOT))

from ratiopath.masks import write_big_tiff
from ratiopath.openslide import OpenSlide

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
TILING = PROJECT / "preprocessing" / "tiling_200"
EMB = PROJECT / "preprocessing" / "embeddings" / "prov-gigapath"
CKPT = PROJECT / "training" / "embeddings_huncrc" / "last-v2.ckpt"

PRED_OUT = PROJECT / "preprocessing" / "mlp_predictions"
EPIS_RAW = PROJECT / "preprocessing" / "epithelium_masks" / "raw"
EPIS_L3 = PROJECT / "preprocessing" / "epithelium_masks" / "level3"
ANNOT = PROJECT / "preprocessing" / "annotation_masks"
OUTLINES = PROJECT / "preprocessing" / "tile_masks" / "outlines"

EXPERIMENT_ID = 168
#: Same artifact host as the 186 tile-masks report (its annotation + outline
#: layers already live there).
RUN = "45a957c5a79d4560b2b3efb68e19eed6"
VIEWER = "https://xopat.rationai.cloud.trusted.e-infra.cz"
V2 = f"{VIEWER}/xopat/redirect.php"
V3 = f"{VIEWER}/v3/"
MLFLOW_URL = "https://mlflow.rationai.cloud.e-infra.cz"
MLFLOW_USER = "519192@mail.muni.cz"

#: The val set of the morning MLP run (configs/dataset/embeddings_huncrc.yaml:
#: val = split test_preliminary).
SPLIT = "test_preliminary"
#: Tile grid at the tiling level -> level-3 mask canvas (2x coarser).
TILE_EXTENT = 224
STRIDE = 112
MASK_EXTENT = TILE_EXTENT // 2
MASK_STRIDE = STRIDE // 2


def val_slides() -> pd.DataFrame:
    """The val-set slides that have an embedding parquet, with their
    dysplasia-annotation tile count (0 = unannotated slide)."""
    slides = pd.read_parquet(TILING / "slides" / "slides.parquet")
    slides["stem"] = slides["path"].map(lambda p: Path(p).stem)
    slides = slides[slides["split"] == SPLIT].copy()
    slides = slides[slides["stem"].map(lambda s: (EMB / f"{s}.parquet").exists())]

    tiles = pd.concat(
        [
            pd.read_parquet(
                f, columns=["slide_id", "lowgrade_dysplasia", "highgrade_dysplasia"]
            )
            for f in sorted((TILING / "tiles").glob("*.parquet"))
        ]
    )
    ann = (
        (tiles["lowgrade_dysplasia"] > 0) | (tiles["highgrade_dysplasia"] > 0)
    ).groupby(tiles["slide_id"]).sum()
    slides["ann_tiles"] = slides["id"].map(ann).fillna(0).astype(int)
    return slides.sort_values("stem").reset_index(drop=True)


def emb_matrix(tab: pa.Table) -> np.ndarray:
    """The list<double> `embedding` column as (n, 1536) float32 -- the values
    buffer is dense (asserted), so this never materializes Python floats."""
    col = tab["embedding"].combine_chunks()
    off = np.frombuffer(col.buffers()[1], dtype=np.int32)
    n = len(col)
    #: row width from the offsets (all rows equal, asserted); the values are
    #: the child array (ListArray.buffers() holds only validity + offsets)
    d = np.diff(off)
    w = int(d[0])
    assert (d == w).all(), "embedding list rows are not equal length"
    vals = col.values.to_numpy(zero_copy_only=False)
    assert len(vals) == w * n, "values buffer not dense for the offsets"
    return vals.reshape(n, w).astype(np.float32, copy=False)


def predict_slide(stem: str, mrxs_path: str, model) -> None:
    """One slide: per-tile probability -> 2x2-tile-averaged level-3 mask."""
    tab = pq.read_table(EMB / f"{stem}.parquet", columns=["x", "y", "embedding"])
    ex = tab["x"].to_numpy().astype(np.int64)
    ey = tab["y"].to_numpy().astype(np.int64)
    mat = emb_matrix(tab)
    del tab

    import torch

    X = torch.from_numpy(np.ascontiguousarray(mat))
    with torch.no_grad():
        probs = model(X).sigmoid().numpy().ravel()
    del mat, X

    with OpenSlide(mrxs_path) as ws:
        w3, h3 = ws.level_dimensions[3]
        mpp_x, mpp_y = ws.slide_resolution(3)

    #: Tile grid -> lattice of MASK_STRIDE cells; a mask pixel is covered by
    #: up to 4 tiles (50% overlap), so each cell takes the 2x2-tile mean.
    kx = np.round(ex / STRIDE).astype(np.int64)
    ky = np.round(ey / STRIDE).astype(np.int64)
    nrow, ncol = h3 // MASK_STRIDE + 1, w3 // MASK_STRIDE + 1
    Pn = np.full((nrow, ncol), np.nan)
    Pn[ky, kx] = probs

    present = ~np.isnan(Pn)
    Pf = np.where(present, Pn, 0.0)
    Sf = np.pad(Pf, ((1, 0), (1, 0)))
    Sn = np.pad(present.astype(np.float64), ((1, 0), (1, 0)))
    num = Sf[:-1, :-1] + Sf[1:, :-1] + Sf[:-1, 1:] + Sf[1:, 1:]
    den = Sn[:-1, :-1] + Sn[1:, :-1] + Sn[:-1, 1:] + Sn[1:, 1:]
    A = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    import pyvips

    #: nearest-neighbor 56x upscale + crop streams through pyvips (no full
    #: uint8 canvas in memory -- that would be 5 GB per slide on this pod).
    im = pyvips.Image.new_from_array((A * 255).astype(np.uint8))
    im = im.resize(MASK_STRIDE, kernel="nearest")
    im = im.crop(0, 0, w3, h3)
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    write_big_tiff(im, PRED_OUT / f"{stem}.tiff", mpp_x, mpp_y)

    frac_hot = float((probs > 0.5).mean())
    print(
        f"{stem}: {len(probs)} tiles, mean prob {probs.mean():.3f}, "
        f">0.5 {frac_hot:.1%}",
        flush=True,
    )


def predict_all() -> None:
    from ml.modeling.embeddings import DysplasiaEmbeddingsClassifier

    model = DysplasiaEmbeddingsClassifier.load_from_checkpoint(
        CKPT, map_location="cpu"
    ).eval()
    df = val_slides()
    for _, row in df.iterrows():
        predict_slide(row["stem"], row["path"], model)
    print(f"\ndone: {len(df)} prediction masks in {PRED_OUT}")


def eval_val() -> None:
    """The checkpoint's AUC/acc on the same filtered val set the morning run
    monitored (test_preliminary, cluster_k=2.0 serial-sections rule) -- the
    number the report quotes."""
    import torch
    from sklearn.metrics import roc_auc_score

    from ml.data.embeddings_dataset import _annotated_piece_mask
    from ml.modeling.embeddings import DysplasiaEmbeddingsClassifier

    model = DysplasiaEmbeddingsClassifier.load_from_checkpoint(
        CKPT, map_location="cpu"
    ).eval()
    slides = pd.read_parquet(TILING / "slides" / "slides.parquet")
    slides["stem"] = slides["path"].map(lambda p: Path(p).stem)
    slides = slides[slides["split"] == SPLIT]
    slides = slides[slides["stem"].map(lambda s: (EMB / f"{s}.parquet").exists())]
    stem_of = dict(zip(slides["id"], slides["stem"]))
    tiles = pd.concat(
        [
            pd.read_parquet(
                f,
                columns=[
                    "slide_id", "x", "y", "annotation",
                    "lowgrade_dysplasia", "highgrade_dysplasia",
                ],
            )
            for f in sorted((TILING / "tiles").glob("*.parquet"))
        ]
    )
    tiles = tiles[tiles["slide_id"].isin(set(slides["id"]))].reset_index(drop=True)
    stride_of = dict(
        zip(slides["id"], np.maximum(slides["stride_x"], slides["stride_y"]).astype(float))
    )

    probs, ys = [], []
    for sid, t in tiles.groupby("slide_id", sort=False):
        kept = t[_annotated_piece_mask(t, stride_of[sid], 2.0)]
        lab = (
            (kept["lowgrade_dysplasia"] > 0) | (kept["highgrade_dysplasia"] > 0)
        ).to_numpy()
        tab = pq.read_table(EMB / f"{stem_of[sid]}.parquet", columns=["x", "y", "embedding"])
        ex = tab["x"].to_numpy().astype(np.int64)
        ey = tab["y"].to_numpy().astype(np.int64)
        pos = {(int(x), int(y)): k for k, x, y in zip(range(len(ex)), ex, ey)}
        rows = np.fromiter(
            (pos[(int(x), int(y))] for x, y in zip(kept["x"], kept["y"])),
            dtype=np.int64, count=len(kept),
        )
        mat = emb_matrix(tab)[rows]
        X = torch.from_numpy(np.ascontiguousarray(mat))
        with torch.no_grad():
            probs.append(model(X).sigmoid().numpy().ravel())
        ys.append(lab)
        print(f"{stem_of[sid]}: {len(kept)} tiles", flush=True)

    p = np.concatenate(probs)
    y = np.concatenate(ys)
    print(f"val AUC={roc_auc_score(y, p):.4f}  acc@0.5={((p > 0.5) == y).mean():.4f}  "
          f"n={len(p)} pos={int(y.sum())}", flush=True)


def _epi_one(stem: str) -> None:
    """raw level-2 epi -> level-3 (2x shrink, mpp*2), same as the
    tile-masks report's --shrink-epi."""
    import pyvips

    src = EPIS_RAW / f"{stem}.tiff"
    out = EPIS_L3 / f"{stem}.tiff"
    img = pyvips.Image.new_from_file(str(src))
    mpp = 1000 / img.xres  #: px/cm -> um/px
    write_big_tiff(img.shrink(2, 2, ceil=False), out, mpp * 2, mpp * 2)
    print(f"{stem}: epi level3 done ({out.stat().st_size / 1e6:.0f} MB)", flush=True)


def epi_shrink() -> None:
    EPIS_L3.mkdir(parents=True, exist_ok=True)
    stems = [
        s
        for s in (p.stem for p in sorted(EPIS_RAW.glob("*.tiff")))
        if not (EPIS_L3 / f"{s}.tiff").exists()
        and s in set(val_slides()["stem"])
    ]
    print(f"{len(stems)} epi masks to shrink -> {EPIS_L3}", flush=True)
    for s in stems:
        _epi_one(s)
    print("epi shrink done", flush=True)


def _client():
    from mlflow.tracking import MlflowClient

    #: 3.x clients take no credentials in the constructor -- auth rides on the
    #: environment. Job mode: the submitter exports the env vars; pod mode:
    #: ~/.mlflow_key (same as make_tile_masks_report.py).
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
    stems = set(val_slides()["stem"])
    jobs = [
        (PRED_OUT, "mlp_predictions"),
        (EPIS_L3, "epithelium_masks/level3"),
    ]
    total = sum(
        len([p for p in local.glob("*.tiff") if p.stem in stems]) for local, _ in jobs
    )
    done = 0
    for local_dir, artifact_dir in jobs:
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            have = {
                i.path.rsplit("/", 1)[-1]
                for i in client.list_artifacts(RUN, path=artifact_dir)
            }
        except Exception:
            have = set()
        for src in sorted(local_dir.glob("*.tiff")):
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


#: (display name, local dir, artifact dir, color, visible by default). A layer
#: appears for a slide only when BOTH its local file and its uploaded artifact
#: exist; the annotation layers are added per-slide (annotated slides only).
BASE_LAYERS = [
    ("Dysplasia prediction (MLP)", PRED_OUT, "mlp_predictions", "#ff44ff", True),
    ("Epithelium", EPIS_L3, "epithelium_masks/level3", "#00cc44", True),
    ("Tile outlines", OUTLINES, "tile_masks_186/outlines", "#ffffff", True),
]
LGD_LAYER = (
    "LGD annotation",
    ANNOT / "lowgrade_dysplasia",
    "annotation_masks/lowgrade_dysplasia",
    "#ffcc00",
    True,
)
HGD_LAYER = (
    "HGD annotation",
    ANNOT / "highgrade_dysplasia",
    "annotation_masks/highgrade_dysplasia",
    "#ff2200",
    True,
)


def _layers_for(row, uploaded: dict) -> list:
    layers = []
    if row["ann_tiles"]:
        for L in (LGD_LAYER, HGD_LAYER):
            if (L[1] / f"{row['stem']}.tiff").exists() and f"{row['stem']}.tiff" in uploaded.get(L[2], set()):
                layers.append(L)
    for L in BASE_LAYERS:
        if (L[1] / f"{row['stem']}.tiff").exists() and f"{row['stem']}.tiff" in uploaded.get(L[2], set()):
            layers.append(L)
    return layers


def session_doc(stem: str, bg_id: str, layers: list, shape: str) -> dict:
    """Same byte-shape as make_tile_masks_report.session_doc (proven on this
    deployment): v2 = redirect.php with the inline protocol patched back in;
    v3 = registered per-data-entry protocols (wsi_service background,
    iipimage masks) and the required `order` on the visualization."""
    mask_ids = [_remote(a, stem) for _, _, a, _, _ in layers]
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
        for i, (name, _, _, color, vis) in enumerate(layers)
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
                "name": f"{stem} mlp predictions",
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
                    for i, (name, _, _, color, vis) in enumerate(layers)
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
            "name": f"{stem} mlp predictions",
            "order": list(shaders),
            "shaders": shaders,
        }],
    }


def build_report(out: Path) -> list[Path]:
    """Writes OUT.v3.html (primary) and OUT.v2.html (proven fallback)."""
    from report_fast import Report, SlideGrid, XopatEndpoint, XopatSession

    client = _client()
    uploaded = {}
    for _, _, artifact_dir, _, _ in [*BASE_LAYERS, LGD_LAYER, HGD_LAYER]:
        try:
            infos = client.list_artifacts(RUN, path=artifact_dir)
        except Exception:
            infos = []
        uploaded[artifact_dir] = {i.path.rsplit("/", 1)[-1] for i in infos}

    df = val_slides()
    #: the split's missing-embeddings slides, for the subtitle (dynamic: if the
    #: 14-slide job lands before the report is built, they simply appear)
    all_slides = pd.read_parquet(TILING / "slides" / "slides.parquet")
    all_slides = all_slides[all_slides["split"] == SPLIT]
    n_total = len(all_slides)
    missing = sorted(
        set(all_slides["path"].map(lambda p: Path(p).stem)) - set(df["stem"])
    )
    n_ann = int((df["ann_tiles"] > 0).sum())
    missing_tail = (
        f" Slides {', '.join(missing)} await their embeddings." if missing else ""
    )
    #: single-dot filenames: the mlflow-artifacts proxy 500s on "name.v3.html"
    v3 = out.with_name(f"{out.stem}_v3.html")
    v2 = out.with_name(f"{out.stem}_v2.html")
    for shape, page in (("v3", v3), ("v2", v2)):
        base = V2 if shape == "v2" else V3
        endpoint = XopatEndpoint(base_url=base)
        grid = SlideGrid(title="Val set (test_preliminary)", endpoint=endpoint)
        for _, row in df.iterrows():
            stem = row["stem"]
            layers = _layers_for(row, uploaded)
            doc = session_doc(stem, _bg_data_id(row["path"]), layers, shape)
            if shape == "v2":
                proto = doc["background"][0].pop("protocol")
                s = XopatSession.from_config(json.dumps(doc))
                s.background[0]["protocol"] = proto
            else:
                s = XopatSession.from_config(json.dumps(doc))
            note = f"{row['ann_tiles']} annotated tiles" if row["ann_tiles"] else "unannotated"
            grid.add(s, name=stem, note=f"{len(layers)} layers, {note}")
        report = Report(
            title=f"HunCRC val set -- MLP dysplasia predictions ({len(df)} slides)",
            subtitle=(
                f"{len(df)}/{n_total} test_preliminary slides ({n_ann} annotated); "
                "magenta = per-tile dysplasia probability (2x2-tile mean), "
                "yellow/red = pathologist LGD/HGD, green = epithelium, "
                "white = tile grid." + missing_tail
            ),
            preamble=(
                "One card per val-set slide: the MLP tile-prediction mask over "
                "the original LGD/HGD annotations (annotated slides only), the "
                "epithelium mask and the tiling grid. Built from the morning "
                "run's checkpoint (training/embeddings_huncrc/last-v2.ckpt); "
                "mask geometry matches preprocessing/tile_masks. Links "
                "unverified from the build pod -- report blank cards."
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
    return (f"{MLFLOW_URL}/api/2.0/mlflow-artifacts/artifacts/"
            f"{EXPERIMENT_ID}/{RUN}/artifacts/{artifact_dir}/{html_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", action="store_true",
                    help="checkpoint AUC/acc on the filtered val set (logs first)")
    ap.add_argument("--predict", action="store_true",
                    help="per-tile probabilities -> level-3 tiff per val slide")
    ap.add_argument("--epi", action="store_true",
                    help="raw epi masks -> level-3 for the val stems (resumable)")
    ap.add_argument("--upload", action="store_true",
                    help="upload prediction + epi tiffs into the run (resumable)")
    ap.add_argument("--report", type=Path, metavar="OUT.html",
                    help="build the val-set report (v3 + v2 pages)")
    ap.add_argument("--publish", action="store_true",
                    help="log the built pages to the run and print the URLs")
    args = ap.parse_args()

    if args.eval:
        eval_val()
    if args.predict:
        predict_all()
    if args.epi:
        epi_shrink()
    if args.upload:
        upload()
    pages = []
    if args.report:
        pages = build_report(args.report)
    if pages and args.publish:
        for p in pages:
            print("PUBLISHED:", publish(p))
