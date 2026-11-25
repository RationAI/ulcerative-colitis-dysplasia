#!/usr/bin/env python
"""Build the 186-slide hunCRC tile-masks xOpat report (+ original masks).

Probe verdict (2026-10-26, slide 169): the tile server serves masks ONLY
through the MLflow-artifact route on this cluster -- /public_mlflow/168/
{RUN}/artifacts/... works, projects/... is blank -- and the proven session
shape is the redirect.php one from make_epi_viewer_link.py (v2-style
inline background protocol). So every mask layer is uploaded into RUN and
linked from there.

Layers per slide (added only when the slide has the file, in this order):
  * Epithelium          -- raw epi mask shrunk 2x (level 3, canvas matches
                           the tile masks), green, visible
  * LGD / HGD           -- the original annotation masks (level 5, 0/255;
                           only the 105/35 annotated slides have them)
  * Tile outlines       -- white grid, visible
  * Epithelium/LGD/HGD (tiles), Blur (tiles), Artifacts (tiles) -- the
                           tile_masks_186 scalar maps, hidden by default

Usage (from the notebook pod):
    # once: shrink the 37 GB raw epi masks to ~9 GB level-3 versions
    .venv/bin/python scripts/make_tile_masks_report.py --shrink-epi
    # resumable: uploads every layer dir into RUN, skips files already there
    nohup .venv/bin/python scripts/make_tile_masks_report.py --upload \
        > /tmp/upload_masks.log 2>&1 &
    # build + publish the report (a layer appears once its file is uploaded)
    .venv/bin/python scripts/make_tile_masks_report.py --report /tmp/tile_masks_report.html --publish

Session shape follows scripts/make_epi_viewer_link.py (proven
2026-09-30 on xopat.rationai.cloud.trusted.e-infra.cz; see
reportfast-feedback.md). ReportFast 0.1.0 joins base_url with '/#'
(white-screens redirect.php), so the HTML is post-processed after write().

MLflow creds: ~/.mlflow_key (0600) + the e-mail below; the public tracking
URL -- the pod cannot resolve the in-cluster one.
"""

import argparse
import json
import urllib.parse
from pathlib import Path

import pandas as pd
from ratiopath.masks import write_big_tiff

PROJECT = Path("/mnt/projects/huncrc_dysplasia")
TILING = PROJECT / "tiling_186_merged"
TILE_MASKS = PROJECT / "tile_masks_186"
EPIS_RAW = PROJECT / "epithelium_masks" / "raw"
EPIS_L3 = PROJECT / "epithelium_masks" / "level3"
ANNOT = PROJECT / "annotation_masks"
EXPERIMENT_ID = 168
RUN = "45a957c5a79d4560b2b3efb68e19eed6"  #: report/publish run (proven artifact host)
VIEWER = "https://xopat.rationai.cloud.trusted.e-infra.cz"
V2 = f"{VIEWER}/xopat/redirect.php"
V3 = f"{VIEWER}/v3/"  #: v3 mount; ReportFast's '/#' join is correct for this one
MLFLOW_URL = "https://mlflow.rationai.cloud.e-infra.cz"
MLFLOW_USER = "519192@mail.muni.cz"

#: (display name, local dir, artifact subdir, color, visible by default).
#: Layer file for a slide = local_dir / f"{stem}.tiff", linked as
#: /public_mlflow/{EXPERIMENT_ID}/{RUN}/artifacts/artifact_dir/{stem}.tiff.
LAYERS = [
    ("Epithelium", EPIS_L3, "epithelium_masks/level3", "#00cc44", True),
    ("LGD annotation", ANNOT / "lowgrade_dysplasia", "annotation_masks/lowgrade_dysplasia", "#ffcc00", True),
    ("HGD annotation", ANNOT / "highgrade_dysplasia", "annotation_masks/highgrade_dysplasia", "#ff2200", True),
    ("Tile outlines", TILE_MASKS / "outlines", "tile_masks_186/outlines", "#ffffff", True),
    ("Epithelium (tiles)", TILE_MASKS / "epithelium", "tile_masks_186/epithelium", "#99dd00", False),
    ("LGD (tiles)", TILE_MASKS / "lowgrade_dysplasia", "tile_masks_186/lowgrade_dysplasia", "#ffcc00", False),
    ("HGD (tiles)", TILE_MASKS / "highgrade_dysplasia", "tile_masks_186/highgrade_dysplasia", "#ff2200", False),
    ("Blur (tiles)", TILE_MASKS / "blur", "tile_masks_186/blur", "#cc44ff", False),
    ("Artifacts (tiles)", TILE_MASKS / "artifacts", "tile_masks_186/artifacts", "#00ccff", False),
]


def _client():
    import os

    from mlflow.tracking import MlflowClient

    #: 3.x clients take no credentials in the constructor -- auth rides on the
    #: environment (make_epi_viewer_link.py still has the old username/password
    #: kwargs and would fail the same way).
    key = Path.home().joinpath(".mlflow_key").read_text().strip()
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = key
    return MlflowClient(tracking_uri=MLFLOW_URL)


def _slides() -> pd.DataFrame:
    df = pd.read_parquet(TILING / "slides")
    df["stem"] = df["path"].map(lambda p: Path(p).stem)
    assert df["stem"].is_unique
    return df


def _bg_data_id(path: str) -> str:
    """mrxs path on the DATA share -> DataID (path relative to /mnt)."""
    p = str(path)
    assert p.startswith("/mnt/"), p
    return p[len("/mnt/"):]


def _remote(artifact_dir: str, stem: str) -> str:
    return f"/public_mlflow/{EXPERIMENT_ID}/{RUN}/artifacts/{artifact_dir}/{stem}.tiff"


def _uploaded_files(client) -> dict[str, set]:
    """artifact dir -> {f'{stem}.tiff'} already in the run (upload gate).

    No `_exists` preflight: mlflow 3.x `get_artifact_details` on a DIRECTORY
    raises, and the gate silently zeroed every layer (cards said "0 mask
    layers"). `list_artifacts` alone is the whole check."""
    out = {}
    for _, _, artifact_dir, _, _ in LAYERS:
        try:
            infos = client.list_artifacts(RUN, path=artifact_dir)
        except Exception:
            infos = []
        out[artifact_dir] = {i.path.rsplit("/", 1)[-1] for i in infos}
    return out


def _exists(client, path: str) -> bool:
    try:
        client.get_artifact_details(RUN, path=path)
        return True
    except Exception:
        return False


def session_doc(stem: str, bg_id: str, layers: list, shape: str) -> dict:
    """One slide + its mask heatmaps.

    shape "v2": the proven redirect.php byte-shape (make_epi_viewer_link.py,
    2026-09-30). shape "v3": the golden-example form (data[] entries with the
    registered wsi_service protocol; "order" on the visualization is required
    -- my first v3 probe lacked it and rendered nothing).
    """
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
                "name": f"{stem} tile masks",
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
        #: wsi_service = slides only (mrxs); MLflow-artifact TIFFs resolve via
        #: iipimage (proven 2026-10-27, proto probe 2 row G -- the tile server's
        #: PathsMapper 404s artifacts, the IIP store serves them).
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
            "name": f"{stem} tile masks",
            "order": list(shaders),
            "shaders": shaders,
        }],
    }


def _shrink_one(src: str) -> str:
    """One raw epi mask -> level-3 (2x shrink, mpp*2). Module-level: the
    Pool worker cannot pickle a closure."""
    import pyvips

    src = Path(src)
    img = pyvips.Image.new_from_file(str(src))
    mpp = 1000 / img.xres  #: px/cm -> um/px
    out = img.shrink(2, 2, ceil=False)
    write_big_tiff(out, EPIS_L3 / src.name, mpp * 2, mpp * 2)
    print(f"{src.name} done ({(EPIS_L3 / src.name).stat().st_size / 1e6:.0f} MB)",
          flush=True)
    return str(src)


def shrink_epitile() -> None:
    """raw level-2 epi -> level-3 (2x shrink = the tile-mask canvas, mpp*2)."""
    import multiprocessing as mp

    EPIS_L3.mkdir(parents=True, exist_ok=True)
    jobs = [
        str(src) for src in sorted(EPIS_RAW.glob("*.tiff"))
        if not (EPIS_L3 / src.name).exists()
    ]
    print(f"{len(jobs)} epi masks to shrink -> {EPIS_L3}", flush=True)
    with mp.Pool(4) as pool:
        pool.map(_shrink_one, jobs)
    print("shrink done", flush=True)


def upload_masks() -> None:
    client = _client()
    client.get_run(RUN)  #: fail fast on auth
    #: raw epi dir holds all 200 slides; the report covers the 186 tiling'd
    #: stems, so skip the rest (they would only waste egress).
    stems = {row["stem"] for _, row in _slides().iterrows()}
    done = total = 0
    for name, local_dir, artifact_dir, _, _ in LAYERS:
        local_dir.mkdir(parents=True, exist_ok=True)
        have = _uploaded_files(client).get(artifact_dir, set())
        for src in sorted(local_dir.glob("*.tiff")):
            if src.stem not in stems:
                continue
            total += 1
            if src.name in have:
                done += 1
                continue
            client.log_artifact(RUN, str(src), artifact_path=artifact_dir)
            have.add(src.name)
            done += 1
            print(f"[{done}/{total}] {artifact_dir}/{src.name}", flush=True)
    print(f"upload done: {done}/{total} files in the run")


def publish(html_path: Path, artifact_dir: str = "report") -> str:
    #: log_artifact treats artifact_path as a DIRECTORY and appends the
    #: basename -- so pass the dir only, or the name doubles in the path.
    client = _client()
    client.log_artifact(RUN, str(html_path), artifact_path=artifact_dir)
    return (f"{MLFLOW_URL}/api/2.0/mlflow-artifacts/artifacts/"
            f"{EXPERIMENT_ID}/{RUN}/artifacts/{artifact_dir}/{html_path.name}")


def _build_shape(shape: str, out: Path, df: pd.DataFrame,
                 uploaded: dict[str, set]) -> Path:
    from report_fast import Report, SlideGrid, XopatEndpoint, XopatSession

    base = V2 if shape == "v2" else V3
    endpoint = XopatEndpoint(base_url=base)
    grid = SlideGrid(title="Slides", endpoint=endpoint)
    covered = {name: 0 for name, *_ in LAYERS}
    for _, row in df.iterrows():
        stem = row["stem"]
        layers = [
            L for L in LAYERS
            if (L[1] / f"{stem}.tiff").exists() and f"{stem}.tiff" in uploaded[L[2]]
        ]
        for L in layers:
            covered[L[0]] += 1
        doc = session_doc(stem, _bg_data_id(row["path"]), layers, shape)
        if shape == "v2":
            #: from_config round-trip, then patch raw (0.1.0 rejects backticks)
            proto = doc["background"][0].pop("protocol")
            s = XopatSession.from_config(json.dumps(doc))
            s.background[0]["protocol"] = proto
        else:
            s = XopatSession.from_config(json.dumps(doc))
        grid.add(s, name=stem, note=f"{len(layers)} mask layers")
    report = Report(
        title=f"HunCRC tile masks + original masks -- 186 slides (xOpat {shape})",
        subtitle=f"186 cases; layer coverage: "
                 + ", ".join(f"{k} {v}/186" for k, v in covered.items()),
        preamble=(
            "One card per slide: the .mrxs with original epithelium / LGD / "
            "HGD masks and the tile-level maps (outlines, per-tile epithelium, "
            "dysplasia, blur, artifacts) from tiling_186_merged. Visible by "
            "default: epithelium, dysplasia annotations, tile outlines; the "
            "tile-level twins and QC maps toggle on in the viewer. Links "
            "unverified from the build pod -- report blank cards."
        ),
    )
    report.add(grid)
    report.write(str(out))
    if shape == "v2":
        #: ReportFast 0.1.0 joins base with '/#'; redirect.php white-screens
        #: on it (the /v3/ mount wants '/#', so v3 needs no fix).
        html = out.read_text()
        fixed = html.replace(f"{V2}/#", f"{V2}#")
        if fixed != html:
            out.write_text(fixed)
    return out


def build_report(out: Path) -> list[Path]:
    """Writes OUT.v3.html (primary) and OUT.v2.html (proven fallback)."""
    client = _client()
    uploaded = _uploaded_files(client)
    df = _slides().sort_values("stem")
    #: single-dot filenames: the mlflow-artifacts proxy 500s on "name.v3.html"
    v3 = out.with_name(f"{out.stem}_v3.html")
    v2 = out.with_name(f"{out.stem}_v2.html")
    _build_shape("v3", v3, df, uploaded)
    _build_shape("v2", v2, df, uploaded)
    return [v3, v2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shrink-epi", action="store_true",
                    help="raw epi masks -> level-3 versions (4-way, resumable)")
    ap.add_argument("--upload", action="store_true",
                    help="upload all layer dirs into the run (resumable)")
    ap.add_argument("--report", type=Path, metavar="OUT.html",
                    help="build the 186-slide report (v3 + v2 pages)")
    ap.add_argument("--publish", action="store_true",
                    help="log the built pages to the run and print the URLs")
    args = ap.parse_args()

    if args.shrink_epi:
        shrink_epitile()
    if args.upload:
        upload_masks()
    pages = []
    if args.report:
        pages = build_report(args.report)
        for p in pages:
            print(f"report -> {p}")
    if pages and args.publish:
        for p in pages:
            #: publish() appends the basename itself -- dir only, or the name doubles
            url = publish(p, "report")
            print("PUBLISHED:", url)
