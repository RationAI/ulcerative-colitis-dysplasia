# hunCRC dysplasia project — state of the world

Living document: what we have, where it lives, and why. Update as things
change; the "Decisions" sections are the record of what the team agreed.

Last updated: 2026-11-05.

## 1. What we are doing

Binary **dysplasia detection** (LGD or HGD present) on the **TCIA_HunCRC**
cohort — 200 MIRAX whole-slide images from Hungarian colorectal screening.
Current result: an MLP on prov-gigapath tile embeddings, **val AUC 0.9528**
(monitoring/early-stopping on the val split = `test_preliminary`), MLflow run
`45a957c5a79d4560b2b3efb68e19eed6` (exp 168), published.

## 2. The data

- **Slides**: `/mnt/data/Public/Public/colorectum/TCIA_HunCRC/PKG - Hungarian-Colorectal-Screening_v2/*.mrxs` (200).
- **Slide metadata**: `.../TCIA_HunCRC/slide_level_annotations.csv` — 200 rows:
  `slideID, CATEGORY, Haggit-level, Biopsy or polyp, Polyp type`.
  Categories: **adenoma 106, CRC 46, non_neoplastic_lesion 38, negative 10**.
  Polyp types: tubular 88, tubulovillous 15, hyperplastic 24, sessile_serrated 1,
  Not_applicable rest. **Known oddity: one `non_neoplastic_lesion` slide is typed
  `tubular_adenoma`** — excluded from the v2 negative pool (see §7).
  Also `metadata_age_sex_icd10status.csv` (age/sex/ICD-10) — not used yet.
- **Annotations**: per-slide label CSVs in `..._ML-dataset/.../zoom_1_partC/`
  (one class per cell of a regular level-0 square grid; pitch 1024 px, except
  slide **077** which is zoom-2 with pitch 2048). Converted losslessly to EMPIA
  JSON by `scripts/make_huncrc_annotations.py` into
  `/mnt/projects/huncrc_dysplasia/preprocessing/annotations/` (**ground truth**,
  do not delete). Vocabulary = exactly 9 classes, 408,915 label squares total:
  `normal 132,692 · lowgrade_dysplasia 219,896 · highgrade_dysplasia 11,424 ·
  adenocarcinoma 17,980 · artifact 16,060 · suspicious_for_invasion 2,712 ·
  tumor_necrosis 2,367 · inflammation 3,976 · resection_edge 1,808`.
  No LVI or other labels exist in the data (despite the literature).
- **Splits** (slide-level, production): **train 140 / test_preliminary 30 /
  test_final 30**. val split = `test_preliminary` (monitored),
`test_final` = held-out test.

## 3. The pipeline, stage by stage

All on-disk outputs under `/mnt/projects/huncrc_dysplasia/preprocessing/`,
all MLflow runs in experiment **168** (public server
`https://mlflow.rationai.cloud.e-infra.cz`).

| Stage | On disk | MLflow | Notes |
|---|---|---|---|
| QC masks | `quality_control/` (**8.5 G**) | main run `574f5c6d` = 186 slides only | 4 families (blur/artifacts × per_pixel/per_tile) + `qc_metrics.csv`. See §4. |
| Tissue masks | `tissue_masks/` (1.8 G) | `75812a7e` | 200 masks; uploaded OK. |
| Epithelium masks | `epithelium_masks/level3/` (1.6 G) | `cc9822d8` | `raw/` (37 G) = service staging, deletable. Tiling did **not** use epi — tissue masks double as the epithelium stand-in (`preprocessing/tiling.py:resolve_epithelium_folder`). |
| Tiling | `tiling_200/` (55 M) | no run on this server (job ran in-cluster) | 224 px @ 0.52 mpp (level 2, ds 4.0), stride 112. ~4.3 M tiles, tile table carries `split`, `blur`, `artifacts`, `tissue`, `clarity`, `fold` + only the 2 dysplasia classes. |
| Embeddings | `embeddings/prov-gigapath/` (39 G, 200 parquets) | `c844af9f` | one parquet per slide, rows = tiling_200 tiles. **Do not delete** (model input; recomputation = new gigapath job). |
| Tile masks (report layers) | `tile_masks/` (8.4 G) | uploaded to report runs | 186-slide era → **superseded, deletable** (see §6). |
| Annotation restamp (9 classes) | `tiling_7annot/` (7 slides), `tiling_200_allclasses/` (200, NEW 2026-11-05) | — | see §7. |
| Training | `training/` | run `45a957c5` | MLP, see §7 for the v2 change. |

Superseded on-disk dirs (deleted 2026-11-05 after this review):
`epithelium_masks/raw/`, `tile_masks/`, `tiling/`, `tiling_missing14/`,
`dataset_split.csv`, `prov_test/qc/*.dat` — ~52 G.

## 4. QC: history and the one-copy risk

- QC = the rationai QC service: 4 mask families + `qc_metrics.csv`
  (`mean_coverage(Piqe)` = blur, `mean_coverage(ResidualArtifactsAndCoverage)`
  = artifacts, …; `slide_name` is the **last** column and stored unpadded —
  pandas re-parses `059` → `59`; tiling zero-pads the index,
  `preprocessing/tiling.py:240-246`).
- **Tiling does NOT filter on QC.** Only filter = `tissue ≥ 0.5` (central
  112×112 half-ROI, from the tissue mask). QC is *recorded*: slide-level
  `blur_mean`/`artifacts_mean` (hard dependency — `qc_agg` KeyErrors on a
  missing slide) and per-tile `blur`/`artifacts` = mean mask value over the
  full 224×224 tile. Tiling downloads the whole ~8 GB QC tree but reads only
  the two `*_per_pixel` subdirs.
- The original job (`574f5c6d`, "🎭 QC Masks Creation") completed **186/200**;
  the 13 dead + 1 in-flight slides were retried (job run `12bc1f64`). The
  merged 200-slide tree was then **never successfully uploaded**: the merge
  upload run `0e3d8a66` ("QC output (200 slides, huncrc") **failed with zero
  artifacts**. Verified 2026-11-02: a scan of all 35 exp-168 runs finds the
  only `qc_output` on `574f5c6d` (186 masks).
- **The 200-slide tree exists in exactly one place:
  `preprocessing/quality_control/`** (200 files × 4 families + 200-row CSV;
  verified 2026-11-02, including slide 059's row matching tiling_200's values
  exactly). It was moved here from `…/huncrc_dysplasia/quality_control` during
  the disk reorg. **Do not delete**; re-upload to MLflow when possible
  (`scripts/rerun_missing_qc.py --upload` — its `JOB_OUT` constant in
  `scripts/qc_one_slide.py:111` still points at the old path and needs a
  one-line update first).
- Slide 010 (colleague's "completely unusable" example): blur_mean 0.718 =
  9th worst of 200 (median 0.542), 54.9 % of its tiles have per-tile blur > 0.5,
  0 dysplasia-annotated tiles, split = `test_preliminary` → it contributed
  nothing to the published val AUC. It is a **blur** problem, not artifacts.

## 5. Reports (all published, all on run `45a957c5`)

- MLP prediction report — `scripts/make_mlp_prediction_report.py`.
- 186-slide tile-masks report.
- **7-slide all-classes annotation report** — `scripts/make_annot_inspection_report.py`,
  pages `report/annot7_report_v3.html` (primary) + `_v2.html` (fallback).
  Design decision (2026-11-01): a class appears as a layer **only if the slide
  has tiles of it** — the layer list itself shows what the slide is annotated
  with. 8 of 9 classes covered across the 7 slides (no HGD slide in the set).
  Built with ReportFast (the user's own tool — keep usability notes in
  `reportfast-feedback.md`).

## 6. The tiling-vs-annotations label problem (superseded by §7)

`tiling_200`'s dysplasia columns disagree with the current EMPIA JSONs.
**Measured 2026-11-05** (restamp cross-check, all 200 slides): **132/200
slides, 225,986 / 4,318,193 tiles (5.2 %)** differ by > 1e-9. Worst: 179
(15,979), 095 (14,796), 133 (9,272), 059 (9,066), 114 (8,750). Exact cause
still not established (JSONs converted after tiling, or re-issued).
**Status 2026-11-05: effectively fixed** — the v2 label policy (below)
re-derives every label from the current JSONs
(`preprocessing/tiling_200_allclasses/`), so the tiling columns are no
longer used for training labels. Per-slide counts: rerun
`scripts/restamp_allclasses_200.py` (idempotent, ~1 min).

## 7. Label policy v2 (decided 2026-11-05, per Adam Kukučka)

**Context**: Adam reviewed the 7-slide report and the prediction report:
(1) the pathologists **only annotated part of each slide**, so unannotated
tiles are not trustworthy negatives; (2) he suspects the model latched onto
**carcinoma** as a proxy for dysplasia — the current negative class is *all*
non-dysplasia tiles, so carcinoma tiles are in it today
(`ml/data/embeddings_dataset.py:172`); (3) he wants a QC-based blur cut.

**The policy** (tile × slide-category × QC):

- **Positive** = tile with any LGD/HGD coverage.
- **Negative**, only:
  - `normal` / `inflammation` tiles on **negative / non_neoplastic** slides —
    no conditions (the slide diagnosis guarantees no dysplasia; unannotated
    tiles OK). Non_neoplastic slides typed serrated or the one
    tubular_adenoma outlier: whole slide dropped.
  - `normal` / `inflammation` tiles on **adenoma / CRC** slides — **annotated
    tiles only**, ≥ ~500 µm from any neoplastic annotation (buffer set:
    adenoma → dysplasia classes; CRC → carcinoma/SFI/necrosis/dysplasia), no
    dysplasia overlap, weight 0.5 (CRC = noisiest pair, first to drop).
  - optional robustness set: `artifact` / `resection_edge` tiles on
    negative/non_neoplastic slides, **only when our QC doesn't flag them**
    (Adam: artifact labels are judged by our QC, not the pathologist's).
- **Excluded**: unannotated tiles on adenoma/CRC slides; all carcinoma /
  SFI / necrosis tiles; any tile with QC `blur > 0.5` (both classes,
  symmetric); any tile with QC `artifacts > 0.5`.
- 500 µm ≈ 240 level-0 px (mpp 0.52 × ds 4 = 2.08 µm/px).

**Open decisions to confirm with Adam** (defaults in parentheses):
1. Unannotated tiles on the 48 negative/non_neoplastic slides count as
   negatives (yes — otherwise no easy negatives at all).
2. Blur threshold 0.5, applied to positives too (yes).
3. 0.5 down-weight on neoplastic-slide negatives (yes).
4. Keep the artifact/resection-edge robustness set (yes).
5. "≥500 µm from any annotation" read as "from any **neoplastic** annotation"
   (a normal tile is inside its own normal annotation, so the literal reading
   is impossible).

**Implementation** (done 2026-11-05; new files, production configs untouched):

- **Stage A** ✅ — `scripts/restamp_allclasses_200.py`: per-tile coverage of
  all 9 classes for all 200 slides by grid lookup on the label squares (same
  math as the 7-slide restamp; 224-px tile at level 2 = 896 level-0 px <
  pitch, so ≤ 2×2 squares per tile; vectorized, ~1 min, no WSI pixels).
  Output: `preprocessing/tiling_200_allclasses/tiles/{stem}.parquet` =
  tiling_200 tile rows + `annot_{class}` columns (idempotent, resumable).
- **Stage B** ✅ — `scripts/make_training_tiles.py`: the policy as one pure
  function (`assign_policy`) + the join. Output:
  `preprocessing/training_tiles/{split}.parquet` (slide_id, x, y, split,
  `label`, `weight`, `negative_type`, …) + `excluded.parquet` (audit:
  reason per excluded tile, `dys` flag). Re-run with `--blur-threshold`
  etc. to try other cuts — nothing downstream changes.
- **Stage C** ✅ — `ml/data/embeddings_dataset.py` gained
  `label_table_dir` (mutually exclusive with `cluster_k`): inner-joins the
  v2 table on `(slide_id, x, y)`, hard-errors if a table row has no grid
  tile, labels/weights from the table, `weight` in the batch metadata.
  `ml/modeling/embeddings.py` `training_step` now uses
  `binary_cross_entropy_with_logits(..., weight=meta["weight"])` — with the
  legacy convention the weights are 1.0, so old runs are unaffected. New
  configs `configs/dataset/embeddings_huncrc200_v2.yaml` +
  `configs/experiment/embeddings_huncrc200_v2.yaml` (val =
  `test_preliminary` as in v1, ckpt dir `training/embeddings_huncrc200_v2`),
  submitter `scripts/embeddings_huncrc200_v2.py` (A100 job, same shape as
  the v1 one). Smoke-tested on the pod (3-slide join + weighted training
  step + legacy-path regression check).
- **Stage D** ⏳ — retrain + validate. Submit:
  `export MLFLOW_TRACKING_PASSWORD=…; uv run python scripts/embeddings_huncrc200_v2.py`
  (needs the branch pushed first). Then: AUC on the val split + a
  prediction report with all-9-class overlays + a "carcinoma-as-proxy" check
  (score held-out carcinoma tiles the model was never trained on).

**Stage B results (defaults: blur > 0.5, artifacts > 0.5, buffer 500 µm,
robust negatives on, neoplastic-slide weight 0.5):**

| split | tiles | pos | neg |
|---|---|---|---|
| train | 1,026,685 | 568,139 | 458,546 |
| test_preliminary (val) | 182,310 | 90,626 | 91,684 |
| test_final | 330,754 | 218,140 | 112,614 |

1,539,649 tiles kept of 4,318,193. Excluded: unannotated_neoplastic_slide
1,336,630 · blur 1,325,285 · neoplastic_label 69,523 · buffer 16,219 ·
qc_artifact 11,848 · artifact_neoplastic_slide 10,558 · polyp_type 8,381.

**⚠ The blur cut is the expensive one — 319,452 positives (26.6 % of all
1,198,732 dysplasia tiles) die at blur > 0.5.** Sensitivity:

| blur T | positives kept |
|---|---|
| 0.2 | 296,706 (24.8 %) |
| 0.3 | 524,642 (43.8 %) |
| 0.4 | 720,559 (60.1 %) |
| 0.5 | 879,280 (73.4 %) |

Blur is genuinely pervasive (per-tile blur > 0.5 on ~31 % of all tiles).
Decide before Stage D: threshold 0.5 (default) vs higher vs down-weighting
blurred positives instead of dropping them.

## 8. Provenance cut-down (rationai-provenance chain, 2 slides: 059/155)

Chain in exp **212** (`provenance_huncrc_test`) + registry runs in
`Dataset_Registry` (211): cohort `4ef2fb98` → create_dataset `bc5160ba` →
split `2385fd81` → QC **reuse** `d9af25ea` (reuses production run `0e3d8a66`,
masks from the share — the share is the only copy, §4) → tissue `5d812b44` →
tiling **reuse** `780b3e37` (filters tiling_200 to the 2 slides) →
embeddings **reuse** `dd529985`. **Pending: tile_masks, then training** (local
on the pod, CPU). All cut-downs under `provenance/`; they pin run ids as
constants and refuse to run on `TODO` pins. Cleanup list (runs to delete):
`26f8a20e` (168), `5978d867` (212), `9044d2e2` (212), `0a94d709` (failed
tissue), `ca884744` (compute-QC, killed), `a0fec4a1` (stale embeddings),
`03ee3030` (duplicate tiling, deleted by user).

## 9. Environment & operational rules

- **Notebook pod**: 32 GB cgroup cap, silent OOM-kills on lazy numpy
  allocation — no oversized arrays; heavy work goes to kube-jobs.
- **Ray/scripts on the pod**: always `.venv/bin/python`, **never `uv run`**
  (ray workers hang under it).
- **MLflow**: public URL `https://mlflow.rationai.cloud.e-infra.cz`, user
  `519192@mail.muni.cz`, key in `~/.mlflow_key` (0600). In-cluster tracking
  URI only works from kube-jobs. Known server quirk: `download_artifacts`
  on a missing path hangs at 0 %; `list_artifacts` on missing paths returns
  empty (so "0 files" ≠ "empty dir"); public-API artifact downloads
  generally unreliable — prefer listing/tags or in-cluster.
- **xOpat viewer** `https://xopat.rationai.cloud.trusted.e-infra.cz` is
  unreachable from the pod — report links are unverified from here; DataID
  root = path relative to `/mnt/`, raw spaces, artifact prefix
  `/public_mlflow/…` (see `reportfast-feedback.md` for the proven session
  byte-shapes).
- Git: user pushes from their terminal (no GitHub creds on the pod); commits
  end with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
