"""Smoke test: scripts/huncrc/prelim_breakdown.py score + report on synthetic runs."""

import json
from pathlib import Path

from tests.huncrc_synth import build, run_script


def test_prelim_breakdown_score_and_report(tmp_path: Path) -> None:
    c = build(tmp_path)
    runs, out = tmp_path / "runs", tmp_path / "bd"
    common = ["--train-cache", c["train"], "--prelim-cache", c["test_preliminary"]]
    run_script("linear_cv.py", "train", *common, "--runs", runs, "--epochs", 1)
    bd = [
        "--runs", runs,
        "--all-cache", c["test_preliminary_all"],
        "--prelim-cache", c["test_preliminary"],
    ]  # fmt: skip
    proc = run_script("prelim_breakdown.py", "score", *bd)
    assert proc.stdout.count("tiles scored") == 10
    # resumable: a second run skips everything
    assert run_script("prelim_breakdown.py", "score", *bd).stdout.count("skipped") == 10
    run_script(
        "prelim_breakdown.py", "report", *bd,
        "--tiling", c["tiling"], "--slide-csv", c["slide_csv"],
        "--out", out, "--n-boot", 20,
    )  # fmt: skip
    res = json.loads((out / "breakdown.json").read_text())
    # 5 prelim slides: 2 adenoma (pos), negative + hyperplastic (safe), CRC without LG (other)
    assert res["n_slides"] == {"pos": 2, "safe": 2, "other": 1}
    crc = [r for r in res["slides"].values() if r["class"] == "other"]
    assert len(crc) == 1 and crc[0]["n_labelled"] == 0 and crc[0]["a_lab_mean"] is None
    for v in ("a", "b"):
        R = res["variants"][v]
        assert R["5a"]["spec_hyperplastic"]["n_slides"] == 1
        assert (
            R["5a"]["sens_hg_tiles"]["value"] is None
        )  # no HG in the synthetic cohort
        for agg in ("mean_all", "top1pct_mean", "frac_above_thr"):
            d = R["5b_slide_auroc"][agg]
            assert (
                d["pos_vs_safe"]["n_pos"] == 2
                and d["pos_plus_crc4_vs_safe"]["n_pos"] == 3
            )
        assert set(R["5b_diag"]) == {"annotated_lghg", "unannotated"}
    assert "5b slide AUROC" in (out / "breakdown.txt").read_text()
