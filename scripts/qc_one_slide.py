r"""Check one slide through the QC service, from the notebook, to prove the path works.

    uv run python scripts/qc_one_slide.py --slide 011.mrxs
    uv run python scripts/qc_one_slide.py --slide 011.mrxs --out /mnt/projects/huncrc_dysplasia/quality_control

Answers three things the job log cannot, because the job goes green whether or not any
mask is ever written (rationai/resources/qc.py:151-154 turns every failure into a
success=False row that only appends to qc_errors.log):

  1. Does the QC service answer at all from the cluster?
  2. Can it READ the slide -- i.e. does it mount rationai/Data at /mnt/data/Public?
  3. Can it WRITE -- i.e. does it mount rationai-projects at /mnt/projects, at the
     exact path we send in output_path?

Items 2 and 3 are the point. The service writes the masks itself: output_path travels in
the request body (resources/qc.py:50), so the pod never touches the files, and a path the
service cannot see fails silently. The notebook shares the same mounts under the same uid
1000, so this script writes to the same directory the job would and you can watch it.

The xOpat link it returns is the real deliverable: open it and you see the masks overlaid
on the slide, which is the only end-to-end proof that focus/artifact masks were produced
and not just an HTTP 200.

BUDGET YOUR TIME: one slide takes ~3.6 HOURS, not minutes. Measured from the completed
run's mtimes: 186 slides over 476695s = one completion every ~43 min, and that is the
INTER-ARRIVAL rate at max_concurrent 5 (configs/preprocessing/qc.yaml:5), so per-slide
service time is 43 min x 5 ~= 3.6 h. --timeout therefore defaults to qc.yaml's own
request_timeout of 18000s; an hour is not enough and fails a slide that is working fine.

Writes to a scratch dir (qc_probe), NOT the job's quality_control dir. Point --out at the
job dir only when the slide succeeds: files are named per slide, so this adds slides
without disturbing others, but a partial slide's masks would be uploaded to MLflow
alongside the job's good ones. The intended loop is: run here, inspect the 4 masks and the
xOpat link, then --out the job dir for the slides that are still missing.


RESUMING WHERE THE 5-DAY JOB LEFT OFF -- ledger as at 2026-10-07 12:56
----------------------------------------------------------------------
200 slides. 186 have a csv (done), 13 are dead, 1 was still in flight: 186 + 13 + 1 = 200.

  dead   (error line, no csv)      011 036 039 042 045 059 064 086
                                   091 125 155 162 174
  in flight (neither csv nor error) 194
  timed out but recovered anyway   002 006 019 024 030 053 088 093 094 149 160 176 182

Recompute rather than trusting this list -- it rots the moment anything else writes:

    Q=/mnt/projects/huncrc_dysplasia/quality_control
    SL="/mnt/data/Public/Public/colorectum/TCIA_HunCRC/PKG - Hungarian-Colorectal-Screening_v2"
    ls "$SL" | sed -n 's/\.mrxs$//p' | sort -u > /tmp/all.txt
    ls $Q/*.csv | xargs -n1 basename | sed 's/\.csv$//' | sort -u > /tmp/done.txt
    grep -o '/[0-9]\+\.mrxs' $Q/qc_errors.log | tr -d / | sed 's/\.mrxs$//' | sort -u > /tmp/err.txt
    comm -23 /tmp/err.txt /tmp/done.txt                    # dead: retry these
    comm -23 /tmp/all.txt <(sort -u /tmp/done.txt /tmp/err.txt)   # in flight: wait

WHY THEY ARE MISSING -- the 26 error lines are two different failures, and only one of
them means "too slow". Read the text after the colon, it is the whole diagnosis:

  * 25 lines say `Failed to process .../011.mrxs: ` with NOTHING after the colon. The
    empty string is the signature: `str(httpx.ReadTimeout())` is `''`, so the message is
    the exception type, not the slide's problem. That is the CLIENT giving up at
    configs/preprocessing/qc.yaml:4 request_timeout=18000s (5 h). It says nothing about
    the slide. The service keeps working after the client disconnects, which is how 13
    of those 25 later produced a csv anyway -- proof, not speculation, since they appear
    in both files.
  * 1 line, slide 042 at 12:39 today, says
    `RetryError[<Future ... raised HTTPStatusError>]`. Different failure: the service
    ANSWERED, with HTTP 500, three times -- tenacity retries only on 500
    (rationai/resources/qc.py:15-26). 500 is a real error inside the service, not slow
    service, so 042 may fail again on retry for a reason the other 12 do not have.

Neither case is ever retried by the job: check_slides issues one request per slide and
@retry re-sends only on 500, never on a client timeout. So a slide that times out is
abandoned by the job for good even if the service finishes it minutes later, and a green
job exit proves nothing because failures only append to qc_errors.log
(rationai/resources/qc.py:151-155).

To retry the dead 13, once the job releases the service -- ~9 h at qc.yaml's
max_concurrent 5 (13 x 3.6 h / 5), one overnight run:

    for s in 011 036 039 042 045 059 064 086 091 125 155 162 174; do
        uv run python scripts/qc_one_slide.py --slide $s.mrxs \
            --out /mnt/projects/huncrc_dysplasia/quality_control --timeout 18000
    done

Use --out the job dir here (not the scratch default) since these are expected to succeed,
and note the job's organize_masks moves masks into subdirs as its last step, so if it has
completed by then the masks you add later land flat beside `blur_per_tile/` etc. rather
than inside them -- the layout is inconsistent, not wrong.
"""

import argparse
import sys
import time
from pathlib import Path

import httpx
from rationai.client import Client
from rationai.types import SlideCheckConfig


#: The hunCRC mount root as the QC service must see it. storage.public.DATA mounts
#: rationai/Data at /mnt/data/Public, and configs/dataset/raw.yaml resolves slides under
#: the /Public/colorectum subpath, hence the doubled "Public".
SLIDES_DIR = (
    "/mnt/data/Public/Public/colorectum/TCIA_HunCRC/"
    "PKG - Hungarian-Colorectal-Screening_v2"
)
#: The job's own output dir, per configs/preprocessing/qc.yaml:3 under the hunCRC
#: project_path. Only pass this to --out once the slide has succeeded elsewhere.
JOB_OUT = "/mnt/projects/huncrc_dysplasia/quality_control"
#: Scratch default: a failed attempt must not land where the job's 186 good masks are.
DEFAULT_OUT = "/mnt/projects/huncrc_dysplasia/qc_probe"


def service_up(base_url: str) -> bool:
    """Cheap reachability probe, so a 3.6 h wait is not spent on a dead service.

    Uses the openapi route rather than /health: the service is a FastAPI app behind Ray
    Serve that exposes PUT /, PUT /report and PUT /modify_and_reconstruct and has no
    /health at all (it answers that with a 404 in ~5 s on a cold replica, 10 ms warm).
    GET on anything unregistered proves the same thing for less money.

    Also worth knowing: the FIRST request after an idle period can take minutes while a
    Ray Serve replica spins up, which is why this retries rather than trusting one
    attempt. A 30s single-shot probe here once called a healthy service dead and
    cancelled a run before it started -- the exact misdiagnosis that cost the previous
    session days, so the retry budget (3 attempts, ~2 min apart) is load-bearing.
    """
    for attempt in range(1, 4):
        try:
            r = httpx.get(f"{base_url}/openapi.json", timeout=120, follow_redirects=True)
            if r.status_code == 200:
                if attempt > 1:
                    print(f"service answered on attempt {attempt} (was cold)")
                return True
            print(f"attempt {attempt}: HTTP {r.status_code}, not 200")
        except Exception as e:
            print(
                f"attempt {attempt}: {type(e).__name__}: {e} "
                "-- a cold replica can do this; retrying",
                flush=True,
            )
        if attempt < 3:
            time.sleep(120)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slide", default="011.mrxs", help="file name under the v2 dir")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output_path sent to service")
    parser.add_argument(
        "--timeout",
        type=int,
        default=18000,
        help="seconds to wait (one slide is ~3.6h; 18000 = qc.yaml's request_timeout)",
    )
    args = parser.parse_args()

    slide_path = Path(SLIDES_DIR) / args.slide
    if not slide_path.exists():
        sys.stderr.write(
            f"{slide_path} not visible from here. The QC service may still see it, but "
            "the notebook and the service mount the same rationai/Data, so a miss here "
            "means the path is wrong for both.\n"
        )
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.glob("*.tiff")) | set(out_dir.glob("*.csv"))

    #: Same defaults the job uses, so timings and masks are comparable
    #: (configs/preprocessing/qc.yaml:7-13).
    config = SlideCheckConfig(
        mask_level=3,
        sample_level=1,
        check_residual=True,
        check_folding=False,
        check_focus=True,
        wb_correction=True,
    )

    print(f"slide : {slide_path}")
    print(f"output: {out_dir}  ({len(before)} files before)")

    #: Check the service before spending ~3.6 h to find out it is unreachable.
    with Client() as probe:
        qc_url = probe.qc_base_url
        if not service_up(qc_url):
            sys.stderr.write(
                f"\nQC service at {qc_url} did not answer /openapi.json. Not sending "
                "the request -- a real check costs ~3.6 h and would fail the same "
                "way. Ask for the rayservice-qc pods before retrying.\n"
            )
            return 1
    print(f"service: {qc_url} is up")
    print("waiting for the service... (a whole-slide QC run is slow; up to "
          f"{args.timeout}s)\n", flush=True)

    started = time.time()
    with Client() as client:
        try:
            xopat_url = client.qc.check_slide(
                slide_path, out_dir, config=config, timeout=args.timeout
            )
        except Exception as e:
            elapsed = time.time() - started
            sys.stderr.write(
                f"\nFAILED after {elapsed:.0f}s: {type(e).__name__}: {e}\n\n"
                "Interpretation: a connect error means the service is unreachable from "
                "this network; an HTTP 4xx naming the path means the service cannot read "
                "the slide (missing rationai/Data mount); a 200 that produced no files "
                "means it cannot write output_path (missing rationai-Projects mount).\n"
            )
            return 1

    elapsed = time.time() - started
    after = set(out_dir.glob("*.tiff")) | set(out_dir.glob("*.csv"))
    new = sorted(after - before)
    print(f"done in {elapsed:.0f}s")
    print(f"new files in {out_dir}: {len(new)}")
    for path in new[:8]:
        print(f"  {path.name}")
    if not new:
        print(
            "\nHTTP succeeded but the service wrote NOTHING. That is the silent failure "
            "the job would also pass: the service cannot write this path. Ask for the QC "
            "pod's volume mounts and compare against this exact directory."
        )
        return 1

    print("\nxOpat link (open it to see the masks over the slide):\n  " + xopat_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
