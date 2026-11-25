# ReportFast field report — HunCRC smoke test, 2026-09-30

ReportFast 0.1.0 @ 810c09f, installed into an existing uv-managed `.venv` (py3.12),
skill installed at `~/.claude/skills/reportfast`.

Task: one test report, a few HunCRC MIRAX slides, background only, published to
MLflow run `45a957c5a79d4560b2b3efb68e19eed6` (experiment 168, public cluster),
viewer `xopat.rationai.cloud.trusted.e-infra.cz`. Outcome after ~2 hours: **working
report**, one slide (`001.mrxs`), opens correctly in xOpat.

Most of the time went into two failures, one of them mine and one of the
deployment/docs not matching the library's model of the world. Everything else is
small API friction that cost minutes each but piled up.

---

## 1. The two real failures

### 1.1 Explicit `"plugins": null` kills the viewer with a cryptic error — severity: high

`XopatSession.to_config()` always emits `plugins`. I overrode it to `None`
(thinking I was matching a working session — I wasn't; that session has **no
`plugins` key at all**), and the viewer died before loading anything:

> `TypeError: Cannot read properties of null (reading 'gui_annotations')`

JS semantics: an **absent** key gets the viewer's default; an explicit `null` is
read unguarded and throws. The session was otherwise byte-identical to a
confirmed-working one. Three related problems for the library:

1. **The skill's troubleshooting table doesn't have these error strings.**
   `xopat-v3.md` has a good debug checklist but none of the three failures I hit
   this session appears in it. Each maps 1:1 to a cause and each cost real time to
   find; all three are visible from the link alone, with no browser involved:
   - `Cannot read properties of null (reading 'gui_annotations')` → session has an
     explicit `plugins: null`.
   - `Cannot read properties of null (reading 'visualization')` → corrupted or
     truncated URL fragment (I produced one of these by hand-copying a 1.5 KB URL
     into chat — the reason §4's "publish the probe, never paste the link" loop
     matters).
   - **Bare `Error`, white screen, nothing in "details"** → the URL path, not the
     session: a `/` between the script and the fragment (`redirect.php/#`). See
     §3.3 for the measurement. This one is the worst of the three to debug,
     because the session is innocent and the payload is byte-identical to a
     working link — the only way to see it is to diff the URL *outside* the
     fragment, which nobody thinks to do when the payload is a 1.5 KB blob.
2. **`plugins: {}` is fine, `null` is not** (probe link B, §3.3) — so the empty
   dict `to_config()` materializes needs no change on correctness grounds.
3. **There is no way to *not* emit `plugins`.** Only a round-trip-fidelity
   argument left, since `{}` opens fine: `to_config()` hardcodes the five keys and
   `extra` can override `plugins` to anything including poison but not to
   *absent*, so a session that arrives via `from_url` does not leave byte-identical.
   Suggested fixes, either:
   - omit `plugins` from `to_config()` when `self.plugins` is empty, so
     `from_config(…).to_config()` reproduces the input bytes; or
   - keep emitting it but validate: raise if `extra["plugins"]` is set to a
     non-mapping. `extra` silently overlaying core keys let me inject my own
     poison — `extra` should probably refuse the five `SESSION_KEYS` or warn.
4. **The viewer's error UX is the actual interface here.** Nothing in a report
   author's workflow surfaces the console; "details" showing a minified
   `TypeError`, or nothing at all, is where we all end up. If ReportFast ships a
   validator (`reportfast check <link>`) it pays for itself immediately — and it
   has to check **both halves** of a link: the fragment against known viewer
   landmines (`plugins: null`, `dataReference` out of range, unregistered shader
   type), and the path before the `#` against the deployment's join rule. Every
   failure this session was decodable from the link string alone; none of them
   were decodable from the session JSON alone.

### 1.2 The deployment reality doesn't match `DEFAULT_BASE_URL` / the v2→v3 story — severity: high (docs), medium (library)

What worked, byte for byte, is the **legacy v2 entry point**:

```
https://xopat.rationai.cloud.trusted.e-infra.cz/xopat/redirect.php#<urlencoded json>
```

with the **v2 deprecated inline protocol** on the background:

```json
"protocol": "`/wsi-service/v3/files/info?paths=<DataID>`"
```

while the library and skill say:

- `DEFAULT_BASE_URL = …/v3/` and "the v2 `redirect.php` relay is gone"
  (`viewer_url` docstring, `hydra-v2-to-v3.md`: "`redirect.php` deleted upstream");
- "`background[].protocol` is v2 and deprecated", and inline-JS templates are
  "refused outright … as an RCE sink" — `xopat.py` **raises** on a backtick.

Both statements are wrong *for this deployment*, which is the one the reference
report and my working report both use:

- `redirect.php` is not gone; it is what the known-good BRACS report links, and it
  correctly boots a v3 session.
- The backtick protocol template is not refused; it is **load-bearing**. Proven by
  A/B: same session verbatim from the working report → slide + overlay renders;
  same session with `protocol` popped → background blank, overlay still draws.
  Without it the viewer never resolves the slide.

Consequences:

- The security check is right in principle and I'd keep it, but the escape hatch
  is "mutate `session.background` by hand", which is exactly how you end up with a
  report that differs from the reference in one accidental byte. A deliberate
  opt-in — `XopatEndpoint(image_protocol="…")` is for *registered names* only;
  something like `XopatSession(..., inline_protocol_ok=True)` or
  `background_protocol_template=…` on the endpoint — would let the library emit
  this session shape honestly. (Per the skill, v3's blessed form is `protocol` on
  the `data[]` entry — **we never got that variant to work or fail**, because the
  failures in §1.1 and §2 dominated. Worth a test: if `data[0].protocol` works on
  this deployment, the library's defaults are fine and only the docs are stale.)
- `deployment.md` and `xopat-v3.md` should describe both mounts as live until
  someone proves otherwise, with the probe command to tell them apart. The
  "v2 parses a v3 session and looks like…" warning sent me *away* from the mount
  that actually works here.
- Unverified question that would settle the docs: does `…/v3/#<json>` work on
  this host at all? I never tested it because every attempt died on §1.1 first.

## 2. DataID discovery — the missing chapter

Nothing in the library or skill tells you **what DataID a given slide has on a
given deployment**, and every failure of the session kind bottoms out in that
string. This session I burned many rounds on candidates (`/PKG folder` vs not,
`%20` vs raw space, "the slide IS registered") before one cheap experiment
settled it: a published HTML page of `<img>` tags hitting
`/v3/slides/thumbnail/max_size/…?slide_id=<candidate>` — rendered *in the user's
browser*, because the notebook pod cannot reach the xOpat hosts at all
(every curl from the pod is a connection timeout; the skill's
`curl … /v3/slides/info` probe silently proves nothing from that environment).

Learned the hard way, worth ten lines in `deployment.md`:

- DataID = path relative to `mount_root`, **including** whatever top folder the
  tile-server mount exposes (`PKG - Hungarian-Colorectal-Screening_v2/…` here —
  `mount_path()` strips `/mnt` and nothing else knows the rest).
- DataIDs carry **raw spaces**; the payload gets percent-encoded exactly once on
  the way into the URL. Writing `%20` in a DataID ships as `%2520` and addresses
  a path that does not exist. No error anywhere says this; the tile server just
  serves nothing.
- A thumbnail probe page distinguishes "tile server has no idea about this path"
  from "viewer session broken" in one click, without opening a devtools console.
  It was the single most useful thing built this session and it's ~20 lines on
  top of `thumbnail_url()`. **Suggest: `reportfast probe <slides.yaml> --publish`
  — a grid of candidate DataIDs × thumbnail, per endpoint.** That's the tool I
  hand-rolled at 3am.

## 3. API friction (each small, all real)

1. **`SlideGrid([SlideCard(...)])` dies inside pathlib.** `as_session()` falls
   through to `from_slide(card_object)` → `TypeError: argument should be a str or
   an os.PathLike … not 'SlideCard'` from `pathlib`. The grid's whole job is
   wrapping sessions in cards; handing it a card is a natural mistake. Either
   pass `SlideCard`s through, or `raise TypeError("SlideGrid takes sessions
   (or paths); card options go in card={…}")`.
2. **`Report.render()` doesn't exist** — it's `to_html()` / `write()`. Every
   component in the library has `.render()`; I guessed `.render()` on the one
   object that doesn't. Alias it, or rename for consistency.
3. **`viewer_url()` forces `rstrip('/') + '/#'` — measured: this one is fatal.** A
   configured base of `…/redirect.php` ships as `…/redirect.php/#<json>`, while
   every known-good link on this deployment is `…/redirect.php#<json>`. My report
   built through `SlideGrid` white-screened with a bare "Error"; the hand-built
   page of the same sessions opened fine. `probe/mask_link_probe.html` isolated it
   with four links varying one thing each:

   | link | result |
   |---|---|
   | A `redirect.php#` + heatmap layers | opens |
   | B A + explicit `"plugins": {}` | opens |
   | C `redirect.php/#`, payload identical to A | **white screen** |
   | D A with no shader layers | opens |

   So the trailing slash before the fragment kills it (most likely: the app
   resolves its relative assets one level too deep), while `plugins: {}` — the
   other delta `to_config()` imposes — is harmless. I had written "probably
   harmless (the fragment is all the script reads)" in the first draft of this
   file. Wrong, and the reasoning was wrong too: the fragment may be all the
   *script* reads, but the URL path is what the browser resolves relative
   resources against. `base_url` should be joined literally — if I configured a
   path, I meant that path — and `XopatEndpoint` could reject a `base_url` that
   does not end in `/` when the caller did not write one… no: accept it verbatim,
   exactly as configured, and document that `redirect.php` takes no separator.
   Two more findings from the same probe, both good news for the library:
   heatmap layers over `.mrxs` backgrounds render, and reconstructing
   `to_config()`'s output is byte-safe apart from `plugins`.
4. **Round-trip honesty.** The natural move on a new deployment is: take a working
   link, `XopatSession.from_url(link)`, change the slide, re-emit. `from_url`
   exists and is documented — good — but I didn't find it and regex-parsed the
   report HTML by hand, because the skill's recipes never walk that path. It's
   *the* golden workflow for a new deployment (§1.2's whole problem disappears
   under it). Make it the first recipe in `SKILL.md`, and note the byte-level
   deltas it currently introduces (`plugins` materializes to `{}`, key order,
   the `/` in the join) so authors know what "equivalent" costs.
5. **`publish()` prints the run URL, not the artifact URL.** `Published.__str__`
   is `one/one_slide.html on run 45a95… -- …/#/experiments/168/runs/45a95…`, so
   the reader gets a relative path plus a link to the run and has to find the
   artifact tab and the file themselves. Most of this session's loop was "publish
   a probe, click it", and that navigation happened every single round.

   The direct form exists and works — verified on the public cluster:

   ```
   {MLFLOW_WEB_URL}/api/2.0/mlflow-artifacts/artifacts/{exp}/{run}/artifacts/{path}
   ```

   serves the HTML as `text/html` (200, correct bytes) to a request carrying
   credentials. Caveat worth encoding in whatever emits it: unauthenticated, that
   same URL answers **200 with the OIDC login page** rather than a 401 — so it
   works in a browser that already holds the login cookie, and silently shows a
   login page otherwise, never a clear error. `Published.artifact_url` should
   probably be built from `experiment_id` (which `publish()` has to fetch
   anyway) rather than the artifact URI string, and documented as
   login-session-only.

   Related: the stored path is `mlflow-artifacts:/168/…/artifacts/one/one_slide.html`,
   which looks wrong next to the S3-style URIs from previous clusters but is
   correct — the experiment's `artifact_location` is `mlflow-artifacts:/168`, so
   artifacts are proxied through the tracking server. `mlflow.md` could say so,
   because the `[mlflow]` extra hinting at boto3 makes the healthy config look
   broken.
6. **`thumbnail_url()` percent-encodes the DataID's slashes (`%2F`)** — my
   hand-built probes used raw slashes. Both render, so this is a question, not a
   complaint: pick one deliberately and document it, because the first thing
   anyone does when a thumbnail is blank is diff these URLs character by
   character.
7. **`endpoint` is threaded by hand and its absence is silent.** An `ENDPOINT`
   built at the top of the script and not passed to `SlideGrid` made every card
   link fall back to `DEFAULT_BASE_URL` (`…/v3/`) with no warning — and the
   symptom was indistinguishable from a wrong DataID, which sent debugging a
   full loop around. The session carries its own `endpoint` and `SlideCard`
   takes one and `SlideGrid` takes one; three places to remember, one to get
   wrong. `XopatSession.url(endpoint)` could prefer the session's own endpoint
   when set (it does fall back correctly — the bug was never passing it, but a
   session that knows its endpoint should win over a component that doesn't),
   or `from_env()` results being *required* somewhere enforceable.
8. **Mask/Inventory type wall (early detour).** Trying to attach masks with
   `Mask(background=…)` raised `ValueError: background= cannot be a Inventory`
   — the message says *what* is wrong but not the fix (the background object
   must subclass `MaskSource`; discovered via source read, not via the message
   or the skill). The task then turned out to need no masks at all, so this is
   second-hand — but a message ending in "…must be a MaskSource (see
   masks.md)" would have been the whole fix.

Also, environment setup friction the **README/skill install** path should absorb:
the README's skill-install snippet (`cp -r "$sp/skill" …` pattern) failed as
written — `dirname` applied twice resolved to the wrong directory; needed one
`dirname`. And `reportfast skill show` working correctly afterward is evidence the
CLI side is fine.

## 4. What worked well — keep

- **`Mlflow.publish()`** did exactly the right thing first try, on a cluster with
  OIDC auth, printing a clickable URL — and publishing probe pages as artifacts
  was what finally made the debugging loop work (user clicks a link instead of
  copy-pasting a 1.5 KB URL, which I had corrupted by hand twice). If publishing
  is cheap, make "publish the probe, don't paste the link" the recommended loop
  everywhere in the docs.
- `as_session()` accepting a pasted URL / raw JSON / mapping / path is a genuinely
  good union; the only miss was not type-checking `SlideCard` (§3.1).
- `from_config()`'s hard validation of `dataReference` range is exactly the right
  check to keep — it's arithmetic, not tribal knowledge.
- The `background` list as a documented "reach any field we don't model" surface
  is what made §1.2 survivable at all. Keep that property while adding
  first-class support.
- `sessionName`-as-cache-key, `dataReference` singular vs `dataReferences`
  plural, the positional-pool warning: the skill encodes the real landmines.
  What it lacked was the two error strings in §1.1 and the DataID chapter in §2.

## 5. Connecting to MLflow — what it actually took

Roughly the first hour of the session went into this, none of it ReportFast's fault
directly, but ReportFast is the thing being pointed at so the library absorbs the
confusion. The working configuration, in full:

```bash
export MLFLOW_TRACKING_URI=https://mlflow.rationai.cloud.e-infra.cz   # public cluster
export MLFLOW_TRACKING_USERNAME=<full email, e.g. name@domain>        # not the short username
export MLFLOW_TRACKING_PASSWORD=<access key, NOT the SSO login password>
export MLFLOW_WEB_URL=https://mlflow.rationai.cloud.e-infra.cz        # see 5.2
```

Worth a "public cluster in five minutes" box in `mlflow.md`, because each line was
found by experiment and none is derivable from the docs:

### 5.1 The OIDC plugin answers unauthenticated API calls with **200 + a login page**

`mlflow-oidc-auth` serves its HTML login page at HTTP **200** for anonymous
requests — not 401, not `WWW-Authenticate`. Consequences that cost real time:

- The mlflow client's failure on an unauthenticated call is a JSON parse error on
  `<!DOCTYPE html>`, not an auth error. My first connection probe looked for the
  401 that never comes.
- "Wrong username" (short username instead of full email) and "wrong password"
  (login password instead of an access key minted under Permission → create access
  key) are invisible: also 200 + login page. There is no wrong-credential signal
  at all until something tries to parse the body.

**Suggest: `Mlflow.check()` / `reportfast doctor`** — one round trip
(`/api/2.0/mlflow/experiments/get?experiment_id=0` is enough), classifying by
Content-Type: JSON → authenticated; HTML → "not logged in, check
`MLFLOW_TRACKING_USERNAME`/`PASSWORD`"; refused → "this host is only resolvable
in-cluster". It must detect auth by content type, not status code — that insight
is the whole tool, and it took a manual curl to learn. The skill says credentials
come from the environment (right rule, and I followed it — including refusing to
write them to a file); it just has no way to *verify* them.

### 5.2 The defaults point at the other cluster

`DEFAULT_WEB_URL` is the **trusted** cluster's UI. From the public cluster, unset
`MLFLOW_WEB_URL` means `publish()` uploads correctly but prints run URLs on a
server that does not know the run — the docstring for `from_env` actually warns
about exactly this, yet the failure mode is a silently wrong link, not an error.
Options: same note in `mlflow.md`'s quickstart (cheapest), or `publish()`
comparing the run's tracking host against `web_url`'s host once and warning on
mismatch.

### 5.3 `DEFAULT_ARTIFACT_PREFIX = "mflow"` does not match this deployment

`artifact_data_id()` builds `mflow/<exp>/<run>/artifacts/<path>`, but the working
sessions on the public cluster address artifacts as
**`/public_mlflow/<exp>/<run>/artifacts/<path>`** — different prefix *and* a
leading slash. Checked against the reference report's mask DataIDs:

| `artifact_prefix` | `artifact_data_id(...)` reproduces the reference DataID? |
|---|---|
| `mflow` (default) | no |
| `/public_mlflow` | **yes, byte-identical** |
| `public_mlflow` (no slash) | no — the leading slash matters |

So `REPORTFAST_MLFLOW_ARTIFACT_PREFIX=/public_mlflow` is the correct setting on
the public cluster, and the function's composition is fine — it's only the
default that is cluster-specific. Still untested end-to-end (no artifacts-as-layers
report was made here), but the string equality is exact. Nothing in the library or
skill says the prefix is per-cluster, and how to read one off a working session is
exactly the kind of thing that only gets discovered at 2am on mask day —
`deployment.md` should tabulate per-cluster prefixes.

### 5.4 The in-cluster URIs are real but unreachable from notebooks

The wiki documents `http://mlflow.rationai-mlflow:5000` style URIs; those resolve
only inside kube-jobs (the notebook pod sits in a different namespace and hangs on
`getaddrinfo` for minutes trying). Not a ReportFast problem, but `deployment.md`
gaining "from a notebook, always the public URL; the short names are for jobs"
would stop the next person waiting on DNS.

### 5.5 Out of scope but recorded

- `uv add report-fast[mlflow]` hit an unrelated `rationai-kube-jobs` ssh dep
  (host-key verification); `uv pip install --python .venv/bin/python` was fine.
  Also note `uv pip` defaults to `$VIRTUAL_ENV` (system venv) — it silently
  upgraded that env's mlflow until pinned back; worth a line in the README's
  install section.
- The `[mlflow]` extra hinting at boto3 is a red herring when the run's
  `artifact_location` is `mlflow-artifacts:/…` (proxied) — nothing needs boto3.
- Credentials hygiene went right by skill design: the "never write a token to a
  file, never echo a value" rules held up under pressure (and one `.env` write
  attempt was blocked, correctly). The skill's rule "a token pasted into chat is a
  rotated token" also fired in practice — the user's token did get pasted and must
  now be rotated; keeping that line blunt was right.

## 6. Priority wishlist

Ordered by measured damage this session, not by how elegant the fix is.

1. **Join `base_url` literally in `viewer_url()` — no inserted `/` (§3.3).** One
   character, and it white-screens the viewer. Everything else here cost time;
   this cost a working report.
2. **Never emit `plugins: null`, and reject it if a caller asks for it (§1.1).**
   `{}` opens fine (probe B), so this is specifically about `null` — which
   `extra["plugins"] = None` currently lets anyone write, silently.
3. **Troubleshooting table with all three error strings (§1.1)**, including the
   bare-`Error` white screen whose cause is in the URL *path*, not the session —
   the counter-intuitive one.
4. **A publishable DataID/thumbnail probe in the library (§2)** — the discovery
   tool this stack is missing, and the only one that works when the pod can't
   reach the viewer host. Both decisive experiments this session were hand-rolled
   publish-and-click pages; that's ~20 lines the library could own.
5. **`reportfast doctor` / `check` (§5.1, §1.1)** — validate auth by Content-Type,
   and validate a link in both halves: fragment against viewer landmines, path
   against the deployment's join rule.
6. Fix the v2/v3 mount story in `deployment.md` / `hydra-v2-to-v3.md` against a
   live public-cluster deployment; test `data[0].protocol` (v3 form) on the
   trusted cluster and write down which of the four combinations
   (`/v3/` vs `redirect.php`) × (inline vs registered vs data-entry protocol)
   actually work where.
7. `SlideGrid` card pass-through (or a real error); `Report.render()` alias;
   `Published.artifact_url` (§3.5).
8. Per-cluster table of tracking URI / web URL / artifact prefix (§5.2, §5.3) —
   the "first connection" hour of every new deployment is this, and it isn't
   documented in one place.

---

# Session 2, 2026-10-26 — 186-slide tile-mask report (same deployment)

Build: 186 cards, each session = .mrxs background + up to 8 mask layers,
hand-authored sessions (v2 redirect.php byte-shape from
`scripts/make_epi_viewer_link.py`) through `XopatSession.from_config` +
`SlideGrid.add(session, name=, note=)`. Library 0.1.0 still joins base with
`'/'` + `'#'` — the `html.replace(f"{V2}/#", f"{V2}#")` post-process from
session 1 is still required and still works.

## New findings

1. **The `mlflow-artifacts` proxy 500s on artifact filenames with two dots** —
   `report/tile_masks_report.v3.html` → 500 Internal Server Error, while
   `report/report.html` in the same dir is 200, same size class (~0.5 MB),
   same auth. Renaming to `tile_masks_report_v3.html` → 200. No error body
   detail (generic Werkzeug 500). If `publish()`/`write()` ever accepts a
   name like `report.v3.html`, the page lands and the URL 500s — a report
   that built fine and looks complete but won't open. Suggest: validate
   artifact filenames in `publish()` (single trailing dot) or warn on
   multi-dot names.
2. **`/v3/slides/thumbnail` only thumbnails real WSI slides.** A mask TIFF
   DataID that renders perfectly inside a session answers blank at the
   thumbnail endpoint. So thumbnail `<img>` probes cannot verify masks —
   only session links can. Session 1's probe page mixed both, and the blank
   mask thumbnail was easily misread as "route dead". Suggest: probe pages
   (the library could own these, §6.4) should render session links, not
   thumbnails, when the object under test is a mask.
3. **`/mnt/projects` is NOT served by this deployment's tile server.**
   `projects/<anything>` DataIDs (our dir and another project's dir, both
   real files) are blank; the golden `examples/dysplasia_case.json`
   `projects/…` paths belong to a project that doesn't exist on this
   cluster — the example is from the other deployment. Masks must be MLflow
   artifacts here (`/public_mlflow/…`), which is ~17 GB of egress for a
   186-slide × 8-mask report. Per-cluster "which roots does the tile server
   actually serve" belongs in `deployment.md` next to the artifact prefix.
4. **v3-form visualization needs `"order"`** (the golden example has it;
   my first v3 session without it rendered no overlays — A/B'd against the
   v2 shape on the same file). Not isolated as the sole delta, but the
   golden example is the only known-good v3 multi-shader session and it has
   it. `normalise_layer`/`add_visualization` should emit `order` from the
   shader insertion order so nobody has to know this.
5. **mlflow 3.12 `MlflowClient` takes no `username`/`password` kwargs** —
   auth is environment-only (`MLFLOW_TRACKING_USERNAME/PASSWORD`).
   `make_epi_viewer_link.py` still passes the kwargs and would crash on
   first use; check `report_fast/mlflow.py` against this too.

## What worked well

- `SlideGrid.add(session, name=…, note=…)` + `XopatEndpoint(base_url=V2)`
  end-to-end: 186 cards in ~20 s, `/`-fix included, page ~0.5 MB. The
  grid/card API is fine; session 1's card-pass-through issue didn't recur
  because sessions (not cards) were passed.
- `from_config` round-trip + raw `background[0]["protocol"]` patch survived
  all 186 sessions; no dataReference validation complaints from the
  positional pool.

## Session 3, 2026-10-27 — v3 overlay protocol (the one that actually cost a day)

The "no overlays in v3" bug was **not** session shape: five A/B variants
(heatmap vs colormap, params, layer count, no-protocol) all failed
identically while the background rendered. The cause is per-data-entry
`protocol` resolution in the v3 viewer:

1. v3 only accepts **registered** protocol names (inline templates are
   rejected as an RCE sink — `slide-protocols.ts`). Registered here:
   `iipimage` (default bg AND viz), `wsi_service`, `plain_image`,
   `virtual-region`.
2. `wsi_service` = WSI-Service slide endpoints. WSI-Service (all versions)
   is a plain filesystem lookup under one `data_dir` — it sees the `data/`
   share (mrxs) and 404s both `/mnt/projects` and `/public_mlflow`
   artifacts. It never had MLflow support; the v2 viewer showed masks
   because its legacy data-group route fetched files **raw**, a route v3's
   protocol system no longer uses.
3. `iipimage` serves the MLflow-artifact TIFFs. Working v3 session on this
   deployment: background data entry `protocol: "wsi_service"`, mask data
   entries `protocol: "iipimage"`, layers = the proven v2 heatmap shape
   (`use_channel0/color/opacity`).

Library/deployment notes:

- **The deployment's protocol registry is unreadable from the build pod**
  (pod can't reach the viewer host; the deployment env file is the only
  other source, and it's per-cluster). `xopat-v3.md` already says "no way
  to list them from a script" — but it's worth a worked recipe: the
  browser console, or the deployment env, is the only enumeration. The
  deployed build exposed no `window.SLIDE_PROTOCOLS` global; a deep
  read-only window scan for objects carrying `slide_protocols` found it.
- **Correction to Session 2 item 4:** the missing-`order` A/B was
  confounded — the real drop cause was the protocol, so `order`'s
  requirement stands only as "the golden example has it", not as an
  isolated finding.
- `colormap` with no `color`/`threshold` params renders the 0..255 field
  as 3 Viridis classes **including the value-0 class**, i.e. a full-slide
  paint that hides the background. For binary/continuous masks the v2
  `heatmap` shape (alpha = value) is the right default; a
  `colormap_layer` convenience that hides class 0 by default (or a
  `binary_mask=True`) would prevent this footgun.
