#!/usr/bin/env python3
"""Build a W3 PROV provenance chain for an MLflow run and export it as an image.

Given a run URL (or run id) on the MLflow server, the script:
  1. finds matching runs — same user, or runs sharing the same datasets
     (upstream artifact runs referenced by the run's parameters/tags),
  2. walks the chain upstream from the run until nothing resolves further
     (i.e. back to dataset creation),
  3. collects every parameter and every environment artifact (sha256 + size),
  4. writes the chain as W3 PROV:
       - provenance_<run_id>.prov.json   (CommonProvenanceModel JSON-LD)
       - <out-dir>/prov-<name>.png       (graph, rendered via prov.dot like the
                                          RationAI/crc_ml-provenance repo does:
                                          prov.dot.prov_to_dot(bundle).write_png)
       - <out-dir>/prov-<name>.provn     (PROV-N serialization)

Usage:
    python provenance/provenance.py [RUN_ID_OR_URL] [--out-dir DIR] [--env]

Default run: the training run of the 2-slide test chain (b8863188…, exp 212
provenance_huncrc_test). Default server: our public MLflow cluster.
Auth: export MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD first
(the client reads them from the env).
--env is OFF by default: it downloads + hashes each run's environment/ artifact
dir, and artifact *downloads* over the public API are flaky from the pod. The
chain / params / PNG do not need it.
Self-check (offline): python provenance/provenance.py --self-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import mlflow
import prov.dot
import prov.model
from mlflow.entities import ViewType


#: Our public MLflow cluster. The pod reaches it over the public URL + the
#: basic-auth key (MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD in the
#: env) — the injected in-cluster URI does not resolve from here.
DEFAULT_TRACK_URI = "https://mlflow.rationai.cloud.e-infra.cz"
#: Default root: the end of the 2-slide provenance test chain — the training
#: run in exp 212 (provenance_huncrc_test). Pass another run id/URL to re-root.
DEFAULT_URL = "b88631886d0d471d89fbfbc7a45a2ea1"

# artifact URIs look like mlflow-artifacts:/<exp_id>/<run_id>/artifacts/<path>
URI_RE = re.compile(r"mlflow-artifacts:/(\d+)/([0-9a-f]{32})/artifacts?/?(.*)")
# the lib's input_uris tag links upstream runs as runs:/<run_id> — that is how
# OUR chain is connected (log_provenance writes runs:/ URIs, not artifact
# URIs), so the walk must resolve both conventions.
RUNS_RE = re.compile(r"runs:/([0-9a-f]{32})")

PREFIX = {
    "storage": "http://localhost:8083/api/v1/documents/",
    "meta": "http://localhost:8083/api/v1/documents/meta/",
    "schema": "https://schema.org/",
    # REVISION-2026-10-20 #2 (revises R1): the CPM namespace has /ns/ in it.
    # Same value as the prefix map in rationai/mlkit/provenance/common.py.
    "cpm": "https://www.commonprovenancemodel.org/ns/cpm#",
    "blank": "https://openprovenance.org/blank/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    # REVISION-2026-10-20 #9 (closed R18): gen: is now an absolute IRI.
    "gen": "http://rationai.org/prov/gen#",
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "sosa": "http://www.w3.org/ns/sosa/",
    # R11 (D-c): params / environment payloads live as attributes under our own
    # namespace, not as minted prov:* terms. A local namespace for mlkit's own
    # extensions.
    "mlkit": "http://rationai.org/prov/mlkit#",
}

# namespaces registered on the prov document (same pattern as
# rationai/provenance/provenance.py::prepare_document in RationAI/crc_ml-provenance)
# Must stay a superset-equal of the prefixes actually used by the builders: the
# prov package raises on an unregistered prefix, so anything that appears in an
# identifier here (storage:, gen:, cpm:) has to be declared below too.
NAMESPACE_URIS = {
    # REVISION-2026-10-20 #10: the main activity id moved from blank: to
    # storage:, so storage must be registered here, not only in PREFIX.
    "storage": "http://localhost:8083/api/v1/documents/",
    "gen": "http://rationai.org/prov/gen#",
    "schema": "https://schema.org/",
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/",
    "sosa": "http://www.w3.org/ns/sosa/",
    "blank": "https://openprovenance.org/blank/",
    "meta": "http://localhost:8083/api/v1/documents/meta/",
    "cpm": "https://www.commonprovenancemodel.org/ns/cpm#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    # R11 (D-c): matches PREFIX.
    "mlkit": "http://rationai.org/prov/mlkit#",
}

# The two mandatory cpm:mainActivity attributes.
#   referencedMetaBundleSpecV = the ISO/IEC 23494 provenance standard
#     (REVISION-2026-10-20 #3, value exactly as given by Rudolf; closes D-e).
#   referencedMetaBundleId is still a placeholder: the bundle's own meta
#     entity was removed (REVISION-2026-10-20 #13), so the real value comes
#     later from the CPM store (R5 — a full IRI typed xsd:anyURI, not a
#     qualified name).
CPM_PLACEHOLDERS = {
    "cpm:referencedMetaBundleId": "TODO:referencedMetaBundleId",
    "cpm:referencedMetaBundleSpecV": "ISO-IS-23494-2:2026",
}

# CPM backbone terms, spelled exactly as the CPM namespace defines them. Checked
# against the published vocabulary and the backbone template:
#   https://www.commonprovenancemodel.org/cpm-namespace/
#   https://www.commonprovenancemodel.org/cpm-backbone-template-v1/
# (this closed PROV_TODO.md T9 for the terms used here)
#
# Backbone as far as a chain assembled from MLflow runs uses it:
#   mainActivity  used             currentConnector    (what enters the process)
#   mainActivity  wasGeneratedBy   forwardConnector    (what leaves the process)
#   forwardConnector  wasDerivedFrom  currentConnector (only when affected by it)
#   dct:hasPart on mainActivity lists the sub-activities it covers
# No backwardConnector / receiptActivity here: the receiving step lives outside
# mlkit, so only the sending half of the backbone exists in this bundle.
# The "Specialized Forward Connector" mentioned in the meeting is a different
# object and is deliberately not emitted (see forward_connector_id).
# REVISION-2026-10-20 #1: CPM type names are PascalCase, as the CPM namespace
# defines them — the camelCase spellings were the only non-conforming types.
CPM_CURRENT_CONNECTOR = "cpm:CurrentConnector"
CPM_FORWARD_CONNECTOR = "cpm:ForwardConnector"
# R3: the organisation that produced this bundle — required by CPM (KB B.6, F.3)
# as a cpm:CurrentAgent, distinct from the individual `gen:user_*` agents.
CPM_CURRENT_AGENT = "cpm:CurrentAgent"

# Type of the trained-model entity. `prov:Entity` is the built-in fallback so a
# consumer that only resolves PROV types still sees an entity;
# `schema:CreativeWork` is the declared type — the `prefix` block already ships
# schema.org, and no term here (unlike mlflow's own `mlflow.models.Model` class
# name, or an MLmodel flavor such as `mlflow.pyfunc`) implies a serialization
# format this repo does not pin down.
MODEL_TYPES = ("prov:Entity", "schema:CreativeWork")

# Artifact path fragments that mark a trained model, most specific first.
# `mlmodel` matches MLflow's own python_function flavor marker (a real logged
# model, not a guess); a trailing `/` matches a flavor *directory*, which is how
# MLflow logs models. Lowercased substring match — the leading dots keep ".pt"
# from matching "plot" and friends.
MODEL_ARTIFACT_HINTS = (
    "mlmodel",
    "model/",
    "weights",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".h5",
    "model",
)
# Run names that mean "this run fits a model". Covers a training run that died
# before logging a model artifact, so T6's entity still exists and points at
# artifact_uri instead of vanishing from the graph.
MODEL_RUN_NAME_HINTS = ("train", "fit")


def main_activity_props(has_parts: list[str]) -> dict[str, list[Any]]:
    """Properties of cpm:MainActivity, shared by the JSON and diagram builders.

    Single source for the main activity's *attributes*. The relations that
    attach it to the chain (used / wasGeneratedBy on the connectors) are emitted
    by connect_backbone() / the equivalent prov calls, because the JSON and prov
    paths express a relation through different objects.
    """
    props: dict[str, list[Any]] = {
        "prov:type": [_qn("cpm:MainActivity")],
        # REVISION-2026-10-20 #13: the meta-bundle entity was removed, so this
        # stays the placeholder until the CPM store assigns the real id (R5).
        "cpm:referencedMetaBundleId": [
            CPM_PLACEHOLDERS["cpm:referencedMetaBundleId"]
        ],
        # R4: correct attribute name per KB was referencedMetaspectVersion.
        "cpm:referencedMetaBundleSpecV": [
            CPM_PLACEHOLDERS["cpm:referencedMetaBundleSpecV"]
        ],
    }
    if has_parts:
        props["dct:hasPart"] = [_qn(part) for part in has_parts]
    return props


def main_activity_id(root_id: str) -> str:
    """The one construction site for the main activity identifier.

    REVISION-2026-10-20 #10: storage:, not blank: — the main activity is the
    traversal anchor and must be addressable in the CPM store.
    """
    return f"storage:Run_{root_id[:10]}"


def model_entity_id(run_id: str) -> str:
    """The trained model produced by a run (CPM: the chain's output)."""
    return f"gen:model_{run_id}"


def forward_connector_id(run_id: str) -> str:
    """CPM forwardConnector of a run's main activity.

    Named after the activity it exits from, matching model_entity_id /
    main_activity_id. Note the *entity* identifier is ours to choose; only the
    `cpm:forwardConnector` type is fixed by the standard — and that type is what
    makes this distinguishable from the separate "Specialized Forward Connector"
    object, which we must not emit.
    """
    return f"gen:forward_{run_id}"


def current_connector_id(artifact_uri: str) -> str:
    """CPM currentConnector: what enters the chain from outside it.

    Keyed by artifact URI, not run id, mirroring gen:input_ — the same external
    artifact entering two runs of the chain is one connector, used twice.
    """
    return f"gen:current_{slug(artifact_uri)}"


def input_entity_id(artifact_uri: str) -> str:
    """The domain entity standing in for a consumed artifact."""
    return f"gen:input_{slug(artifact_uri)}"


def chain_input_uris(chain: dict[str, dict[str, Any]]) -> list[str]:
    """Artifact URIs consumed by the chain but produced outside it.

    These are the chain's input boundary — everything else consumed inside the
    chain already has a producer node. De-duplicated, first-seen order kept so
    the output is stable across runs of the script.
    """
    out: list[str] = []
    seen: set[str] = set()
    for info in chain.values():
        for _exp, up_id, uri in info["inputs"]:
            if up_id not in chain and uri not in seen:
                seen.add(uri)
                out.append(uri)
    return out


def resolve_output(
    p: Probe, chain: dict[str, dict[str, Any]], root_id: str
) -> tuple[bool, str, str]:
    """Decide what the chain as a whole produces: (is_model, uri, how).

    A chain end whose run is training produces a trained model, narrowed to the
    model artifact; anything else keeps pointing at the run's artifact root so
    the forward connector is never mislabelled as a model.
    """
    root = chain[root_id]
    artifact_uri = root["run"].info.artifact_uri
    input_uris = [u for _, _, u in root["inputs"]]
    if is_model_run(root["name"], input_uris):
        uri, how = model_artifact_uri(p, root_id, artifact_uri)
        return True, uri, how
    return (
        False,
        artifact_uri,
        "run artifact_uri (chain end is not a model run)",
    )


def output_id(run_id: str, is_model: bool) -> str:
    """Identifier of a run's output entity — one node per real thing.

    A model-typed output gets its own `gen:model_` handle so MOU matching (and a
    reader of the graph) can tell the trained model apart from dataset outputs
    without inspecting types.
    """
    return model_entity_id(run_id) if is_model else f"gen:output_{run_id}"


def output_props(
    name: str, uri: str, is_model: bool, how: str = ""
) -> dict[str, list[Any]]:
    """Entity properties of a run's output: dataset sample or trained model.

    `how` records which signal identified the model artifact, so a wrong guess
    is visible in the document instead of having to be re-derived from code.
    """
    if is_model:
        return {
            "prov:type": [_qn(t) for t in MODEL_TYPES],
            "schema:name": [f"model of {name}"],
            "schema:url": [uri],
            # REVISION-2026-10-20 #12: the MLflow artifact URI is the value by
            # which the object can be located ("the best and only option").
            "cpm:externalId": [uri],
            "cpm:externalIdType": ["MLflow artifact URI"],
            "dct:description": [
                f"Trained model produced by {name}" + (f" ({how})" if how else "")
            ],
        }
    return {
        "schema:name": [name],
        # REVISION-2026-10-20 #4: a run's output is its artifact *folder* —
        # a prov:Collection of the files inside (membership edges deferred
        # together with #5).
        "prov:type": [_qn("prov:Collection")],
        "dct:description": [f"Output of {name}"],
        "schema:url": [uri],
        # REVISION-2026-10-20 #12: as on the model branch.
        "cpm:externalId": [uri],
        "cpm:externalIdType": ["MLflow artifact URI"],
    }


def is_model_run(name: str, input_uris: list[str]) -> bool:
    """Whether a run's *output* is a trained model rather than a dataset.

    Two signals, either sufficient: the run consumes a model artifact (so it is
    downstream of a training run — fine-tuning, evaluation, inference), or its
    name reads as training. Only used on the root run, which is the end of the
    chain the meeting wants referenced; a wrong guess mislabels an entity, it
    does not break the graph.
    """
    lowered = name.lower()
    return any(
        h in uri.lower() for uri in input_uris for h in MODEL_ARTIFACT_HINTS
    ) or (any(h in lowered for h in MODEL_RUN_NAME_HINTS))


def model_artifact_uri(p: Probe, run_id: str, fallback: str) -> tuple[str, str]:
    """Narrow the run's artifact root down to the model itself.

    Returns (uri, how) where `how` records which signal matched, so the entity
    can say how confident it is. Falls back to the run's artifact_uri when the
    listing is empty or nothing looks like a model — the entity then points at
    the whole artifact directory, which is still the run's output.
    """
    try:
        paths = [pth for pth, is_dir, _ in p.walk_artifacts(run_id) if not is_dir]
    except (mlflow.exceptions.MlflowException, OSError) as e:
        # a run whose artifact store is gone is normal on the old server
        logging.getLogger(__name__).debug(
            "artifact listing failed for %s: %s", run_id, e
        )
        paths = []
    lowered = [(pth, pth.lower()) for pth in sorted(paths)]
    for hint in MODEL_ARTIFACT_HINTS:
        for pth, low in lowered:
            if hint in low:
                base = run_base_uri(fallback)
                return f"{base}/artifacts/{pth}", f"artifact hint {hint!r}"
    return fallback, "run artifact_uri (no model artifact identified)"


def run_base_uri(artifact_uri: str) -> str:
    """Strip the trailing /artifacts[...] suffix from an MLflow artifact URI."""
    return re.sub(r"/artifacts?/?.*$", "", artifact_uri.rstrip("/"))


# --- small utilities --------------------------------------------------------


def iso(ms: int | None) -> str:
    """Ms epoch -> ISO-8601 UTC (PROV xsd:dateTime)."""
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:100]


def _qn(name: str) -> dict:
    """A PROV qualified name: {"type": "prov:QUALIFIED_NAME", "$": name}."""
    return {"type": "prov:QUALIFIED_NAME", "$": name}


# --- helpers for talking to MLflow from Python -----------------------------


class Probe:
    """Thin wrapper around MlflowClient with the queries this script needs."""

    def __init__(self, track_uri: str):
        mlflow.set_tracking_uri(track_uri)
        self.client = mlflow.MlflowClient()
        self._cache: dict[str, object] = {}

    # runs
    def get_run(self, run_id: str):
        if run_id not in self._cache:
            self._cache[run_id] = self.client.get_run(run_id)
        return self._cache[run_id]

    def has_run(self, run_id: str) -> bool:
        try:
            self.get_run(run_id)
            return True
        except mlflow.exceptions.MlflowException:
            return False

    def name(self, run_id: str) -> str:
        tags = self.get_run(run_id).data.tags or {}
        return tags.get("mlflow.runName", run_id[:8])

    # artifacts
    def artifacts(self, run_id: str, path: str | None = None):
        return self.client.list_artifacts(run_id, path=path)

    def walk_artifacts(self, run_id: str, path: str = ""):
        """Recursive list of (path, is_dir, size).

        `a.path` is already the full path from the artifact root, NOT a child
        name, so it must never be joined onto `path`. Joining produced
        `environment/environment/uv.lock` — which then failed to download — and,
        worse, made every nested subtree vanish: `checkpoints/<epoch>/MLmodel`
        disappeared from the listing entirely, so the trained model looked like
        it did not exist and the model entity silently degraded to the bare
        artifact directory.
        """
        out = []
        for a in self.client.list_artifacts(run_id, path=path or None):
            out.append((a.path, a.is_dir, a.file_size))
            if a.is_dir:
                out.extend(self.walk_artifacts(run_id, a.path))
        return out

    def download(self, run_id: str, path: str) -> bytes:
        with tempfile.TemporaryDirectory() as d:
            local = self.client.download_artifacts(run_id, path, d)
            return Path(local).read_bytes()

    # provenance-ish queries
    def input_refs(self, run_id: str) -> list[tuple[str, str, str]]:
        """Every upstream (exp_id, run_id, full_uri) referenced by params/tags.

        Two reference conventions are resolved:
          * ``mlflow-artifacts:/<exp>/<run>/artifacts/<path>`` (the old mlkit
            style the script was originally written against), and
          * ``runs:/<run_id>`` — the lib's ``input_uris`` convention, which is
            how OUR chain is actually linked (every cut-down pins its ancestors
            as ``runs:/`` URIs, not artifact URIs).

        A ``runs:`` reference is resolved to the referenced run's ``artifact_uri``
        so the input entity points at a real, locatable artifact (the same string
        the upstream activity declares as its ``cpm:externalId``). If the
        referenced run is gone/unreachable the raw ``runs:/`` URI is kept and the
        exp id is ``"?"`` — such a reference then becomes an external-input
        boundary (a currentConnector) rather than a chain edge.
        """
        run = self.get_run(run_id)
        found: list[tuple[str, str, str]] = []

        def resolve_runs_ref(up: str) -> tuple[str, str]:
            try:
                target = self.get_run(up)
                return str(target.info.experiment_id), target.info.artifact_uri
            except mlflow.exceptions.MlflowException:
                return "?", f"runs:/{up}"

        def scan(v):
            if isinstance(v, str):
                for m in URI_RE.finditer(v):
                    uri = (
                        f"mlflow-artifacts:/{m.group(1)}/{m.group(2)}"
                        f"/artifacts/{m.group(3).strip('/')}"
                    )
                    found.append((m.group(1), m.group(2), uri))
                for m in RUNS_RE.finditer(v):
                    exp, uri = resolve_runs_ref(m.group(1))
                    found.append((exp, m.group(1), uri))
            elif isinstance(v, (list, tuple)):
                for x in v:
                    scan(x)
            elif isinstance(v, dict):
                for x in v.values():
                    scan(x)

        scan(run.data.params)
        scan(run.data.tags)
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str, str]] = []
        for e, r, u in found:
            if (e, r) not in seen:
                seen.add((e, r))
                out.append((e, r, u))
        return out

    def matching_runs(self, run_id: str) -> dict:
        """Runs related to run_id by creator or by shared datasets.

        Returns {'same_user': [...], 'shared_dataset': [...]} where each
        entry is (run_id, name). 'shared_dataset' = runs that consumed an
        artifact of the same upstream run this run consumed (i.e. they were
        fed by the same dataset-producing run).
        """
        run = self.get_run(run_id)
        my_upstreams = {r for _, r, _ in self.input_refs(run_id)}
        same_user: list[tuple[str, str]] = []
        shared_dataset: list[tuple[str, str]] = []
        for r in self.client.search_runs(
            experiment_ids=[run.info.experiment_id], run_view_type=ViewType.ALL
        ):
            rid = r.info.run_id
            if rid == run_id:
                continue
            nm = (r.data.tags or {}).get("mlflow.runName", rid[:8])
            if r.info.user_id == run.info.user_id:
                same_user.append((rid, nm))
            try:
                refs = self.input_refs(rid)
            except mlflow.exceptions.MlflowException:
                refs = []  # summary only — one unreadable run must not abort it
            if my_upstreams & {x[1] for x in refs}:
                shared_dataset.append((rid, nm))
        return {"same_user": same_user, "shared_dataset": shared_dataset}


# --- chain -------------------------------------------------------------------


def build_chain(p: Probe, root_id: str) -> dict[str, dict]:
    """Walk upstream from root_id through artifact references.

    Stops when every referenced run either exists on the server or is gone (old
    store). Returns {run_id: {'run', 'name', 'user',
    'inputs': [(exp, up, uri)]}} in discovery order (root first).
    """
    chain: dict[str, dict] = {}
    queue = [root_id]
    while queue:
        rid = queue.pop(0)
        if rid in chain or not p.has_run(rid):
            continue
        run = p.get_run(rid)
        chain[rid] = {
            "run": run,
            "name": p.name(rid),
            "user": run.info.user_id,
            "inputs": p.input_refs(rid),
        }
        queue.extend(up for _, up, _ in chain[rid]["inputs"] if up not in chain)
    return chain


def environment_info(p: Probe, run_id: str) -> dict | None:
    """Hash and size every file in the run's environment/ artifact dir.

    Returns {artifact_path: {'sha256':..., 'size':...}}, or None when the run
    has no such directory.
    """
    try:
        files = p.walk_artifacts(run_id, "environment")
    except mlflow.exceptions.MlflowException:
        files = []  # no environment/ dir (or store hiccup) -> nothing to hash
    env = {pth: f for pth, is_dir, f in files if not is_dir}
    if not env:
        return None
    return {
        pth: {"sha256": hashlib.sha256(p.download(run_id, pth)).hexdigest(), "size": f}
        for pth, f in env.items()
    }


def collect(p: Probe, root_id: str, include_environment: bool = True):
    """(chain, envs) — everything the builders below need."""
    chain = build_chain(p, root_id)
    envs = (
        {rid: environment_info(p, rid) for rid in chain} if include_environment else {}
    )
    return chain, envs


# --- W3 PROV / CPM document ---------------------------------------------------


def build_provenance(p: Probe, root_id: str, include_environment: bool = True) -> dict:
    """Assemble the CPM bundle (JSON-LD dict) for the chain rooted at root_id.

    Full parameters and environment are embedded as mlkit:params /
    mlkit:environment attributes characterising each activity (R11).
    """
    chain, envs = collect(p, root_id, include_environment)
    # decided once, before the loop: what the end of this chain produces
    root_is_model, root_out_uri, root_out_how = resolve_output(p, chain, root_id)

    def out_of(run_id: str) -> str:
        """Output entity of any run in the chain (only the end can be a model)."""
        return output_id(run_id, run_id == root_id and root_is_model)

    entity: dict = {}
    activity: dict = {}
    agent: dict = {}
    used: dict = {}
    was_generated_by: dict = {}
    was_derived_from: dict = {}
    specialization_of: dict[str, dict[str, str]] = {}
    derived_pairs: set[tuple[str, str]] = set()
    n = 0

    for run_id, info in chain.items():
        run = info["run"]
        info_ = run.info
        act = f"gen:run_{run_id}"
        user = info["user"]

        # agent (user). R12: no wasAttributedTo with an activity as subject —
        # PROV-CONSTRAINTS keeps entities and activities disjoint, and the
        # wasAssociatedWith records below already carry the attribution.
        # R17: typed prov:Person + schema:Person (schema.org alone is not a
        # PROV type).
        agent.setdefault(
            f"gen:user_{user}",
            {
                "schema:name": [user],
                "prov:type": [_qn("prov:Person"), _qn("schema:Person")],
                "schema:affiliation": ["RationAI"],
            },
        )

        # activity (the run itself). R11 (D-c): params / environment live as
        # plain attributes under the mlkit: namespace, characterising the
        # activity, instead of minted prov:Annotation entities + the
        # non-PROV qualifiedAssociation section. (R5/P2: full payloads, so no
        # 120-char truncation like the diagram path — see build_prov_document.)
        tags = run.data.tags or {}
        act_attrs: dict[str, list[Any]] = {
            "prov:type": [_qn("schema:Action")],
            "prov:startTime": [iso(info_.start_time)],
            "prov:endTime": [iso(info_.end_time)],
            "schema:name": [info["name"]],
            "schema:identifier": [run_id],
            "dct:description": [tags.get("mlflow.note.content", "")],
            # REVISION-2026-10-20 #12: the run is located by its artifact URI.
            "cpm:externalId": [info_.artifact_uri],
            "cpm:externalIdType": ["MLflow artifact URI"],
            "mlkit:params": [
                json.dumps(dict(run.data.params), ensure_ascii=False, sort_keys=True)
            ],
        }
        env = envs.get(run_id) if include_environment else None
        if env:
            act_attrs["mlkit:environment"] = [
                json.dumps(env, ensure_ascii=False, sort_keys=True)
            ]
        activity[act] = act_attrs

        # output entity generated by this activity. On the root run this is the
        # chain's output, so a training run's output is model-typed (T6).
        this_is_model = run_id == root_id and root_is_model
        out_ent = out_of(run_id)
        entity[out_ent] = output_props(
            info["name"],
            root_out_uri if this_is_model else info_.artifact_uri,
            this_is_model,
            root_out_how if this_is_model else "",
        )
        was_generated_by[f"_:n{n}"] = {"prov:entity": out_ent, "prov:activity": act}
        n += 1

        # inputs: entity per artifact URI, used by this activity; when the
        # producing run is in the chain, link the two (wasDerivedFrom) — this
        # is what makes the chain a chain.
        for _exp, up_id, uri in info["inputs"]:
            inp = input_entity_id(uri)
            entity.setdefault(
                inp,
                {
                    "schema:name": [uri],
                    "schema:url": [uri],
                    "prov:type": [_qn("sosa:Sample")],
                    # REVISION-2026-10-20 #12: located by its artifact URI.
                    "cpm:externalId": [uri],
                    "cpm:externalIdType": ["MLflow artifact URI"],
                },
            )
            used[f"_:n{n}"] = {"prov:activity": act, "prov:entity": inp}
            n += 1
            # one derivation per (input, upstream-output) pair — every run
            # consuming the same artifact URI maps to the same input entity.
            # R10: PROV-JSON grammar keys (prov:usedEntity / prov:generatedEntity),
            # not the lossy prov:entity / prov:derivation the old dialect used.
            # Generated entity = the direct entity (input artifact), used
            # entity = the indirect source (the upstream output). Kept in
            # lockstep with build_prov_document's bndl.wasDerivedFrom(inp, out):
            # the prov package emits generatedEntity for its FIRST argument.
            if up_id in chain and (uri, up_id) not in derived_pairs:
                derived_pairs.add((uri, up_id))
                was_derived_from[f"_:n{n}"] = {
                    "prov:generatedEntity": inp,
                    "prov:usedEntity": out_of(up_id),
                }
                n += 1

    # REVISION-2026-10-20 #13: the cpm:BundleMetadata "meta" entity was removed
    # entirely — BundleMetadata is an undefined CPM term, organization-as-entity
    # duplicated the org agent, and gen:input_uris duplicated the used edges.

    # ── CPM backbone over the chain (PROV_TODO.md T5 + T6) ──
    #
    # Up to here every node hangs off its own run and the main activity would be
    # an orphan. The backbone gives it the two relations the standard prescribes
    # (https://www.commonprovenancemodel.org/cpm-backbone-template-v1/):
    #   mainActivity used            currentConnector   — the chain's input boundary
    #   mainActivity wasGeneratedBy   forwardConnector   — the chain's output
    #   forwardConnector wasDerivedFrom currentConnector — output affected by input
    # plus specializationOf from each connector to the domain entity it stands
    # for, which is how the template attaches domain detail without duplicating
    # it. backwardConnector / receiptActivity are not emitted: the receiving step
    # is outside mlkit, so only the sending half of the backbone exists here.
    #
    # dct:hasPart covers every run in the chain, not just the root: the main
    # activity is what the team is responsible for — the whole process — so that
    # the end of the chain (the trained model) can be referenced from it.
    main_act = main_activity_id(root_id)
    activity[main_act] = main_activity_props([f"gen:run_{rid}" for rid in chain])

    # R3: cpm:currentAgent — the organisation that produced this bundle (KB
    # B.6, F.3). Distinct from the individual gen:user_* agents (run authors);
    # CPM wants the producing organisation associated with the main activity.
    # cpm:contactIdPid is a placeholder (D-e) pending a real value.
    org_agent = "gen:org_RationAI"
    agent[org_agent] = {
        "prov:type": [_qn(CPM_CURRENT_AGENT), _qn("prov:Organization")],
        "schema:name": ["RationAI"],
        "cpm:contactIdPid": ["TODO:contactIdPid"],
    }
    # (associated with the main activity in the was_associated block below,
    # where that dict is built — the org is the producer of the whole chain.)

    external_inputs = chain_input_uris(chain)
    # input side: one connector per artifact the chain consumes but does not make
    for uri in external_inputs:
        cur = current_connector_id(uri)
        entity[cur] = {
            "prov:type": [_qn(CPM_CURRENT_CONNECTOR)],
            "schema:name": [uri],
            "schema:url": [uri],
        }
        used[f"_:n{n}"] = {"prov:activity": main_act, "prov:entity": cur}
        n += 1
        # R2 (KB B.3): the *domain entity* is the specific one, the connector
        # the general — specializationOf(domain entity, connector). This is the
        # inverse of the direction T5 first shipped (connector -> domain).
        specialization_of[f"_:n{n}"] = {
            "prov:specificEntity": input_entity_id(uri),
            "prov:generalEntity": cur,
        }
        n += 1

    # REVISION-2026-10-20 #8: the dataset/slide collection this chain draws on —
    # a prov:Collection of the external inputs (consistent with #4), used by the
    # main activity. The backwardConnector that would tie it to the
    # currentConnector is deliberately not emitted yet (MOU exchange pending).
    if external_inputs:
        entity["gen:dataset"] = {
            "prov:type": [_qn("prov:Collection")],
            "schema:name": ["Input dataset (slides/data) of the chain"],
            "cpm:externalId": external_inputs,
            "cpm:externalIdType": ["MLflow artifact URI"],
        }
        used[f"_:n{n}"] = {"prov:activity": main_act, "prov:entity": "gen:dataset"}
        n += 1

    # output side (T6): forward connector out of the main activity, standing for
    # the trained model. Never the "Specialized Forward Connector" object — this
    # is a plain cpm:forwardConnector, typed so nothing else can read it as one.
    fwd = forward_connector_id(root_id)
    entity[fwd] = {
        "prov:type": [_qn(CPM_FORWARD_CONNECTOR)],
        "schema:name": [root_out_uri],
        "schema:url": [root_out_uri],
    }
    was_generated_by[f"_:n{n}"] = {"prov:entity": fwd, "prov:activity": main_act}
    n += 1
    # R2 (KB B.3): domain entity is the specific one, connector the general.
    specialization_of[f"_:n{n}"] = {
        "prov:specificEntity": out_of(root_id),
        "prov:generalEntity": fwd,
    }
    n += 1
    # only when the chain really consumed an external input: the derivation says
    # the output was affected by that input, so it must not be asserted blindly.
    # R10: PROV-JSON grammar keys. The forward connector is the artifact
    # generated (first arg), the current connector the input it was affected by
    # (second arg) — matching bndl.wasDerivedFrom(fwd, cur) on the prov path.
    for uri in external_inputs:
        was_derived_from[f"_:n{n}"] = {
            "prov:generatedEntity": fwd,
            "prov:usedEntity": current_connector_id(uri),
        }
        n += 1

    was_associated: dict = {}
    for run_id, info in chain.items():
        was_associated[f"_:n{n}"] = {
            "prov:activity": f"gen:run_{run_id}",
            "prov:agent": f"gen:user_{info['user']}",
        }
        n += 1
    # R3: the producing organisation is associated with the main activity, not
    # the individual runs (KB B.6).
    was_associated[f"_:n{n}"] = {"prov:activity": main_act, "prov:agent": org_agent}
    n += 1

    return {
        "bundle": {
            f"storage:{root_id}": {
                "prefix": PREFIX,
                "entity": entity,
                "activity": activity,
                "agent": agent,
                "wasAssociatedWith": was_associated,
                "wasGeneratedBy": was_generated_by,
                "used": used,
                "wasDerivedFrom": was_derived_from,
                "specializationOf": specialization_of,
            }
        }
    }


# --- CPF store export (production: cpf.dyn.cloud.e-infra.cz) -------------------
#
# The store speaks the published CPM 1.0 vocabulary, not the internal
# REVISION-2026-10-20 convention this file follows elsewhere (the /ns/cpm#
# namespace, the PascalCase cpm: types, the localhost:8083 prefixes shared
# with rationai/mlkit/provenance/common.py). The 2026-11-05 upload needed
# exactly four fixes, codified here so the store-ready file is generated
# instead of patched by hand:
#   1. storage/meta prefixes -> the production store (storage is org-scoped)
#   2. cpm prefix            -> the published CPM 1.0 namespace
#   3. single-element attribute lists -> scalars (prov:type keeps its list;
#      genuinely multi-valued attributes like dct:hasPart keep theirs)
#   4. the root run's activity is marked cpm:mainActivity with
#      cpm:referencedMetaBundleId = meta:<fresh uuid4> and
#      cpm:referencedMetaBundleSpecV = "1.0" — the store mints the meta
#      bundle itself when the uuid is new; never reuse one.
CPF_STORE_BASE = "https://cpf.dyn.cloud.e-infra.cz/api/v1"
CPF_CPM_NAMESPACE = "https://www.commonprovenancemodel.org/cpm-namespace-v1-0/"


def to_cpf_bundle(
    cpm: dict, org_id: str, root_id: str, store_base: str = CPF_STORE_BASE
) -> dict:
    """Return the store-ready copy of a build_provenance bundle.

    Applies the four CPF-store fixes documented above. The input is not
    mutated; a fresh meta-bundle uuid is minted on every call.
    """
    import copy
    import uuid

    out = copy.deepcopy(cpm)
    doc = out["bundle"][f"storage:{root_id}"]
    doc["prefix"]["storage"] = f"{store_base}/organizations/{org_id}/documents/"
    doc["prefix"]["meta"] = f"{store_base}/documents/meta/"
    doc["prefix"]["cpm"] = CPF_CPM_NAMESPACE

    for section in doc.values():
        if not isinstance(section, dict):
            continue
        for attrs in section.values():
            if not isinstance(attrs, dict):
                continue
            for key, value in attrs.items():
                if key != "prov:type" and isinstance(value, list) and len(value) == 1:
                    attrs[key] = value[0]

    node = doc["activity"][f"gen:run_{root_id}"]
    node["prov:type"] = list(node.get("prov:type", [])) + [_qn("cpm:mainActivity")]
    node["cpm:referencedMetaBundleId"] = {
        "type": "prov:QUALIFIED_NAME",
        "$": f"meta:{uuid.uuid4()}",
    }
    node["cpm:referencedMetaBundleSpecV"] = {"type": "xsd:string", "$": "1.0"}
    return out


# --- prov package document + exports (same way as RationAI/crc_ml-provenance) ---

_MAX_ATTR_CHARS = 120


def _truncate(text: str) -> str:
    """Cap an attribute for the image; prov.dot renders every attribute."""
    if len(text) > _MAX_ATTR_CHARS:
        return text[:_MAX_ATTR_CHARS] + f"… (+{len(text) - _MAX_ATTR_CHARS} chars)"
    return text


def _qn_for(doc: prov.model.ProvDocument, name: str) -> prov.model.QualifiedName:
    """Resolve a 'prefix:local' string into a prov QualifiedName."""
    qn = doc.valid_qualified_name(name)
    if qn is None:
        raise ValueError(f"unregistered PROV namespace in identifier: {name!r}")
    return qn


def _attrs_for_doc(
    doc: prov.model.ProvDocument, props: dict[str, list[Any]]
) -> list[tuple[str, Any]]:
    """Translate the CPM JSON-LD attribute form into prov-native attributes.

    The JSON export stores values as [{"type": "prov:QUALIFIED_NAME", "$": id}]
    (the Java prov_mlflow convention); the prov package wants plain strings and
    QualifiedName objects, and rejects lists of dicts as unhashable. Returned as
    (key, value) pairs, not a dict, so multi-valued attributes (a main activity
    with several dct:hasPart, an entity with several prov:type) survive instead
    of overwriting each other.

    Attribute keys are passed through unchanged ("prov:type" stays "prov:type",
    which prov renders as the element's type rather than a bare `type` attr).
    """
    attrs: list[tuple[str, object]] = []
    for key, values in props.items():
        for value in values:
            if isinstance(value, dict) and value.get("type") == "prov:QUALIFIED_NAME":
                attrs.append((key, _qn_for(doc, value["$"])))
            else:
                attrs.append((key, value))
    return attrs


def _add_attrs(
    element: prov.model.ProvElement,
    doc: prov.model.ProvDocument,
    props: dict[str, list[Any]],
) -> None:
    """Attach props to an already-created element, preserving multi-values."""
    for key, value in _attrs_for_doc(doc, props):
        element.add_attributes({key: value})


def build_prov_document(
    p: Probe, root_id: str, include_environment: bool = True
) -> prov.model.ProvDocument:
    """The chain as a prov.model.ProvDocument (one bundle).

    Exportable to PNG / PROV-N exactly like rationai.utils.provenance does.
    """
    doc = prov.model.ProvDocument()
    for prefix, uri in NAMESPACE_URIS.items():
        doc.add_namespace(prefix, uri)
    # bare attribute keys resolve against this default namespace
    doc.set_default_namespace("http://example.org/0/")
    chain, envs = collect(p, root_id, include_environment)
    root_is_model, root_out_uri, root_out_how = resolve_output(p, chain, root_id)

    def out_of(run_id: str) -> str:
        """Output entity of any run in the chain (only the end can be a model)."""
        return output_id(run_id, run_id == root_id and root_is_model)

    external_inputs = chain_input_uris(chain)
    derived: set[tuple[str, str]] = set()
    seen_agents: set[str] = set()
    bndl = doc.bundle(f"gen:bundle_{root_id}")

    for run_id, info in chain.items():
        run = info["run"]
        info_ = run.info
        # attributes stay small on purpose: prov.dot renders every attribute
        # (show_element_attributes=True, the repo default) into the image, so
        # mlkit:params / mlkit:environment are truncated. Full, untruncated
        # payloads live in the CPM .prov.json. R11 (D-c): same mlkit:* attribute
        # scheme as the JSON path — no more gen:annotation_* entities.
        env = envs.get(run_id) if include_environment else None
        act_attrs: dict[str, object] = {
            # prov:type kept identical to the JSON path (schema:Action):
            # the diagram and the .prov.json must not drift on types.
            "prov:type": _qn_for(doc, "schema:Action"),
            "schema:name": info["name"],
            "schema:identifier": run_id,
            "dct:description": (run.data.tags or {}).get("mlflow.note.content", ""),
            "prov:startTime": iso(info_.start_time),
            "prov:endTime": iso(info_.end_time),
            # REVISION-2026-10-20 #12: matches the JSON path.
            "cpm:externalId": info_.artifact_uri,
            "cpm:externalIdType": "MLflow artifact URI",
            "mlkit:params": _truncate(
                json.dumps(dict(run.data.params), ensure_ascii=False, sort_keys=True)
            ),
        }
        if env:
            act_attrs["mlkit:environment"] = _truncate(
                json.dumps(env, ensure_ascii=False, sort_keys=True)
            )
        act = bndl.activity(f"gen:run_{run_id}", other_attributes=act_attrs)
        # one agent node per person, not per run — the relations below are still
        # per-run, only the declaration is shared. Mirrors agent.setdefault() on
        # the JSON path; without this an 8-run single-author chain drew the same
        # person 8 times in the image. R17: prov:Person + schema:Person, matching
        # the JSON path (schema.org alone is not a PROV type).
        user = info["user"]
        if user not in seen_agents:
            seen_agents.add(user)
            agent_el = bndl.agent(
                f"gen:user_{user}",
                other_attributes={
                    "schema:name": user,
                    "schema:affiliation": "RationAI",
                },
            )
            # multi-valued prov:type goes through _add_attrs: the constructor
            # would store the list itself as one value (unhashable in prov)
            _add_attrs(
                agent_el,
                doc,
                {"prov:type": [_qn("prov:Person"), _qn("schema:Person")]},
            )
        bndl.wasAssociatedWith(act, f"gen:user_{user}")
        # R12: wasAttributedTo dropped (activity subject is invalid).
        # output entity — model-typed when this run is the end of the chain and
        # trained a model (T6). Same props as the JSON path, so the diagram and
        # the .prov.json cannot drift apart on the node that matters most.
        this_is_model = run_id == root_id and root_is_model
        out_ent = bndl.entity(out_of(run_id))
        _add_attrs(
            out_ent,
            doc,
            output_props(
                info["name"],
                root_out_uri if this_is_model else info_.artifact_uri,
                this_is_model,
                root_out_how if this_is_model else "",
            ),
        )
        bndl.wasGeneratedBy(out_ent, act)
        for _exp, up_id, uri in info["inputs"]:
            inp = bndl.entity(
                input_entity_id(uri),
                other_attributes={
                    "schema:name": uri,
                    "schema:url": uri,
                    # prov:type kept identical to the JSON path (sosa:Sample)
                    "prov:type": _qn_for(doc, "sosa:Sample"),
                    # REVISION-2026-10-20 #12: matches the JSON path.
                    "cpm:externalId": uri,
                    "cpm:externalIdType": "MLflow artifact URI",
                },
            )
            bndl.used(act, inp)
            # one derivation per (input, upstream-output) pair: every consumer
            # of the same URI resolves to the same input entity, and the
            # consumer side is already covered by `used`
            if up_id in chain and (uri, up_id) not in derived:
                derived.add((uri, up_id))
                bndl.wasDerivedFrom(inp, out_of(up_id))

    # REVISION-2026-10-20 #13: the cpm:BundleMetadata "meta" entity was removed,
    # mirroring the JSON path — see there for the rationale.

    # ── CPM backbone over the chain (PROV_TODO.md T5 + T6) ──
    # Mirrors build_provenance(): the main activity is attached to the chain with
    # the relations the backbone template prescribes, and each connector is a
    # specializationOf the domain entity it stands for. See the JSON path for the
    # full rationale and for why no backwardConnector / receiptActivity appears.
    main_act = bndl.activity(main_activity_id(root_id))
    _add_attrs(
        main_act,
        doc,
        main_activity_props([f"gen:run_{rid}" for rid in chain]),
    )

    # R3: cpm:currentAgent — the producing organisation (KB B.6), associated
    # with the main activity. Mirrors the JSON path's gen:org_RationAI.
    org_el = bndl.agent(
        "gen:org_RationAI",
        other_attributes={
            "schema:name": "RationAI",
            "cpm:contactIdPid": "TODO:contactIdPid",
        },
    )
    _add_attrs(
        org_el,
        doc,
        {"prov:type": [_qn(CPM_CURRENT_AGENT), _qn("prov:Organization")]},
    )
    bndl.wasAssociatedWith(main_act, "gen:org_RationAI")

    for uri in external_inputs:
        cur = bndl.entity(
            current_connector_id(uri),
            other_attributes={"schema:name": uri, "schema:url": uri},
        )
        _add_attrs(cur, doc, {"prov:type": [_qn(CPM_CURRENT_CONNECTOR)]})
        bndl.used(main_act, cur)
        # R2 (KB B.3): domain entity is the specific one, connector the general.
        bndl.specializationOf(input_entity_id(uri), cur)

    # REVISION-2026-10-20 #8: mirrors the JSON path — the dataset/slide
    # collection, used by the main activity, no backwardConnector yet.
    if external_inputs:
        dataset = bndl.entity(
            "gen:dataset",
            other_attributes={
                "schema:name": "Input dataset (slides/data) of the chain",
                "cpm:externalIdType": "MLflow artifact URI",
            },
        )
        _add_attrs(
            dataset,
            doc,
            {
                "prov:type": [_qn("prov:Collection")],
                "cpm:externalId": external_inputs,
            },
        )
        bndl.used(main_act, dataset)

    # output side (T6): a plain cpm:forwardConnector, deliberately not the
    # separate "Specialized Forward Connector" object
    fwd = bndl.entity(
        forward_connector_id(root_id),
        other_attributes={"schema:name": root_out_uri, "schema:url": root_out_uri},
    )
    _add_attrs(fwd, doc, {"prov:type": [_qn(CPM_FORWARD_CONNECTOR)]})
    bndl.wasGeneratedBy(fwd, main_act)
    # R2 (KB B.3): domain entity is the specific one, connector the general.
    bndl.specializationOf(out_of(root_id), fwd)
    for uri in external_inputs:
        bndl.wasDerivedFrom(fwd, current_connector_id(uri))
    return doc


def export_provenance_png(
    bundle: prov.model.ProvBundle, name: str, out_dir: str | Path = "output"
) -> Path:
    """Render the bundle to <out-dir>/prov-<name>.png.

    Same two-liner as rationai.utils.provenance.export_to_image in
    RationAI/crc_ml-provenance:
    dot = prov.dot.prov_to_dot(bundle); dot.write_png(f"prov-{name}.png")

    Falls back to writing the Graphviz source (.dot) when the `dot` binary is
    not on PATH (it is a system package, not pip-installable): the export stays
    inspectable anywhere, and `dot -Tpng prov-<name>.dot -o ...` reproduces the
    image later. Returns the path actually written.
    """
    dot = prov.dot.prov_to_dot(bundle)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png_path = out / f"prov-{name}.png"
    try:
        dot.write_png(str(png_path))  # type: ignore[attr-defined]  # missing from pydot stubs
        return png_path
    except (FileNotFoundError, OSError) as e:
        # FileNotFoundError on missing binary; OSError when pydot cannot invoke
        # it. Either way, degrade to DOT source.
        logging.getLogger(__name__).warning(
            "graphviz `dot` unavailable (%s) — writing DOT source instead", e
        )
        dot_path = out / f"prov-{name}.dot"
        dot_path.write_text(dot.to_string(), encoding="utf-8")
        return dot_path


def export_provenance_provn(
    doc: prov.model.ProvDocument, name: str, out_dir: str | Path = "output"
) -> Path:
    """Save the bundle as PROV-N — same as rationai.utils.provenance.export_to_provn."""
    path = Path(out_dir) / f"prov-{name}.provn"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        doc.serialize(f, format="provn")
    return path


def parse_run_id(url_or_id: str) -> str:
    m = re.search(r"[0-9a-f]{32}", url_or_id)
    if not m:
        raise ValueError(f"no 32-char hex run id in: {url_or_id!r}")
    return m.group(0)


# --- CLI ----------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="W3 PROV chain + graph for an MLflow run")
    ap.add_argument("url", nargs="?", default=DEFAULT_URL, help="run URL or run id")
    ap.add_argument(
        "--out-dir",
        default="provenance/output",
        help="folder for the .png/.provn/.prov.json exports",
    )
    ap.add_argument(
        "--json-out", help="CPM JSON path (default: <out-dir>/provenance_<run_id>.prov.json)"
    )
    ap.add_argument("--track-uri", default=DEFAULT_TRACK_URI)
    ap.add_argument(
        "--env",
        action="store_true",
        help="download + hash each run's environment/ artifacts (off by default: "
        "artifact downloads over the public API are flaky from the pod, and the "
        "chain / params / PNG do not need them)",
    )
    ap.add_argument(
        "--cpf-org",
        metavar="ORG_UUID",
        help="also write the CPF-store-ready variant next to the .prov.json "
        "(<name>.prov.json.cpf): production store namespaces, flattened "
        "scalars, root run marked cpm:mainActivity. ORG_UUID is the "
        "organisation's uuid in the CPF store (it appears in the store's "
        "document URLs).",
    )
    ap.add_argument(
        "--self-check", action="store_true", help="run the offline self-check and exit"
    )
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check()

    # early, actionable diagnostics (the public server needs basic auth; the PNG
    # needs graphviz). Neither is fatal — just loud, so a blank/missing output is
    # never a mystery.
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        print(
            "WARNING: MLFLOW_TRACKING_PASSWORD is not set — the public MLflow "
            "server requires basic auth. Export MLFLOW_TRACKING_USERNAME + "
            "MLFLOW_TRACKING_PASSWORD first.",
            file=sys.stderr,
        )
    if shutil.which("dot") is None:
        print(
            "WARNING: graphviz `dot` not found on PATH — a PNG cannot be "
            "rendered, so the script will write the .dot source instead. "
            "Install graphviz (e.g. `apt-get install graphviz`) for a real PNG.",
            file=sys.stderr,
        )

    run_id = parse_run_id(args.url)
    p = Probe(args.track_uri)
    if not p.has_run(run_id):
        sys.exit(f"run {run_id} not found on {args.track_uri}")

    include_env = args.env
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = (
        Path(args.json_out)
        if args.json_out
        else out_dir / f"provenance_{run_id}.prov.json"
    )

    cpm = build_provenance(p, run_id, include_environment=include_env)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(cpm, indent=2, ensure_ascii=False) + "\n")

    cpf_out = None
    if args.cpf_org:
        cpf_out = json_out.with_name(json_out.name + ".cpf")
        cpf_out.write_text(
            json.dumps(
                to_cpf_bundle(cpm, args.cpf_org, run_id), indent=2, ensure_ascii=False
            )
            + "\n"
        )

    # graph exports, same way as RationAI/crc_ml-provenance
    name = slug(p.name(run_id))
    doc = build_prov_document(p, run_id, include_environment=include_env)
    bundle = next(iter(doc.bundles))
    png = export_provenance_png(bundle, name, args.out_dir)
    provn = export_provenance_provn(doc, name, args.out_dir)

    # human-readable summary (best-effort: the public API is flaky, and the
    # outputs above are already on disk by the time this runs)
    run = p.get_run(run_id)
    print(f"Run:          {p.name(run_id)}  ({run_id})")
    print(f"User:         {run.info.user_id}")
    print(f"Started:      {iso(run.info.start_time)}")
    try:
        matches = p.matching_runs(run_id)
        print(
            f"Matching:     {len(matches['same_user'])} runs by same user, "
            f"{len(matches['shared_dataset'])} sharing a dataset"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"Matching:     (skipped — {e})")

    chain = build_chain(p, run_id)
    print(f"Chain:        {len(chain)} runs, dataset creation -> this run")
    for rid, info in chain.items():
        upstreams = [up for _, up, _ in info["inputs"] if up != rid]
        mark = "  [root]" if not any(p.has_run(u) for u in upstreams) else ""
        print(f"  - {info['name']}  ({rid}) user={info['user']}{mark}")
    print(f"Provenance:   {json_out.resolve()}")
    if cpf_out is not None:
        print(f"CPF store:    {cpf_out.resolve()}")
    print(f"Graph:        {png.resolve()}")
    print(f"PROV-N:       {provn.resolve()}")
    return 0


# ponytail: self-check is offline by design (no server, no graphviz); live
# behaviour verified by running the default URL.
def self_check() -> int:
    assert re.fullmatch(r"[0-9a-f]{32}", parse_run_id(DEFAULT_URL))
    assert (
        parse_run_id("998f2e710351420db30ce182f284c321")
        == "998f2e710351420db30ce182f284c321"
    )
    m = URI_RE.search(
        "x mlflow-artifacts:/111/e8175ecf823d403ca5b629a9bb3cf874/artifacts/train.csv y"
    )
    assert m.group(1) == "111" and m.group(2) == "e8175ecf823d403ca5b629a9bb3cf874"
    m = URI_RE.search(
        "mlflow-artifacts:/1/" + "ab" * 16 + "/artifacts/embeddings/train"
    )
    assert m.group(2) == "ab" * 16 and m.group(3).strip() == "embeddings/train"
    assert URI_RE.search("no artifact here") is None
    # the lib's input_uris convention: runs:/<32-hex> (how our chain is linked)
    assert RUNS_RE.search("no ref here") is None
    runs_tag = '["runs:/' + "ab" * 16 + '", "runs:/' + "cd" * 16 + '"]'
    assert RUNS_RE.findall(runs_tag) == ["ab" * 16, "cd" * 16]
    # bare 32-hex run ids (our cfg_provenance_*_run_id params) have NO prefix
    # and must not match — otherwise every cfg param would become a fake edge
    assert RUNS_RE.search("780b3e37719c4ecf855ac060cde18357") is None
    assert slug("🏋️‍♂️ Training - high - vgg16") == "training_high_vgg16"
    assert iso(None) == "" and iso(0) == "1970-01-01T00:00:00+00:00"

    # a stub Probe must produce a CPM bundle AND a prov document with every
    # expected section / record type
    # rid_a is the chain end; rid_b/rid_c resolve upstream; rid_d does NOT
    # resolve, so what it produced is the chain's external input boundary
    rid_a, rid_b, rid_c, rid_d = "a" * 32, "b" * 32, "c" * 32, "d" * 32

    class FakeRun:
        class info:
            run_id, user_id, experiment_id = rid_a, "u1", "1"
            start_time, end_time = 1700000000000, 1700000060000
            artifact_uri = f"mlflow-artifacts:/1/{rid_a}/artifacts"

        class data:
            params: ClassVar[dict[str, str]] = {"lr": "0.001"}
            tags: ClassVar[dict[str, str]] = {"mlflow.runName": "Root Train"}
            metrics: ClassVar[dict[str, float]] = {}

    stub = SimpleNamespace(
        get_run=lambda rid: FakeRun(),
        has_run=lambda rid: rid in (rid_a, rid_b, rid_c),
        name=lambda rid: "Root Train" if rid == rid_a else "Upstream Dataset",
        input_refs=lambda rid: {
            # rid_a <- rid_b (dataset.csv) + rid_c (model.h5); rid_c <- rid_b
            # (dataset.csv): two runs in the chain consume the same artifact
            rid_a: [
                ("1", rid_b, f"mlflow-artifacts:/1/{rid_b}/artifacts/dataset.csv"),
                ("1", rid_c, f"mlflow-artifacts:/1/{rid_c}/artifacts/model.h5"),
            ],
            rid_c: [
                ("1", rid_b, f"mlflow-artifacts:/1/{rid_b}/artifacts/dataset.csv"),
                # produced outside the chain (rid_d does not resolve): this is
                # what puts a currentConnector on the chain's input boundary
                ("1", rid_d, f"mlflow-artifacts:/1/{rid_d}/artifacts/wsi.svs"),
            ],
        }.get(rid, []),
        walk_artifacts=lambda rid, path=None: [],
        download=lambda rid, path: b"",
    )

    # CPM dict
    cpm = build_provenance(stub, rid_a)
    b = cpm["bundle"][f"storage:{rid_a}"]
    for key in (
        "prefix",
        "entity",
        "activity",
        "agent",
        "wasAssociatedWith",
        "wasGeneratedBy",
        "used",
        "wasDerivedFrom",
        "specializationOf",
    ):
        assert b.get(key), f"missing/empty section {key}"
    # R12: wasAttributedTo with an activity subject is invalid — it must stay
    # out, so its absence is asserted rather than left to drift back in.
    assert "wasAttributedTo" not in b
    # R11: qualifiedAssociation (non-PROV) and the gen:annotation_* entities are
    # gone; params / environment now live as mlkit:* attributes on the activity.
    assert "qualifiedAssociation" not in b
    assert not any("gen:annotation_" in k for k in b["entity"])
    assert "mlkit:params" in b["activity"][f"gen:run_{rid_a}"]
    assert f"gen:run_{rid_a}" in b["activity"] and f"gen:run_{rid_b}" in b["activity"]
    assert any("gen:input_" in k for k in b["entity"])
    assert any("gen:output_" in k for k in b["entity"])
    assert any(k == "gen:user_u1" for k in b["agent"])
    # R10: PROV-JSON grammar keys, in lockstep with the prov path's
    # bndl.wasDerivedFrom(inp, out_of(up_id)): generatedEntity is the DIRECT
    # entity (the input artifact), usedEntity the indirect source (the
    # upstream output). Exactly ONE record even though two runs consume the URI.
    assert [
        d
        for d in b["wasDerivedFrom"].values()
        if d.get("prov:usedEntity") == f"gen:output_{rid_b}"
    ] == [
        {
            "prov:generatedEntity": "gen:input_"
            + slug(f"mlflow-artifacts:/1/{rid_b}/artifacts/dataset.csv"),
            "prov:usedEntity": f"gen:output_{rid_b}",
        }
    ]
    # ...but two separate `used` relations (one per consumer)
    assert (
        sum(1 for d in b["used"].values() if d["prov:entity"].endswith("dataset_csv"))
        == 2
    )
    # 3 derivations: dataset.csv->rid_b, model.h5->rid_c, and the backbone's
    # forwardConnector->currentConnector (wsi.svs, the one external input)
    assert len(b["wasDerivedFrom"]) == 3

    # ── T5/T6: the CPM backbone ──
    main_id = main_activity_id(rid_a)
    fwd_id = forward_connector_id(rid_a)
    svs = f"mlflow-artifacts:/1/{rid_d}/artifacts/wsi.svs"
    cur_id = current_connector_id(svs)
    # the main activity is no longer an orphan: it uses the chain's input
    # boundary and generates the chain's output
    assert {"prov:activity": main_id, "prov:entity": cur_id} in b["used"].values(), (
        "json: main activity does not use the currentConnector"
    )
    assert {
        "prov:entity": fwd_id,
        "prov:activity": main_id,
    } in b["wasGeneratedBy"].values(), "json: main activity generates no connector"
    assert b["entity"][fwd_id]["prov:type"] == [_qn(CPM_FORWARD_CONNECTOR)]
    assert b["entity"][cur_id]["prov:type"] == [_qn(CPM_CURRENT_CONNECTOR)]
    # REVISION-2026-10-20 #1: CPM types are PascalCase everywhere
    assert b["activity"][main_id]["prov:type"] == [_qn("cpm:MainActivity")]
    # R2: the DOMAIN entity is the specific one, the connector the general —
    # specializationOf(domain entity, connector). Flipped from the T5 direction.
    assert {
        "prov:specificEntity": f"gen:model_{rid_a}",
        "prov:generalEntity": fwd_id,
    } in b["specializationOf"].values()
    assert {
        "prov:specificEntity": input_entity_id(svs),
        "prov:generalEntity": cur_id,
    } in b["specializationOf"].values()
    # R3: the producing organisation (cpm:currentAgent) exists and is associated
    # with the main activity.
    assert b["agent"]["gen:org_RationAI"]["prov:type"] == [
        _qn(CPM_CURRENT_AGENT),
        _qn("prov:Organization"),
    ]
    assert {
        "prov:activity": main_id,
        "prov:agent": "gen:org_RationAI",
    } in b["wasAssociatedWith"].values()
    # T6: the trained model is its own typed, addressable entity
    model_id = f"gen:model_{rid_a}"
    assert model_id in b["entity"], "json: trained-model entity missing"
    assert b["entity"][model_id]["prov:type"] == [_qn(t) for t in MODEL_TYPES]
    assert b["entity"][model_id]["schema:url"] == [
        f"mlflow-artifacts:/1/{rid_a}/artifacts"
    ], "model entity should fall back to the run artifact_uri"
    assert {
        "prov:entity": model_id,
        "prov:activity": f"gen:run_{rid_a}",
    } in b["wasGeneratedBy"].values(), "model entity not generated by the train run"
    # upstream outputs are run artifact folders — REVISION-2026-10-20 #4 types
    # them prov:Collection (hadMember edges are deferred together with #5)
    assert b["entity"][f"gen:output_{rid_b}"]["prov:type"] == [_qn("prov:Collection")]
    # REVISION-2026-10-20 #12: every run activity and artifact entity carries
    # cpm:externalId = its MLflow artifact URI
    run_a_attrs = b["activity"][f"gen:run_{rid_a}"]
    assert run_a_attrs["cpm:externalId"] == [f"mlflow-artifacts:/1/{rid_a}/artifacts"]
    assert run_a_attrs["cpm:externalIdType"] == ["MLflow artifact URI"]
    out_b_attrs = b["entity"][f"gen:output_{rid_b}"]
    assert out_b_attrs["cpm:externalId"] == out_b_attrs["schema:url"]
    assert out_b_attrs["cpm:externalIdType"] == ["MLflow artifact URI"]
    inp_csv = "gen:input_" + slug(f"mlflow-artifacts:/1/{rid_b}/artifacts/dataset.csv")
    assert b["entity"][inp_csv]["cpm:externalId"] == [
        f"mlflow-artifacts:/1/{rid_b}/artifacts/dataset.csv"
    ]
    # REVISION-2026-10-20 #8: the dataset collection exists, used by the main
    # activity, with no backwardConnector
    assert b["entity"]["gen:dataset"]["prov:type"] == [_qn("prov:Collection")]
    assert {"prov:activity": main_id, "prov:entity": "gen:dataset"} in b[
        "used"
    ].values(), "json: main activity does not use the dataset collection"
    # T5: hasPart spans the whole chain, not just the root run
    parts = [p["$"] for p in b["activity"][main_id]["dct:hasPart"]]
    assert parts == [f"gen:run_{rid}" for rid in (rid_a, rid_b, rid_c)], parts
    # no backwardConnector / receiptActivity: the receiving step is not ours
    assert "backwardConnector" not in json.dumps(cpm)
    assert "receiptActivity" not in json.dumps(cpm)

    # CPF store variant (2026-11-05 upload contract): the four fixes hold and
    # the internal bundle is left untouched
    cpf = to_cpf_bundle(cpm, "org-1", rid_a)
    cb = cpf["bundle"][f"storage:{rid_a}"]
    assert cb["prefix"]["storage"] == (
        "https://cpf.dyn.cloud.e-infra.cz/api/v1/organizations/org-1/documents/"
    )
    assert cb["prefix"]["meta"] == "https://cpf.dyn.cloud.e-infra.cz/api/v1/documents/meta/"
    assert cb["prefix"]["cpm"] == CPF_CPM_NAMESPACE
    assert b["prefix"]["storage"] == "http://localhost:8083/api/v1/documents/"
    run_node = cb["activity"][f"gen:run_{rid_a}"]
    assert _qn("cpm:mainActivity") in run_node["prov:type"]
    assert run_node["cpm:referencedMetaBundleId"]["$"].startswith("meta:")
    assert run_node["cpm:referencedMetaBundleSpecV"] == {"type": "xsd:string", "$": "1.0"}
    #: single-element lists become scalars; prov:type and the genuinely
    #: multi-valued dct:hasPart keep their lists
    assert isinstance(run_node["prov:startTime"], str)
    assert isinstance(run_node["prov:type"], list)
    assert isinstance(cb["activity"][main_id]["dct:hasPart"], list)

    # pure helpers, so a wrong guess about the model artifact is testable
    assert is_model_run("Train vgg16", []) is True
    assert (
        is_model_run("Filter tiles", [f"mlflow-artifacts:/1/{'c' * 32}/x.pt"]) is True
    )
    assert is_model_run("Filter tiles", ["mlflow-artifacts:/1/x/dataset.csv"]) is False
    assert run_base_uri(f"mlflow-artifacts:/1/{rid_a}/artifacts") == (
        f"mlflow-artifacts:/1/{rid_a}"
    )
    assert run_base_uri(f"file:///mlruns/1/{rid_a}/artifacts/") == (
        f"file:///mlruns/1/{rid_a}"
    )

    # prov document (same object model RationAI/crc_ml-provenance exports)
    doc = build_prov_document(stub, rid_a)
    assert len(doc.bundles) == 1
    bundle = next(iter(doc.bundles))
    records = bundle.get_records()
    types = [type(r).__name__ for r in records]
    # prov record classes: ProvGeneration=wasGeneratedBy, ProvUsage=used,
    # ProvDerivation=wasDerivedFrom, ProvAssociation=wasAssociatedWith
    for expected in (
        "ProvGeneration",
        "ProvUsage",
        "ProvDerivation",
        "ProvAssociation",
        # ProvSpecialization is how the backbone template attaches a connector
        # to the domain entity it stands for
        "ProvSpecialization",
    ):
        assert expected in types, f"missing record {expected}"
    # R12: no attribution records with an activity subject
    assert "ProvAttribution" not in types
    # R11: the wasInfluencedBy annotation edges are gone
    assert "ProvInfluence" not in types
    # 3 consumed (input, derivation) pairs, 2 unique: dataset.csv->rid_b
    # consumed by both rid_a and rid_c collapses to one record; the third is the
    # backbone's forwardConnector -> currentConnector
    assert types.count("ProvDerivation") == 3
    # one per connector: the external input, and the model output
    assert types.count("ProvSpecialization") == 2
    # R3: one association per run, plus the org agent on the main activity
    assert types.count("ProvAssociation") == 4
    ents = [r for r in bundle.get_records() if isinstance(r, prov.model.ProvElement)]
    ids = {str(e.identifier): e for e in ents}  # QualifiedName -> "gen:xxx"
    assert f"gen:run_{rid_a}" in ids and f"gen:run_{rid_b}" in ids
    inp = "gen:input_" + slug(f"mlflow-artifacts:/1/{rid_b}/artifacts/dataset.csv")
    assert inp in ids and any(
        v.endswith("dataset.csv") for v in ids[inp].get_attribute("schema:name") or []
    )

    # ── T4: the diagram/PROV-N path must carry the CPM nodes the JSON has ──
    assert main_id in ids, f"main activity {main_id} missing from prov document"
    # REVISION-2026-10-20 #13: the meta-bundle entity is gone from BOTH paths
    assert "BundleMetadata" not in json.dumps(cpm), "meta entity crept back in"
    assert not any(str(e).startswith("meta:") for e in ids), (
        "diagram: meta entity crept back in"
    )
    assert "gen:dataset" in ids, "diagram: dataset collection missing"
    # R11: params live as a (truncated) mlkit:params attribute on the activity,
    # not as a gen:annotation_* entity anymore
    assert any(
        "lr" in str(v)
        for v in ids[f"gen:run_{rid_a}"].get_attribute("mlkit:params") or []
    ), "diagram: activity missing mlkit:params"
    main_el = ids[main_id]
    assert any(
        "MainActivity" in str(v) for v in main_el.get_attribute("prov:type") or []
    ), "main activity untyped (REVISION-2026-10-20 #1: PascalCase)"

    # ── T3: both mandatory main-activity attributes present on BOTH paths ──
    for label, getter in (
        ("json", lambda k: b["activity"][main_id].get(k)),
        ("diagram", lambda k: main_el.get_attribute(k)),
    ):
        for key in CPM_PLACEHOLDERS:
            assert getter(key), f"{label}: main activity missing {key}"
    assert b["activity"][main_id]["cpm:referencedMetaBundleId"] == [
        CPM_PLACEHOLDERS["cpm:referencedMetaBundleId"]
    ], "json: referencedMetaBundleId should be the placeholder (meta entity gone)"
    # REVISION-2026-10-20 #3: SpecV is the ISO/IEC 23494 standard string
    assert b["activity"][main_id]["cpm:referencedMetaBundleSpecV"] == [
        "ISO-IS-23494-2:2026"
    ]

    # ── T2a: one identifier scheme, one construction site ──
    # REVISION-2026-10-20 #10: the main activity id moved to the storage prefix
    assert main_id == f"storage:Run_{rid_a[:10]}"
    assert "TrainingRun_" not in json.dumps(cpm), "stale TrainingRun_ id spelling"
    # R4: the KB's spelling, not the misspelling T5 first shipped
    assert "referencedMetaspectVersion" not in json.dumps(cpm)

    # ── T5/T6 parity: the diagram path must carry the same backbone ──
    assert fwd_id in ids, "diagram: forwardConnector missing"
    assert cur_id in ids, "diagram: currentConnector missing"
    assert model_id in ids, "diagram: trained-model entity missing"
    assert any(
        "ForwardConnector" in str(v)
        for v in ids[fwd_id].get_attribute("prov:type") or []
    ), "diagram: forwardConnector untyped (REVISION-2026-10-20 #1: PascalCase)"
    assert any(
        "CreativeWork" in str(v) for v in ids[model_id].get_attribute("prov:type") or []
    ), "diagram: model entity untyped"

    # the main activity must have real edges, not just attributes. Relation
    # endpoints come from formal_attributes, a tuple of (qn, value) pairs whose
    # values are QualifiedNames (which stringify to "prefix:local" and return
    # None from any unknown attribute, so .identifier cannot be used to read them)
    def ends(rel: object) -> dict[str, str]:
        return {
            str(qn).split(":")[-1]: (str(v) if v is not None else "")
            for qn, v in getattr(rel, "formal_attributes", ())
        }

    assert any(
        type(r).__name__ == "ProvGeneration"
        and ends(r).get("entity") == fwd_id
        and ends(r).get("activity") == main_id
        for r in records
    ), "diagram: main activity generates no connector"
    assert any(
        type(r).__name__ == "ProvUsage"
        and ends(r).get("activity") == main_id
        and ends(r).get("entity") == cur_id
        for r in records
    ), "diagram: main activity uses no currentConnector"
    # Compared as a SET: prov keeps a multi-valued attribute in a Python set, so
    # the diagram path has no defined order for several dct:hasPart values while
    # the JSON path preserves insertion order. Same members is the real contract.
    assert set(parts) == {
        str(qn) for qn in (main_el.get_attribute("dct:hasPart") or [])
    }, "diagram: hasPart does not match the JSON path"

    # ── P0 acceptance (review): the document must deserialize cleanly and every
    # derivation must survive with both endpoints intact. This is the review's
    # own validation method, regression-tested: a serialization bug (non-PROV-JSON
    # keys, a dropped argument) fails here rather than at the consumer.
    import io

    buf = io.BytesIO()
    doc.serialize(buf, format="json")
    doc_rt = prov.model.ProvDocument.deserialize(content=buf.getvalue().decode(), format="json")
    assert len(doc_rt.bundles) == 1, "P0: JSON round-trip lost the bundle"
    recs_rt = list(next(iter(doc_rt.bundles)).get_records())
    dervs = [r for r in recs_rt if type(r).__name__ == "ProvDerivation"]
    n_dervs = sum(1 for r in records if type(r).__name__ == "ProvDerivation")
    assert len(dervs) == n_dervs == 3, (
        f"P0: derivation count changed on round-trip: {len(dervs)} vs {n_dervs}"
    )
    for d in dervs:
        fa = {str(qn).split(":")[-1]: v for qn, v in getattr(d, "formal_attributes", ())}
        assert fa.get("generatedEntity") is not None and fa.get("usedEntity") is not None, (
            f"P0: derivation lost an endpoint on round-trip: {fa}"
        )
    # associations must keep their agent too (one per run in the stub chain,
    # plus the org agent on the main activity)
    orig_assoc = [r for r in records if type(r).__name__ == "ProvAssociation"]
    rt_assoc = [r for r in recs_rt if type(r).__name__ == "ProvAssociation"]
    assert len(rt_assoc) == len(orig_assoc) == 4
    for r in rt_assoc:
        agents = [
            str(v)
            for qn, v in getattr(r, "formal_attributes", ())
            if str(qn).endswith("agent")
        ]
        assert agents in (["gen:user_u1"], ["gen:org_RationAI"]), (
            f"P0: association agent lost: {agents}"
        )
    print("self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
