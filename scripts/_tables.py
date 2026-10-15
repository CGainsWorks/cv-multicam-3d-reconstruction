"""The tables the README and docs/results.md quote, built from the result files.

Both ``make_tables.py`` (which writes them) and ``check_docs.py`` (which
verifies them) import this, so there is exactly one definition of every table
and a document can only ever be stale, never quietly different.

A table is delimited in the Markdown by::

    <!-- table:name -->
    ...generated content...
    <!-- /table:name -->

Anything between the markers is replaced wholesale.  Anything outside them is
written by hand and is never touched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MISSING = "_Not generated yet: run `make experiments`._"


def load(results, name):
    path = Path(results) / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, spec="%.3f"):
    if value is None:
        return "--"
    try:
        if value != value:          # NaN
            return "--"
        return spec % value
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------
# Builders.  Each returns a Markdown fragment, or MISSING.
# --------------------------------------------------------------------------


def table_calibration(results):
    d = load(results, "calibration.json")
    if d is None:
        return MISSING
    rows = [
        "| camera | views | intrinsic RMS (px) | fx error (px) | cx error (px) | "
        "cy error (px) | baseline (mm) | extrinsic error (mm) | extrinsic error (deg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    errors = {e["camera"]: e for e in d.get("extrinsic_errors", [])}
    intrinsic_errors = {e["camera"]: e for e in d.get("intrinsic_errors", [])}
    for i, cam in enumerate(d["intrinsics"]):
        e = errors.get(i, {})
        ie = intrinsic_errors.get(i, {})
        rows.append(
            "| `%s` | %d | %s | %s | %s | %s | %s | %s | %s |"
            % (
                cam["camera"], cam["views"], _fmt(cam["rmse_px"], "%.4f"),
                _fmt(ie.get("fx_error_px"), "%+.2f"),
                _fmt(ie.get("cx_error_px"), "%+.2f"),
                _fmt(ie.get("cy_error_px"), "%+.2f"),
                _fmt(e.get("baseline_mm"), "%.1f"),
                _fmt(e.get("translation_error_mm"), "%.2f"),
                _fmt(e.get("rotation_error_deg"), "%.3f"),
            )
        )
    ext = d.get("extrinsics", {})
    rows.append(
        "| **rig** | %d stations | **%s** | | | | | **%s max** | **%s max** |"
        % (
            ext.get("frames", 0), _fmt(ext.get("rmse_px"), "%.4f"),
            _fmt(d["headline"]["extrinsic_max_translation_error_mm"], "%.2f"),
            _fmt(d["headline"]["extrinsic_max_rotation_error_deg"], "%.3f"),
        )
    )
    return "\n".join(rows)


def table_reconstruction(results):
    sparse = load(results, "sparse.json")
    dense = load(results, "dense.json")
    if sparse is None:
        return MISSING

    rows = [
        "| stage | points on the part | median (mm) | mean (mm) | 90th pct (mm) | "
        "RMS (mm) | completeness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    a = sparse.get("cad_accuracy_part", {})
    c = sparse.get("cad_completeness_part", {})
    rows.append(
        "| sparse | %d | **%s** | %s | %s | %s | %s%% at 3 mm |"
        % (sparse["headline"]["part_points"], _fmt(a.get("median_mm")),
           _fmt(a.get("mean_mm")), _fmt(a.get("p90_mm")), _fmt(a.get("rms_mm")),
           _fmt(100 * c.get("completeness", float("nan")), "%.1f"))
    )
    if dense is not None:
        a = dense.get("cad_accuracy_part", {})
        c1 = dense.get("cad_completeness_1mm", {})
        rows.append(
            "| dense | %d | **%s** | %s | %s | %s | %s%% at 1 mm |"
            % (dense["headline"]["part_points"], _fmt(a.get("median_mm")),
               _fmt(a.get("mean_mm")), _fmt(a.get("p90_mm")), _fmt(a.get("rms_mm")),
               _fmt(100 * c1.get("completeness", float("nan")), "%.1f"))
        )
    return "\n".join(rows)


def table_pose(results):
    d = load(results, "pose_classical.json")
    if d is None:
        return MISSING
    m = d.get("metrics", {})
    rows = [
        "| metric | median | mean | 90th percentile |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key, spec in (
        ("translation (mm)", "translation_mm", "%.3f"),
        ("rotation (deg)", "rotation_deg", "%.4f"),
        ("ADD (mm)", "add_mm", "%.3f"),
        ("ADD-S (mm)", "adds_mm", "%.3f"),
    ):
        block = m.get(key, {})
        rows.append("| %s | **%s** | %s | %s |"
                    % (label, _fmt(block.get("median"), spec),
                       _fmt(block.get("mean"), spec), _fmt(block.get("p90"), spec)))
    rows.append("")
    rows.append("| recall | ADD | ADD-S |")
    rows.append("| --- | ---: | ---: |")
    for pct in (2, 5, 10):
        rows.append(
            "| within %d%% of diameter (%s mm) | %s%% | %s%% |"
            % (pct, _fmt(pct / 100.0 * m.get("diameter_mm", float("nan")), "%.1f"),
               _fmt(100 * m.get("add_recall_%d_pct_diameter" % pct, float("nan")), "%.1f"),
               _fmt(100 * m.get("adds_recall_%d_pct_diameter" % pct, float("nan")), "%.1f"))
        )
    rows.append("| within 1 mm | %s%% | -- |"
                % _fmt(100 * m.get("add_recall_1mm", float("nan")), "%.1f"))
    rows.append("")
    rows.append("%d stations, %d solved, %d refused."
                % (m.get("n", 0), m.get("n_solved", 0), m.get("failures", 0)))
    return "\n".join(rows)


def table_ablation(results):
    d = load(results, "ablation.json")
    if d is None:
        return MISSING
    dep = d.get("deployment")
    if not dep:
        return MISSING
    rows = [
        "| configuration | median ADD (mm) | median translation (mm) | "
        "median rotation (deg) | recall at 2% of diameter |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, block in (("classical alone", dep["classical"]),
                         ("classical + network", dep["classical_plus_network"])):
        rows.append(
            "| %s | %s | %s | %s | %s%% |"
            % (label, _fmt(block["add_mm"]["median"]),
               _fmt(block["translation_mm"]["median"]),
               _fmt(block["rotation_deg"]["median"], "%.4f"),
               _fmt(100 * block["add_recall_2_pct_diameter"], "%.1f"))
        )
    delta = dep.get("add_median_delta_mm", float("nan"))
    verdict = "**improves** ADD by %.3f mm" % -delta if delta < 0 else \
              "**does not improve** ADD; it is %.3f mm worse" % delta
    rows.append("")
    rows.append("The network %s, at %s ms per station on %s."
                % (verdict, _fmt(dep.get("refine_ms_per_station"), "%.0f"),
                   d.get("device", "cpu")))
    return "\n".join(rows)


def table_head_to_head(results):
    d = load(results, "ablation.json")
    if d is None or "head_to_head" not in d:
        return MISSING
    rows = [
        "| perturbation applied | start | classical LM | network | better |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for band in d["head_to_head"]:
        lm = band["classical_lm"]["add_mm"]["median"]
        net = band["network"]["add_mm"]["median"]
        rows.append(
            "| %s | %s | **%s** | %s | %s |"
            % (band["band"], _fmt(band["start"]["add_mm"]["median"]),
               _fmt(lm), _fmt(net),
               "network" if net < lm else "classical")
        )
    rows.append("")
    rows.append("Median ADD in mm, same perturbed starting pose for both refiners.")
    return "\n".join(rows)


def table_colmap(results):
    d = load(results, "colmap.json")
    if d is None:
        return MISSING
    if not d.get("available", False):
        return ("_COLMAP was not run: %s._" % d.get("reason", "pycolmap unavailable"))
    if "cad_accuracy" not in d:
        return "_COLMAP ran but produced no reconstruction to compare._"
    rows = [
        "| | points on the part | median to CAD (mm) | RMS (mm) | completeness at 3 mm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("ours", "ours"), ("COLMAP", "colmap")):
        a = d["cad_accuracy"][key]
        c = d["cad_completeness_3mm"][key]
        rows.append("| %s | %d | %s | %s | %s%% |"
                    % (label, a["points"], _fmt(a["median_mm"]), _fmt(a["rms_mm"]),
                       _fmt(100 * c["completeness"], "%.1f")))
    rows.append("")
    rows.append(
        "COLMAP registered %d of %d images. Its reconstruction is metric only "
        "after a scale factor of **%s**, which is what the rig calibration "
        "supplies for free; camera centres then agree with ours to a median of "
        "**%s mm**."
        % (d["colmap"]["images_registered"], d["colmap"]["images_total"],
           _fmt(d["alignment"]["scale"], "%.4f"),
           _fmt(d["camera_centre_agreement_mm"]["median"], "%.2f"))
    )
    return "\n".join(rows)


def table_headline(results):
    calibration = load(results, "calibration.json")
    sparse = load(results, "sparse.json")
    pose = load(results, "pose_classical.json")
    if not (calibration and sparse and pose):
        return MISSING
    return (
        "The rig calibrates to **%s px** of reprojection residual and to within "
        "**%s mm** of its true geometry on a %s mm baseline. The part "
        "reconstructs to a median of **%s mm** from its CAD surface. 6-DoF pose "
        "on held-out stations is **%s mm** and **%s deg** at the median."
        % (
            _fmt(calibration["headline"]["extrinsic_rmse_px"], "%.3f"),
            _fmt(calibration["headline"]["extrinsic_max_translation_error_mm"], "%.2f"),
            _fmt(max(e["baseline_mm"] for e in calibration["extrinsic_errors"]), "%.0f"),
            _fmt(sparse["headline"]["cad_median_error_mm"], "%.2f"),
            _fmt(pose["headline"]["translation_median_mm"], "%.2f"),
            _fmt(pose["headline"]["rotation_median_deg"], "%.3f"),
        )
    )


def table_timing(results):
    rows = [
        "| stage | wall clock | what it does |",
        "| --- | ---: | --- |",
    ]
    pieces = []
    for name, label, what in (
        ("sparse.json", "reconstruct", "features, tracks, bundle adjustment"),
        ("dense.json", "dense", "plane sweep, fusion, meshing"),
        ("pose_classical.json", "pose", "generalised PnP on the held-out stations"),
        ("ablation.json", "ablate", "the four ablation experiments"),
    ):
        d = load(results, name)
        if d is None:
            continue
        seconds = d.get("seconds") or d.get("sweep_seconds")
        if seconds is None:
            continue
        pieces.append("| `make %s` | %s s | %s |" % (label, _fmt(seconds, "%.0f"), what))
    if not pieces:
        return MISSING
    return "\n".join(rows + pieces)


#: Every table the documents may contain.
BUILDERS = {
    "headline": table_headline,
    "calibration": table_calibration,
    "reconstruction": table_reconstruction,
    "pose": table_pose,
    "ablation": table_ablation,
    "head_to_head": table_head_to_head,
    "colmap": table_colmap,
    "timing": table_timing,
}


def marker_pattern(name):
    return re.compile(
        r"(<!-- table:%s -->\n)(.*?)(\n<!-- /table:%s -->)" % (re.escape(name),
                                                               re.escape(name)),
        re.DOTALL,
    )


def find_tables(text):
    return set(re.findall(r"<!-- table:([a-z_]+) -->", text))


def render(text, results):
    """Replace every marked table in ``text``.  Returns ``(new_text, names)``."""
    names = sorted(find_tables(text))
    for name in names:
        builder = BUILDERS.get(name)
        if builder is None:
            raise KeyError("no builder for table %r" % name)
        content = builder(results)
        text = marker_pattern(name).sub(
            lambda m, c=content: m.group(1) + c + m.group(3), text
        )
    return text, names


#: The documents that carry generated tables.
DOCUMENTS = ("README.md", "docs/results.md")
