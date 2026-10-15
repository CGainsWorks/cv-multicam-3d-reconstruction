"""Regenerate every figure in docs/plots from the result records.

    python scripts/plots.py --results results --out docs/plots

Reads only the JSON and NPZ files the pipeline scripts wrote, so a figure can
never disagree with the number it is drawn from.  A figure whose inputs are
missing is skipped with a message rather than drawn from nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1b2733"
GRID = "#d7dde3"
ACCENT = "#2f6f9f"
WARM = "#c2622d"
MUTED = "#8a97a3"


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)


def load(results, name):
    path = Path(results) / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return np.load(path)


def save(fig, out, name, say):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say("  wrote %s" % path)


# --------------------------------------------------------------------------


def plot_calibration(results, out, say):
    data = load(results, "calibration.json")
    if data is None:
        return say("  skipping calibration.png: no calibration.json")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    cameras = [r["camera"] for r in data["intrinsics"]]
    rmse = [r["rmse_px"] for r in data["intrinsics"]]
    linear = [r["linear_rmse_px"] for r in data["intrinsics"]]

    x = np.arange(len(cameras))
    axes[0].bar(x - 0.2, linear, 0.4, color=MUTED, label="linear stage")
    axes[0].bar(x + 0.2, rmse, 0.4, color=ACCENT, label="after refinement")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cameras)
    axes[0].legend(fontsize=7, frameon=False)
    style(axes[0], "Intrinsic residual", ylabel="RMS reprojection (px)")

    errors = data.get("extrinsic_errors", [])
    baseline = [r["baseline_mm"] for r in errors]
    trans = [r["translation_error_mm"] for r in errors]
    rot = [r["rotation_error_deg"] for r in errors]
    axes[1].plot(baseline, trans, "o-", color=ACCENT, markersize=5)
    for b, t, r in zip(baseline, trans, errors):
        axes[1].annotate("cam%d" % r["camera"], (b, t), textcoords="offset points",
                         xytext=(4, 4), fontsize=7, color=MUTED)
    style(axes[1], "Extrinsic error against truth",
          xlabel="baseline from reference (mm)", ylabel="translation error (mm)")

    axes[2].plot(baseline, rot, "o-", color=WARM, markersize=5)
    style(axes[2], "Extrinsic rotation error",
          xlabel="baseline from reference (mm)", ylabel="rotation error (deg)")

    fig.suptitle("Calibration: residuals, and the error the residuals do not show",
                 color=INK, fontsize=11, x=0.02, ha="left", y=1.04)
    fig.tight_layout()
    save(fig, out, "calibration.png", say)


def plot_reconstruction(results, out, say):
    data = load(results, "sparse.json")
    npz = load(results, "sparse.npz")
    if data is None or npz is None:
        return say("  skipping reconstruction.png: no sparse.json/npz")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

    comp = data.get("composition", {})
    labels = ["part", "turntable", "background"]
    counts = [comp.get(k, 0) for k in labels]
    axes[0].bar(labels, counts, color=[ACCENT, MUTED, WARM])
    for i, c in enumerate(counts):
        axes[0].annotate("%d" % c, (i, c), ha="center", va="bottom", fontsize=8, color=INK)
    style(axes[0], "What was reconstructed", ylabel="points")

    acc = data.get("cad_accuracy_part", {})
    tracks = np.asarray(npz["track_stations"]) if "track_stations" in npz else None
    if tracks is not None:
        axes[1].hist(tracks, bins=np.arange(1.5, tracks.max() + 1.5), color=ACCENT)
        style(axes[1], "Track length", xlabel="stations a point was seen at",
              ylabel="points")

    motions = data.get("reconstruction", {}).get("motions", {})
    rots, inliers, static = [], [], []
    for info in motions.values():
        for m in info.get("motions", []):
            rots.append(m["rotation_deg"])
            inliers.append(m["inliers"])
            static.append(m["static"])
    if rots:
        static = np.asarray(static)
        rots = np.asarray(rots)
        inliers = np.asarray(inliers)
        axes[2].scatter(rots[static], inliers[static], s=18, color=WARM,
                        label="static (rejected)")
        axes[2].scatter(rots[~static], inliers[~static], s=18, color=ACCENT,
                        label="moving (kept)")
        axes[2].legend(fontsize=7, frameon=False)
        style(axes[2], "Every motion the alignment found",
              xlabel="rotation between stations (deg)", ylabel="inliers")

    fig.suptitle(
        "Sparse reconstruction: %d part points, median %.2f mm from the CAD surface"
        % (comp.get("part", 0), acc.get("median_mm", float("nan"))),
        color=INK, fontsize=11, x=0.02, ha="left", y=1.04,
    )
    fig.tight_layout()
    save(fig, out, "reconstruction.png", say)


def plot_dense(results, out, say):
    data = load(results, "dense.json")
    if data is None:
        return say("  skipping dense.png: no dense.json")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    acc = data.get("cad_accuracy_part", {})
    depth = data.get("depth_accuracy", {})
    names = ["median", "mean", "p90", "rms"]
    values = [acc.get(n + "_mm", np.nan) for n in names]
    axes[0].bar(names, values, color=ACCENT)
    for i, v in enumerate(values):
        axes[0].annotate("%.2f" % v, (i, v), ha="center", va="bottom", fontsize=8,
                         color=INK)
    style(axes[0], "Dense cloud against CAD", ylabel="error (mm)")

    sweep = data.get("sweep", [])
    if sweep:
        valid = [s.get("cross_checked", s.get("valid", 0)) for s in sweep]
        rejected_a = [s.get("rejected_ambiguous", 0) for s in sweep]
        rejected_s = [s.get("rejected_score", 0) for s in sweep]
        x = np.arange(len(sweep))
        axes[1].bar(x, valid, color=ACCENT, label="kept")
        axes[1].bar(x, rejected_a, bottom=valid, color=MUTED, label="ambiguous")
        axes[1].bar(x, rejected_s, bottom=np.array(valid) + np.array(rejected_a),
                    color=WARM, label="low score")
        axes[1].legend(fontsize=7, frameon=False)
        style(axes[1], "Plane sweep, per depth map", xlabel="depth map",
              ylabel="pixels")

    fig.suptitle(
        "Dense reconstruction: depth median %.2f mm, cloud median %.2f mm from CAD"
        % (depth.get("median_mm", np.nan), acc.get("median_mm", np.nan)),
        color=INK, fontsize=11, x=0.02, ha="left", y=1.04,
    )
    fig.tight_layout()
    save(fig, out, "dense.png", say)


def plot_pose(results, out, say):
    npz = load(results, "pose_classical.npz")
    data = load(results, "pose_classical.json")
    if npz is None or data is None:
        return say("  skipping pose.png: no pose_classical.json/npz")

    from multicam.geometry import se3

    est, tru, ok = npz["estimates"], npz["truths"], npz["solved"]
    trans, rot = [], []
    for e, t, s in zip(est, tru, ok):
        if not s:
            continue
        a, b = se3.pose_error(e, t)
        trans.append(a * 1000)
        rot.append(b)
    trans, rot = np.asarray(trans), np.asarray(rot)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    order = np.argsort(trans)
    axes[0].plot(np.arange(len(trans)) / max(len(trans) - 1, 1) * 100, trans[order],
                 color=ACCENT, linewidth=1.6)
    axes[0].set_yscale("log")
    style(axes[0], "Translation error, sorted",
          xlabel="percentile of held-out stations", ylabel="error (mm)")

    order = np.argsort(rot)
    axes[1].plot(np.arange(len(rot)) / max(len(rot) - 1, 1) * 100, rot[order],
                 color=WARM, linewidth=1.6)
    axes[1].set_yscale("log")
    style(axes[1], "Rotation error, sorted",
          xlabel="percentile of held-out stations", ylabel="error (deg)")

    metrics = data.get("metrics", {})
    fig.suptitle(
        "Classical 6-DoF pose on %d held-out stations: median %.2f mm, %.3f deg"
        % (metrics.get("n", 0), metrics.get("translation_mm", {}).get("median", np.nan),
           metrics.get("rotation_deg", {}).get("median", np.nan)),
        color=INK, fontsize=11, x=0.02, ha="left", y=1.04,
    )
    fig.tight_layout()
    save(fig, out, "pose.png", say)


def plot_ablation(results, out, say):
    data = load(results, "ablation.json")
    if data is None:
        return say("  skipping ablation.png: no ablation.json")

    panels = sum(k in data for k in ("deployment", "head_to_head", "iterations", "starved"))
    if not panels:
        return say("  skipping ablation.png: ablation.json has no experiments")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    if "head_to_head" in data:
        bands = data["head_to_head"]
        x = np.arange(len(bands))
        start = [b["start"]["add_mm"]["median"] for b in bands]
        lm = [b["classical_lm"]["add_mm"]["median"] for b in bands]
        net = [b["network"]["add_mm"]["median"] for b in bands]
        axes[0].plot(x, start, "o--", color=MUTED, label="perturbed start")
        axes[0].plot(x, lm, "o-", color=ACCENT, label="classical LM")
        axes[0].plot(x, net, "o-", color=WARM, label="network")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([b["band"] for b in bands], fontsize=6, rotation=20)
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=7, frameon=False)
        style(axes[0], "Same start, two refiners", ylabel="median ADD (mm)")

    if "iterations" in data:
        its = data["iterations"]
        x = [r["iterations"] for r in its]
        y = [r["metrics"]["add_mm"]["median"] for r in its]
        axes[1].plot(x, y, "o-", color=WARM)
        axes[1].axhline(y[0], color=ACCENT, linestyle="--", linewidth=1,
                        label="classical alone")
        axes[1].legend(fontsize=7, frameon=False)
        style(axes[1], "Network iterations", xlabel="refinement steps",
              ylabel="median ADD (mm)")

    if "starved" in data:
        st = data["starved"]
        x = [100 * r["keep_fraction"] for r in st]
        base = [r["classical"]["add_mm"]["median"] for r in st]
        net = [r["classical_plus_network"]["add_mm"]["median"] for r in st]
        axes[2].plot(x, base, "o-", color=ACCENT, label="classical")
        axes[2].plot(x, net, "o-", color=WARM, label="classical + network")
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        axes[2].legend(fontsize=7, frameon=False)
        style(axes[2], "As correspondences are starved",
              xlabel="correspondences kept (%)", ylabel="median ADD (mm)")

    dep = data.get("deployment", {})
    verdict = "the network helps" if dep.get("network_helps") else "the network does not help"
    fig.suptitle(
        "Ablation on %d held-out stations: %s (ADD %+.3f mm)"
        % (data.get("stations", 0), verdict, dep.get("add_median_delta_mm", np.nan)),
        color=INK, fontsize=11, x=0.02, ha="left", y=1.04,
    )
    fig.tight_layout()
    save(fig, out, "ablation.png", say)


def plot_summary(results, out, say):
    """The one figure the README opens with."""
    calibration = load(results, "calibration.json")
    sparse = load(results, "sparse.json")
    dense = load(results, "dense.json")
    pose = load(results, "pose_classical.json")
    ablation = load(results, "ablation.json")
    if not any((calibration, sparse, dense, pose)):
        return say("  skipping summary.png: no results at all")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    stages, values, colours = [], [], []
    if calibration:
        stages.append("rig\nextrinsics")
        values.append(calibration["headline"]["extrinsic_max_translation_error_mm"])
        colours.append(MUTED)
    if sparse:
        stages.append("sparse\npoints")
        values.append(sparse["headline"]["cad_median_error_mm"])
        colours.append(ACCENT)
    if dense:
        stages.append("dense\ncloud")
        values.append(dense["headline"]["cad_median_error_mm"])
        colours.append(ACCENT)
    if pose:
        stages.append("6-DoF\npose")
        values.append(pose["headline"]["translation_median_mm"])
        colours.append(WARM)
    axes[0].bar(stages, values, color=colours)
    for i, v in enumerate(values):
        axes[0].annotate("%.2f" % v, (i, v), ha="center", va="bottom", fontsize=8,
                         color=INK)
    axes[0].tick_params(axis="x", labelsize=8)
    style(axes[0], "Error through the pipeline",
          ylabel="mm (max for the rig, median otherwise)")

    if sparse and dense:
        names = ["sparse", "dense"]
        completeness = [
            100 * sparse["headline"].get("cad_completeness_3mm", np.nan),
            100 * dense["headline"].get("completeness_1mm", np.nan),
        ]
        axes[1].bar(names, completeness, color=[MUTED, ACCENT])
        for i, v in enumerate(completeness):
            axes[1].annotate("%.0f%%" % v, (i, v), ha="center", va="bottom", fontsize=8,
                             color=INK)
        axes[1].tick_params(axis="x", labelsize=8)
        style(axes[1], "Surface completeness\n(sparse at 3 mm, dense at 1 mm)",
              ylabel="per cent of CAD surface")

    if ablation and "deployment" in ablation:
        dep = ablation["deployment"]
        names = ["classical", "classical\n+ network"]
        add = [dep["classical"]["add_mm"]["median"],
               dep["classical_plus_network"]["add_mm"]["median"]]
        axes[2].bar(names, add, color=[ACCENT, WARM])
        for i, v in enumerate(add):
            axes[2].annotate("%.3f" % v, (i, v), ha="center", va="bottom", fontsize=8,
                             color=INK)
        axes[2].tick_params(axis="x", labelsize=8)
        style(axes[2], "The ablation", ylabel="median ADD (mm)")
    else:
        axes[2].axis("off")

    fig.suptitle("Multi-camera reconstruction and 6-DoF pose", color=INK, fontsize=12,
                 x=0.02, ha="left", y=1.04)
    fig.tight_layout()
    save(fig, out, "summary.png", say)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(_bootstrap.default_results_root()))
    p.add_argument("--out", default=str(_bootstrap.ROOT / "docs" / "plots"))
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print
    say("figures from %s" % args.results)
    for fn in (plot_summary, plot_calibration, plot_reconstruction, plot_dense,
               plot_pose, plot_ablation):
        fn(args.results, args.out, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
