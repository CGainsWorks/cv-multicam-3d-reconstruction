"""Train the learned pose-refinement network.

    python scripts/train_refiner.py --data data/turntable --out results

Trains on the *training* half of the pose session only.  The held-out stations
are chosen by a fixed rule and are never loaded here; ``scripts/ablate.py``
evaluates on them.

The perturbation range the refiner is trained over is printed alongside the
measured error distribution of the classical estimator, so it is visible
whether the network was trained on the problem it will actually be given.  A
refiner trained to correct 20 mm errors and then handed 0.5 mm ones has been
set up to fail, and a refiner trained on exactly the classical error
distribution has been set up to look better than it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from multicam.io.dataset import Dataset
from multicam.pose.dataset import PerturbationSpec, load_pose_session
from multicam.pose.render_compare import ModelRenderer
from multicam.pose.train import TrainConfig, resolve_device, train
from multicam.types import RigCalibration


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(_bootstrap.default_data_root() / "turntable"))
    p.add_argument("--out", default=str(_bootstrap.default_results_root()))
    p.add_argument("--models", default=None, help="default <repo>/models")
    p.add_argument("--rig", default=None)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--samples-per-frame", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--stations", type=int, default=0, help="limit training stations")
    p.add_argument("--max-translation-mm", type=float, default=15.0)
    p.add_argument("--max-rotation-deg", type=float, default=8.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="refiner")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print
    data = Dataset(args.data)

    rig_path = Path(args.rig) if args.rig else Path(args.out) / "rig_estimated.yaml"
    rig = RigCalibration.load(rig_path) if rig_path.exists() else data.truth_rig
    cad = data.object_mesh

    train_frames, test_frames = data.pose_split()
    if args.stations:
        train_frames = train_frames[: args.stations]
    say("training on %d stations, holding out %d" % (len(train_frames), len(test_frames)))
    if set(train_frames) & set(test_frames):
        raise SystemExit("the training and test splits overlap; refusing to train")

    images, poses = load_pose_session(data, train_frames)
    renderer = ModelRenderer(cad)

    perturbation = PerturbationSpec(
        max_translation_m=args.max_translation_mm * 1e-3,
        max_rotation_deg=args.max_rotation_deg,
    )
    say("perturbation range: %.2f-%.1f mm, %.2f-%.1f deg (log-uniform magnitude)"
        % (perturbation.min_translation_m * 1000, perturbation.max_translation_m * 1000,
           perturbation.min_rotation_deg, perturbation.max_rotation_deg))

    classical_path = Path(args.out) / "pose_classical.json"
    if classical_path.exists():
        blob = json.loads(classical_path.read_text(encoding="utf-8"))
        m = blob.get("metrics", {})
        say("for comparison, the classical estimator's error on the held-out "
            "stations: translation median %.3f mm / p90 %.3f mm, rotation median "
            "%.4f deg / p90 %.4f deg"
            % (m.get("translation_mm", {}).get("median", float("nan")),
               m.get("translation_mm", {}).get("p90", float("nan")),
               m.get("rotation_deg", {}).get("median", float("nan")),
               m.get("rotation_deg", {}).get("p90", float("nan"))))

    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        samples_per_frame=args.samples_per_frame, learning_rate=args.lr,
        width=args.width, num_workers=args.workers, seed=args.seed, device=args.device,
    )
    device = resolve_device(args.device)
    say("device: %s" % device)

    interactive = sys.stdout.isatty() and not args.quiet

    def progress(row):
        line = ("  epoch %2d  train %.3f mm  val %.3f mm  (identity %.3f mm, "
                "%.0f%% better)  lr %.2e"
                % (row["epoch"], row["train_mm"], row["val_mm"], row["identity_mm"],
                   100 * row["improvement"], row["lr"]))
        if interactive:
            print(line)
        elif row["epoch"] % 4 == 0 or row["epoch"] == args.epochs - 1:
            print(line)

    result = train(images, poses, rig, renderer, cad, config, perturbation,
                   progress if not args.quiet else None)

    say("\nbest validation point-matching error: %.4f mm (%.1f s, %d parameters)"
        % (result.best_validation * 1000, result.seconds,
           result.stats["network"]["parameters"]))

    models_dir = Path(args.models) if args.models else _bootstrap.ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = models_dir / ("%s.pt" % args.tag)
    torch.save(
        {
            "state_dict": result.state_dict,
            "config": result.config,
            "history": result.history,
            "stats": result.stats,
            "commit": _bootstrap.git_commit(),
            "versions": _bootstrap.library_versions(),
        },
        checkpoint,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("%s_training.json" % args.tag)).write_text(
        json.dumps(
            {
                "tag": args.tag,
                "checkpoint": str(checkpoint.as_posix()),
                "config": result.config,
                "stats": result.stats,
                "history": result.history,
                "best_validation_mm": result.best_validation * 1000,
                "seconds": result.seconds,
                "commit": _bootstrap.git_commit(),
                "versions": _bootstrap.library_versions(),
            },
            indent=2, sort_keys=True, default=float,
        ),
        encoding="utf-8",
    )
    say("wrote %s and %s" % (checkpoint, out / ("%s_training.json" % args.tag)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
