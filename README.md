# cv-multicam-3d-reconstruction

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

A calibrated multi-camera rig producing markerless 3D reconstruction and 6-DoF object pose. Classical feature matching and bundle adjustment provide the geometry, and a PyTorch refinement network improves pose accuracy on top, with an ablation proving the network earns its place.

---

## What this covers

CV entries this repository is the evidence for:

- Multi-Camera 3D Reconstruction and Pose Estimation

---

## Requirements

### Hardware

- Multiple synchronised cameras (three or more)
- Rigid mounting rig
- Calibration target (checkerboard or ChArUco)
- Turntable or a robotic pick cell
- Target objects with known CAD models

### Software

- OpenCV for feature matching and bundle adjustment
- Open3D for point-cloud fusion
- PyTorch pose-refinement network
- Multi-camera calibration toolchain (Kalibr or custom)
- COLMAP as a reference baseline

### Report and documentation deliverables

- Calibration residuals
- Reconstruction accuracy against ground truth
- 6-DoF pose error statistics
- Ablation showing the network contribution over classical refinement alone

---

## Inputs

- Synchronised multi-view image sets
- Calibration target captures
- Camera intrinsics and extrinsics
- Object CAD models for evaluation

## Outputs

- Camera calibration files
- Sparse and dense point clouds
- Fused mesh
- 6-DoF object poses with covariance
- Reprojection error and pose error metrics

---

## Intended structure

```
src/
  calibration/            intrinsics, extrinsics, rig calibration
  reconstruction/         features, matching, bundle adjustment
  fusion/                 Open3D point cloud fusion, meshing
  pose/                   classical PnP, learned refinement network
  evaluation/             pose error, reprojection error
capture/                  synchronised capture scripts
data/                     calibration sets, object CAD, DATASET_CARD.md
analysis/
docs/
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] Synchronised multi-camera capture
- [ ] Intrinsic calibration per camera
- [ ] Rig extrinsic calibration with residuals reported
- [ ] Feature matching and sparse reconstruction
- [ ] Bundle adjustment
- [ ] Dense reconstruction and Open3D fusion
- [ ] Classical 6-DoF pose via PnP
- [ ] Learned pose-refinement network
- [ ] Ablation: classical only vs classical plus network
- [ ] Evaluation against CAD ground truth

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: cv-multicam-3d-reconstruction
A calibrated multi-camera rig producing markerless 3D reconstruction and 6-DoF object pose. Classical feature matching and bundle adjustment provide the geometry, and a PyTorch refinement network improves pose accuracy on top, with an ablation proving the network earns its place.

TARGET STACK
- OpenCV for feature matching and bundle adjustment
- Open3D for point-cloud fusion
- PyTorch pose-refinement network
- Multi-camera calibration toolchain (Kalibr or custom)
- COLMAP as a reference baseline

HARDWARE CONTEXT
- Multiple synchronised cameras (three or more)
- Rigid mounting rig
- Calibration target (checkerboard or ChArUco)
- Turntable or a robotic pick cell
- Target objects with known CAD models

THE SYSTEM MUST TAKE THESE INPUTS
- Synchronised multi-view image sets
- Calibration target captures
- Camera intrinsics and extrinsics
- Object CAD models for evaluation

AND PRODUCE THESE OUTPUTS
- Camera calibration files
- Sparse and dense point clouds
- Fused mesh
- 6-DoF object poses with covariance
- Reprojection error and pose error metrics

INTENDED REPOSITORY LAYOUT
src/
  calibration/            intrinsics, extrinsics, rig calibration
  reconstruction/         features, matching, bundle adjustment
  fusion/                 Open3D point cloud fusion, meshing
  pose/                   classical PnP, learned refinement network
  evaluation/             pose error, reprojection error
capture/                  synchronised capture scripts
data/                     calibration sets, object CAD, DATASET_CARD.md
analysis/
docs/

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.


START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
