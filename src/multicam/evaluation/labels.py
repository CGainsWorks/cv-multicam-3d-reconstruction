"""Labelling reconstructed points against the ground-truth segmentation.

Evaluation only.  The pipeline never calls anything here.

A turntable scan reconstructs the part *and the fixture it is bolted to*.  Both
are rigid with respect to each other, so both are perfectly good reconstruction
targets and both survive every filter -- there is no signal in the images that
says which is which.  Scoring the whole thing against the CAD model of the part
would report a median error of forty millimetres and blame the reconstruction
for the disc.

So the points are labelled, using the renderer's own segmentation masks, and
the accuracy numbers are reported per class.  This is the honest split: the
"part" number says how well the part was reconstructed, the composition says
how much of the cloud is fixture, and neither is hidden inside the other.
"""

from __future__ import annotations

import numpy as np

#: Mesh index 0 is the part and 1 is the turntable disc, by construction in
#: :mod:`multicam.sim.scene`.  Anything else, including -1, is background.
OBJECT, FIXTURE, BACKGROUND = 0, 1, 2
CLASS_NAMES = {OBJECT: "part", FIXTURE: "turntable", BACKGROUND: "background"}


def label_observations(observations, masks_by_station, n_points):
    """Majority-vote a class for each point from its observations' mask pixels.

    ``masks_by_station`` is ``{station: {camera: mask image}}``.  A point whose
    observations disagree -- which happens on the silhouette, where a pixel can
    land either side of the boundary -- takes the majority, and ties go to the
    part, because a point on the part's edge is still on the part.
    """
    votes = np.zeros((int(n_points), 3), dtype=np.int32)
    for (point, station, camera, xy) in observations:
        masks = masks_by_station.get(station)
        if masks is None or camera not in masks:
            continue
        mask = masks[camera]
        u = int(round(float(xy[0])))
        v = int(round(float(xy[1])))
        if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]):
            continue
        m = int(mask[v, u])
        cls = OBJECT if m == 0 else (FIXTURE if m == 1 else BACKGROUND)
        votes[int(point), cls] += 1

    labels = np.full(int(n_points), BACKGROUND, dtype=np.int8)
    seen = votes.sum(axis=1) > 0
    if seen.any():
        labels[seen] = np.argmax(votes[seen], axis=1).astype(np.int8)
    return labels, votes


def composition(labels):
    """Counts and fractions per class, for the record files."""
    labels = np.asarray(labels)
    n = max(len(labels), 1)
    out = {}
    for cls, name in CLASS_NAMES.items():
        count = int((labels == cls).sum())
        out[name] = count
        out[name + "_fraction"] = count / n
    out["total"] = int(len(labels))
    return out
