"""Workarounds for capabilities multimodars 0.3.4 does not have.

0.3.4 predates VTP centerline reading, the branch model, and
``prepare_centerline``.  Everything here reimplements, in numpy, the minimum
needed to feed 0.3.4 the same centerlines the 0.7.0 pipeline gets, so the two
benchmarks compare algorithms rather than input quality.

What is missing in 0.3.4, and what stands in for it here:

* ``mm.load_centerline`` / ``read_centerline_vtp`` -> :func:`read_vtp_polylines`
  plus :func:`ensure_centerline_csvs`, which caches one CSV per vessel so the
  timed pipeline can call ``np.genfromtxt`` exactly as the v0.3.4 example did.
* ``PyCenterline.calculate_branches`` / ``remove_branch_overlap`` /
  ``get_branch`` - 0.3.4 has no branch model at all, so only the main branch
  survives the conversion.  0.7.0's VTP reader sorts branches by descending arc
  length and calls the longest one branch 0, so :func:`read_vtp_main_branch`
  picks the longest polyline, not the first one in the file.
* ``mm.prepare_centerline`` (``trim_start`` / ``resample`` /
  ``orient_to_reference`` / ``orient_by_max_z`` / ``smooth``) ->
  :func:`prepare_centerline_np`.

The helpers below follow the 0.7.0 Rust implementations
(``src/types/native/centerline.rs``) step for step, including their edge-case
behaviour, so the two pipelines see the same centerline points.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

VESSELS = ("ao", "rca", "lca")


def _data_array(parent: ET.Element, name: str) -> np.ndarray:
    """Return the named ascii DataArray under *parent* as a flat float array."""
    for array in parent.iter("DataArray"):
        if array.get("Name") == name:
            if array.get("format") != "ascii":
                raise ValueError(f"DataArray {name!r} is {array.get('format')}, expected ascii")
            return np.fromstring(array.text, sep=" ")
    raise ValueError(f"No DataArray named {name!r}")


def read_vtp_polylines(path: Path | str) -> list[np.ndarray]:
    """Read an ascii VTK PolyData file as a list of ordered (N, 3) polylines.

    Point order within a line comes from the connectivity array, not from the
    order the points happen to be stored in.
    """
    piece = ET.parse(path).getroot().find("./PolyData/Piece")
    if piece is None:
        raise ValueError(f"{path}: not a PolyData file")

    points = _data_array(piece.find("Points"), "Points").reshape(-1, 3)
    lines = piece.find("Lines")
    connectivity = _data_array(lines, "connectivity").astype(int)
    offsets = _data_array(lines, "offsets").astype(int)

    starts = np.concatenate(([0], offsets[:-1]))
    return [points[connectivity[start:end]] for start, end in zip(starts, offsets)]


def read_vtp_main_branch(path: Path | str) -> np.ndarray:
    """Read a VTP centerline's main branch as an (N, 3) array.

    "Main branch" means the longest polyline, matching how 0.7.0's
    ``read_centerline_vtp`` orders branches.  Side branches are dropped: 0.3.4
    has no branch model, so they would otherwise be concatenated into one
    nonsensical polyline.
    """
    polylines = read_vtp_polylines(path)
    if not polylines:
        raise ValueError(f"{path}: no polylines")
    return max(polylines, key=lambda line: _arc_lengths(line)[-1])


def ensure_centerline_csvs(case_path: Path, cache_dir: Path) -> dict[str, Path]:
    """Convert a case's .vtp centerlines to cached CSVs, once.

    Deliberately *outside* the timed pipeline: 0.3.4 never read VTP, it read
    CSV, so charging it for a pure-Python XML parse would measure this module
    rather than the library.
    """
    out_dir = cache_dir / case_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for vessel in VESSELS:
        csv_path = out_dir / f"{vessel}_cl.csv"
        if not csv_path.exists():
            points = read_vtp_main_branch(case_path / f"{vessel}_cl.vtp")
            np.savetxt(csv_path, points, delimiter=",", fmt="%.9g")
        paths[vessel] = csv_path
    return paths


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(step)))


def _trim_start(points: np.ndarray, rm_start_mm: float) -> np.ndarray:
    """Drop the leading *rm_start_mm* of arc length (0.7.0 ``trim_start``).

    Matches the Rust ``remove_trailing_start``: the new first point is the last
    one still within *rm_start_mm*, so the trim never overshoots.
    """
    if rm_start_mm <= 0 or len(points) <= 1:
        return points
    distance = _arc_lengths(points)
    within = np.flatnonzero(distance[1:] <= rm_start_mm)
    if len(within) == 0:
        return points
    return points[within[-1] + 1 :]


def _resample(points: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Resample to even arc-length spacing (0.7.0 ``resample_branch``).

    Samples at 0, spacing, 2*spacing, ... strictly below the total length, then
    appends the exact endpoint, so the final interval may be shorter than
    *spacing_mm*.
    """
    if len(points) < 2 or spacing_mm <= 1e-12:
        return points
    distance = _arc_lengths(points)
    total = distance[-1]
    if total < 1e-12:
        return points

    targets = np.append(np.arange(0.0, total, spacing_mm), total)
    return np.column_stack([np.interp(targets, distance, points[:, axis]) for axis in range(3)])


def resample_centerline(points: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Public stand-in for ``PyCenterline.resample``, absent in 0.3.4."""
    return np.ascontiguousarray(_resample(np.asarray(points, dtype=float), spacing_mm))


def _orient_to_reference(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Order a coronary ostium-first (0.7.0 ``orient_to_reference``).

    Whichever end sits closer to the reference (aortic) centerline becomes the
    start.  0.3.4's ray-triangle occlusion removal scans the *first*
    ``n_points_intramural`` coronary points, so this ordering is load-bearing.
    """
    head = np.linalg.norm(reference - points[0], axis=1).min()
    tail = np.linalg.norm(reference - points[-1], axis=1).min()
    return points if head <= tail else points[::-1]


def _orient_by_max_z(points: np.ndarray) -> np.ndarray:
    """Start at the highest-z point's end (0.7.0 ``orient_by_max_z``)."""
    return points if int(np.argmax(points[:, 2])) == 0 else points[::-1]


def _smooth(points: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth along the centerline (0.7.0 ``smooth``).

    Reproduces the Rust kernel: truncated at 3 sigma and shrunk symmetrically
    near the ends, which preserves a linear trend exactly and leaves both
    endpoints untouched.
    """
    n = len(points)
    if n == 0 or sigma < 1e-12:
        return points

    radius = int(np.ceil(3.0 * sigma))
    smoothed = np.empty_like(points)
    for i in range(n):
        half = min(i, radius, n - 1 - i)
        offsets = np.arange(-half, half + 1)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        smoothed[i] = weights @ points[i - half : i + half + 1] / weights.sum()
    return smoothed


def prepare_centerline_np(
    points: np.ndarray,
    reference: np.ndarray | None = None,
    spacing_mm: float | None = None,
    rm_start_mm: float = 0.0,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """numpy stand-in for 0.7.0's ``prepare_centerline``.

    Same steps in the same order as 0.7.0 - trim, resample, orient, smooth -
    minus the two branch steps, which have no meaning for the flat single
    polyline 0.3.4 understands.  As in 0.7.0, *reference* doubles as the "this
    is a coronary" signal: given one, the centerline is oriented towards it;
    without one (the aorta) it falls back to max-z.
    """
    points = np.asarray(points, dtype=float)
    if rm_start_mm > 0:
        points = _trim_start(points, rm_start_mm)
    if spacing_mm:
        points = _resample(points, spacing_mm)
    points = _orient_to_reference(points, reference) if reference is not None else _orient_by_max_z(points)
    if smooth_sigma > 0:
        points = _smooth(points, smooth_sigma)
    return np.ascontiguousarray(points)
