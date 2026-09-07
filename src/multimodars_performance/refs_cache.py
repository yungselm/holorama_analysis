"""Per-case cache of the three alignment reference points.

``align_combined`` needs an (aortic, superior, inferior) reference triplet at
the coronary ostium.  0.7.0 derives it automatically from
``discretize_vessel_tree`` (``tree.rca_references[0]``); 0.3.4 has no such
function, and the v0.3.4 example simply hardcoded the three points per case by
hand.

So the 0.7.0 run writes its triplets here and the 0.3.4 run reads them back.
That keeps the timing comparison about compute - picking the points was human
work in 0.3.4, not machine work - and keeps both pipelines aligning against
exactly the same references, so every downstream block stays comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

Triplet = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _read(path: Path) -> dict[str, list]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_references(path: Path, case: str, references) -> None:
    """Merge one case's triplet into the cache, rewriting it immediately.

    Written per case rather than at the end of the run so an interrupted or
    partially failing 0.7.0 run still leaves usable references behind.
    """
    cache = _read(path)
    cache[case] = [[float(coord) for coord in point] for point in references]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(cache.items())), indent=2) + "\n")


def load_references(path: Path, case: str) -> Triplet:
    """Return the cached triplet for *case*, or explain how to produce it."""
    cache = _read(path)
    if case not in cache:
        raise RuntimeError(
            f"No cached alignment references for {case} in {path}. "
            "0.3.4 cannot compute them (no discretize_vessel_tree) - run the "
            "0.7.0 pipeline first to populate the cache."
        )
    return tuple(tuple(point) for point in cache[case])
