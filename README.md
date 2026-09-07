# holorama-analysis

Per-block timing of the CCTA/IVUS fusion pipeline across multimodars versions,
to quantify what changed between **0.3.4** and **0.7.0**.

Each version has a module under `src/multimodars_performance/` exposing
`run_case(case_path) -> CaseResult`. `mm_perf.run_pipeline()` picks the module
matching the installed multimodars version and runs it over every `NARCO_*`
folder in the test-data directory, writing `output/fusion_<version>.csv`.

## Running

Only one multimodars version can be installed at a time, so the two runs are
sequential — and **0.7.0 must go first** (see below).

```bash
# 1. 0.7.0 — also writes fixtures/rca_references.json
#    pyproject: multimodars[meshlab]==0.7.0
uv sync && uv run python src/main.py

# 2. 0.3.4 — reads those references back
#    pyproject: multimodars[meshlab]==0.3.4
uv sync && uv run python src/main.py
```

Override the test-data location with the `PATH_TEST_DATA` environment variable.

Once both CSVs exist, build the figures and summary tables into `output/plots/`:

```bash
uv run python src/make_plots.py
```

| File | What it shows |
| --- | --- |
| `per_step.png` | one linear panel per block, 0.3.4 vs 0.7.0 |
| `total.png` | end-to-end time, and the per-case sum over the blocks no case failed |
| `*_normalized.png` | the same, divided by input size |
| `summary.csv`, `summary_normalized.csv` | medians, IQRs, ratios and P values |

Box plots with every case overplotted, since with 14 cases spanning 43k-280k
mesh vertices the spread is the result rather than noise around it. Blocks
differ by three orders of magnitude, so they are drawn as small multiples -
each panel linear and starting at zero - instead of sharing one squeezed axis.
P values are two-sided paired Wilcoxon signed-rank tests over the cases both
versions completed; pairs below five are marked underpowered, because the test
cannot reach significance there. Normalization divides by the case's CCTA mesh
vertex count, except `intravascular_alignment`, which divides by its IVUS
contour-point count.

Blocks one version crashed on are labelled in red with the failure count, and
the totals come in two flavours: end to end over the cases that finished in
both versions, and the per-case sum over only the blocks no case failed, so
that one is backed by every case.

## Why 0.7.0 has to run first

`align_combined` needs an (aortic, superior, inferior) reference triplet at the
coronary ostium. 0.7.0 derives it from `discretize_vessel_tree`
(`tree.rca_references[0]`); 0.3.4 has no such function, and the real v0.3.4
workflow hardcoded the three points per case by hand.

So the 0.7.0 run caches its triplets to `fixtures/rca_references.json` and the
0.3.4 run reads them back (`refs_cache.py`). Both versions then align against
identical references, which keeps every downstream block comparable, and the
timing stays a measure of compute rather than of the human work that picking
those points used to be.

## What 0.3.4 cannot do

The 0.3.4 pipeline mirrors the 0.7.0 one block for block. Where 0.3.4 lacks a
capability, the gap is closed in `compat_0_3_4.py` (reimplementing 0.7.0's Rust
semantics in numpy) and marked `# 0.3.4:` at the call site:

| 0.7.0 | 0.3.4 | Stand-in |
| --- | --- | --- |
| `load_centerline` (`.vtp`) | no VTP reader at all | ascii VTK PolyData parsed in Python, cached as CSV **outside** the timed section — 0.3.4 read CSV, so timing a Python XML parse would measure this repo, not the library |
| branch model (`calculate_branches`, `remove_branch_overlap`, `get_branch`) | none — a centerline is one flat polyline | only the main branch survives; 0.7.0's VTP reader sorts branches by descending arc length, so "main" is the *longest* polyline, not the first in the file |
| `prepare_centerline` | none | `prepare_centerline_np`: trim / resample / orient / smooth, following the Rust in `src/types/native/centerline.rs` step for step |
| `label_geometry(bounding_sphere_radius_mm_{rca,lca}, range_mm_takeoff_*, step_size_mm, acute_takeoff_*)` | one shared radius, a **point count** instead of an arc length, no step size | `n_points_intramural = 45 mm / 0.5 mm = 90`; `acute_takeoff_*` was called `anomalous_*` |
| `label_branches_pair` | none | skipped — no branch model, so no `rca_points_main` / `rca_points_side_N` split exists |
| `discretize_vessel_tree` | none | **no counterpart** — the `discretization` block is left unrecorded and is empty in the 0.3.4 CSV |
| `align_combined(..., align_wall_anomalous=True) -> (geom, spacing_mm, rotation)` | no such flag; returns `(geom, resampled_cl)` | `spacing_mm` recomputed from the mean distance between consecutive lumen centroids, which is how 0.7.0 defines it |
| `PyCenterline.resample` | none | `resample_centerline` |
| `remove_labeled_points_from_mesh(target_boundaries=…)` | no such argument | dropped — removal cannot be told how many boundary loops to end up with |
| `stitch_ccta_to_intravascular(clamp_overshoot=…)` | no such argument | dropped |

## Known 0.3.4 failure

`_stitch_boundary_ring` breaks whenever the CCTA boundary ring has more points
than the IV ring (`n_boundary > n_iv`, the latter fixed at 100). `step` becomes
`0`, so every segment past the hundredth computes `mid = n_iv` and the bridging
triangle indexes one vertex past the end:

```
IndexError: index 276 is out of bounds for axis 0 with size 276
```

Expect this to surface as an error in the `stitching` block for some cases.
It is a real 0.3.4 limitation, and the `target_boundaries` and
`clamp_overshoot` arguments added later are what give 0.7.0 control over it.
