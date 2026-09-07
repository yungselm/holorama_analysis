"""Timed CCTA/IVUS fusion pipeline for multimodars 0.3.4.

Registered in mm_perf.PIPELINES, which calls run_case() per NARCO_ folder.

This mirrors the 0.7.0 pipeline block for block, so the two CSVs line up.
Where 0.3.4 lacks a capability the 0.7.0 pipeline relies on, the gap is closed
in `compat_0_3_4` (centerline reading and preparation) or `refs_cache`
(alignment reference points), and every such point is marked `# 0.3.4:` below.
The shape of those gaps is the result being measured, so they are worked
around rather than papered over:

* `discretization` has no counterpart at all - 0.3.4 has no
  `discretize_vessel_tree` - so that block is left unrecorded and shows up
  empty in the CSV.
* `preprocessing` does in Python what 0.7.0 does in Rust, which is why it is
  the block expected to move the most.

Reference: the v0.3.4 `examples/stitching.py` in multimoda-rs, which is the
same workflow with the reference points hardcoded per case by hand.
"""

from contextlib import chdir
from pathlib import Path

import multimodars as mm
import numpy as np
import trimesh

from multimodars_performance.compat_0_3_4 import (
    ensure_centerline_csvs,
    prepare_centerline_np,
    resample_centerline,
)
from multimodars_performance.mm_perf import CENTERLINE_CSV_DIR, REFERENCES_PATH, CaseResult
from multimodars_performance.refs_cache import load_references

# 0.7.0 asks for an arc length in mm (range_mm_takeoff_rca=45.0, walked at
# step_size_mm=0.5); 0.3.4 counts centerline points instead. At the 0.5 mm
# spacing both pipelines resample to, 45 mm is 90 points.
N_POINTS_INTRAMURAL = int(45.0 / 0.5)


def _frame_spacing_mm(frames) -> float:
    """Mean distance between consecutive lumen centroids.

    0.7.0's `align_combined` returns this as `spacing_mm` so the remaining
    centerlines can be resampled to match the frame spacing. 0.3.4 returns only
    the resampled centerline, so the same number is derived here from the
    aligned frames, which is how 0.7.0 defines it.
    """
    centroids = np.array([frame.lumen.centroid for frame in frames], dtype=float)
    return float(np.linalg.norm(np.diff(centroids, axis=0), axis=1).mean())


def run_case(case_path: Path) -> CaseResult:
    result = CaseResult(case_path.name)
    try:
        # 0.3.4: no VTP reader. Convert once, outside the timed section - 0.3.4
        # read CSV, so timing a pure-Python XML parse would measure this repo
        # rather than the library.
        csv_paths = ensure_centerline_csvs(case_path, CENTERLINE_CSV_DIR)
        # 0.3.4: no discretize_vessel_tree, so the alignment reference triplet
        # cannot be computed; the v0.3.4 workflow hardcoded it per case. Reuse
        # the triplet the 0.7.0 run cached, so both align against the same one.
        ref_points = load_references(REFERENCES_PATH, case_path.name)

        with chdir(case_path):
            with result.timed("preprocessing"):
                # 0.3.4: no load_centerline/prepare_centerline. Read the CSVs as
                # the v0.3.4 example did, then trim/resample/orient/smooth in
                # numpy to match what 0.7.0 does natively.
                aorta_pts = prepare_centerline_np(
                    np.genfromtxt(csv_paths["ao"], delimiter=","),
                    spacing_mm=0.5,
                    smooth_sigma=1.5,
                )
                rca_pts = prepare_centerline_np(
                    np.genfromtxt(csv_paths["rca"], delimiter=","),
                    reference=aorta_pts,
                    spacing_mm=0.5,
                    rm_start_mm=5.0,
                    smooth_sigma=1.5,
                )
                lca_pts = prepare_centerline_np(
                    np.genfromtxt(csv_paths["lca"], delimiter=","),
                    reference=aorta_pts,
                    spacing_mm=0.5,
                    rm_start_mm=5.0,
                )
                aorta_cl = mm.numpy_to_centerline(aorta_pts)
                rca_cl = mm.numpy_to_centerline(rca_pts)
                lca_cl = mm.numpy_to_centerline(lca_pts)

            with result.timed("labelling"):
                # 0.3.4: one shared bounding-sphere radius instead of one per
                # vessel, a point count instead of an arc length for the takeoff
                # range, and no step size. anomalous_* is what 0.7.0 renamed to
                # acute_takeoff_*.
                results, _ = mm.label_geometry(
                    path_ccta_geometry="./aortic_root.stl",
                    path_centerline_aorta=aorta_cl,
                    path_centerline_rca=rca_cl,
                    path_centerline_lca=lca_cl,
                    bounding_sphere_radius_mm=3.5,
                    n_points_intramural=N_POINTS_INTRAMURAL,
                    anomalous_rca=True,
                    anomalous_lca=False,
                    control_plot=False,
                )
                # 0.3.4: no label_branches_pair - there is no branch model, so
                # no rca_points_main / rca_points_side_N split to produce.
            result.n_vertices = len(results["mesh"].vertices)

            # 0.3.4: no discretize_vessel_tree, and no PyDiscretizedVesselTree to
            # hold the result. The whole discretization block has no counterpart,
            # so it stays unrecorded rather than being faked.

            with result.timed("intravascular_alignment"):
                rest, _ = mm.from_file_singlepair(
                    input_path="ivus_rest",
                    labels=["aligned_dia", "aligned_sys"],
                    step_rotation_deg=0.1,
                    sample_size=200,
                    write_obj=False,
                )
            result.n_points = len(rest.geom_a.frames) * len(rest.geom_a.frames[0].lumen.points)

            with result.timed("align_to_centerline"):
                # 0.3.4: no branch model, so the whole (single-branch) centerline
                # stands in for get_branch(0); no align_wall_anomalous; and the
                # return is (geometry, resampled_centerline), not
                # (geometry, spacing_mm, total_rotation_deg).
                aligned, _ = mm.align_combined(
                    rca_cl,
                    rest,
                    ref_points[0],
                    ref_points[1],
                    ref_points[2],
                    results["rca_points"],
                    angle_range_deg=30.0,
                    watertight=False,
                )
            # 0.3.4: PyCenterline has no resample(), so redo it in numpy at the
            # spacing 0.7.0 would have returned.
            spacing_mm = _frame_spacing_mm(aligned.geom_a.frames)
            aorta_cl = mm.numpy_to_centerline(resample_centerline(aorta_pts, spacing_mm))
            rca_cl = mm.numpy_to_centerline(resample_centerline(rca_pts, spacing_mm))
            lca_cl = mm.numpy_to_centerline(resample_centerline(lca_pts, spacing_mm))

            with result.timed("label_anomalous"):
                results = mm.label_anomalous_region(
                    centerline=rca_cl,
                    frames=aligned.geom_a.frames,
                    results=results,
                    results_key="rca_points",
                    debug_plot=False
                )

            with result.timed("scaling"):
                prox_scaling, distal_scaling = mm.find_distal_and_proximal_scaling(
                    frames=aligned.geom_a.frames,
                    centerline=rca_cl,
                    results=results
                )
                aortic_scaling = mm.find_aorta_scaling(
                    frames=aligned.geom_a.frames,
                    cl_aorta=aorta_cl,
                    results=results
                )

            with result.timed("scaling_morphing"):
                for region_points, centerline, adjustment in (
                    (results["distal_points"], rca_cl, distal_scaling),
                    (results["aorta_points"] + results["rca_removed_points"], aorta_cl, aortic_scaling),
                    (results["proximal_points"], rca_cl, prox_scaling),
                ):
                    scaled = mm.scale_region_centerline_morphing(
                        mesh=results["mesh"],
                        region_points=region_points,
                        centerline=centerline,
                        diameter_adjustment_mm=adjustment
                    )
                    results = mm.sync_results_to_mesh(results, results["mesh"], scaled)

            with result.timed("stitching"):
                # 0.3.4: no target_boundaries - removal cannot be told how many
                # boundary loops the result should end up with.
                updated = mm.remove_labeled_points_from_mesh(
                    results, ["anomalous_points", "proximal_points"]
                )
                # 0.3.4: no clamp_overshoot on the stitch.
                stitched = mm.stitch_ccta_to_intravascular(
                    aligned.geom_a,
                    updated["mesh"],
                    updated,
                    prox_start_mode="highest_z"
                )

            with result.timed("post_processing"):
                remeshed = mm.fix_and_remesh_stitched_mesh(
                    stitched["mesh"],
                    target_edge_length_mm=0.5,
                    verbose=True
                )
                trimesh.smoothing.filter_taubin(remeshed, lamb=0.6)
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
        print(f"  failed after {len(result.seconds)}/10 blocks: {result.error}")
    return result
