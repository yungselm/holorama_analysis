"""Timed CCTA/IVUS fusion pipeline for multimodars 0.7.0.

Registered in mm_perf.PIPELINES, which calls run_case() per NARCO_ folder.
"""

from contextlib import chdir
from pathlib import Path

import multimodars as mm
import trimesh

from multimodars_performance.mm_perf import CaseResult


def run_case(case_path: Path) -> CaseResult:
    result = CaseResult(case_path.name)
    try:
        with chdir(case_path):
            with result.timed("preprocessing"):
                aorta_cl = mm.prepare_centerline(
                    mm.load_centerline("./ao_cl.vtp", name="Aorta"), 
                    spacing_mm=0.5, 
                    smoothing_sigma=1.5
                )
                rca_cl = mm.prepare_centerline(
                    mm.load_centerline("./rca_cl.vtp", name="RCA"), 
                    ref_centerline=aorta_cl, 
                    spacing_mm=0.5, 
                    rm_start_mm=5.0, 
                    smoothing_sigma=1.5
                )
                lca_cl = mm.prepare_centerline(
                    mm.load_centerline("./lca_cl.vtp", name="LCA"), 
                    ref_centerline=aorta_cl, 
                    spacing_mm=0.5, 
                    rm_start_mm=5.0
                )

            with result.timed("labelling"):
                results = mm.label_geometry(
                    path_ccta_geometry="./aortic_root.stl", 
                    centerline_aorta=aorta_cl,
                    centerline_rca=rca_cl, 
                    centerline_lca=lca_cl,
                    bounding_sphere_radius_mm_rca=3.5, 
                    bounding_sphere_radius_mm_lca=3.5,
                    range_mm_takeoff_rca=45.0, 
                    range_mm_takeoff_lca=45.0,
                    step_size_mm=0.5,
                    acute_takeoff_rca=True, 
                    acute_takeoff_lca=False,
                    control_plot=False,
                )
                results = mm.label_branches_pair(rca_cl, lca_cl, results)
            result.n_vertices = len(results["mesh"].vertices)

            with result.timed("discretization"):
                tree = mm.discretize_vessel_tree(
                    aorta_cl, 
                    rca_cl, 
                    lca_cl, 
                    results, 
                    step_size=1.0, 
                    n_points=100,
                    b_spline=True, 
                    bspline_smoothing=5.0, 
                    control_plot=False,
                )

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
                ref_points = tree.rca_references[0]
                aligned, spacing_mm, _ = mm.align_combined(
                    rca_cl.get_branch(0), 
                    rest, ref_points[0], 
                    ref_points[1], 
                    ref_points[2],
                    results["rca_points"], 
                    angle_range_deg=30.0, 
                    watertight=False,
                    align_wall_anomalous=True,
                )
            aorta_cl = aorta_cl.resample(spacing_mm)
            rca_cl = rca_cl.resample(spacing_mm)
            lca_cl = lca_cl.resample(spacing_mm)

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
                updated = mm.remove_labeled_points_from_mesh(
                    results, ["anomalous_points", "proximal_points"], 
                    target_boundaries=2
                )
                stitched = mm.stitch_ccta_to_intravascular(
                    aligned.geom_a, 
                    updated["mesh"], 
                    updated, 
                    prox_start_mode="highest_z", 
                    clamp_overshoot=0.5
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
