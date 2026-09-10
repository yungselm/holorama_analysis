"""Drives the per-version multimodars pipeline benchmarks.

Each supported multimodars version has a module in `PIPELINES` exposing a
`run_case(case_path) -> CaseResult`. This module picks the one matching the
installed version and runs it over every NARCO_ case.
"""

import csv
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

PATH_TEST_DATA = Path(os.environ.get("PATH_TEST_DATA", "C:/Users/ADMIN/OneDrive/Dokumente/3_Research/data_holorama_test/anomalies"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"

# Inputs shared between pipeline versions, kept out of the read-only test-data
# tree.  `rca_references.json` is written by 0.7.1 and read by 0.3.4 (see
# refs_cache); `centerlines_csv` holds the .vtp -> CSV conversion 0.3.4 needs.
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
REFERENCES_PATH = FIXTURES_DIR / "rca_references.json"
CENTERLINE_CSV_DIR = FIXTURES_DIR / "centerlines_csv"

PIPELINES = {
    "0.3.4": "multimodars_performance.multimodars_0_3_4",
    "0.7.1": "multimodars_performance.multimodars_0_7_1",
}

BLOCKS = (
    "preprocessing",
    "labelling",
    "discretization",
    "intravascular_alignment",
    "align_to_centerline",
    "label_anomalous",
    "scaling",
    "scaling_morphing",
    "stitching",
    "post_processing",
)


def get_multimodars_version() -> str:
    try:
        return version("multimodars")
    except PackageNotFoundError:
        raise RuntimeError("multimodars not installed")


@dataclass
class CaseResult:
    """Timings and mesh metrics for a single NARCO_ case — one CSV row."""

    case: str
    seconds: dict[str, float] = field(default_factory=dict)
    n_points: int | None = None
    n_vertices: int | None = None
    error: str = ""

    @contextmanager
    def timed(self, block: str):
        start_time = perf_counter()
        yield
        self.seconds[block] = perf_counter() - start_time

    def as_row(self) -> dict:
        row = {"case": self.case}
        row.update({block: f"{self.seconds[block]:.6f}" if block in self.seconds else "" for block in BLOCKS})
        row["total_seconds"] = f"{sum(self.seconds.values()):.6f}"
        row["n_points"] = self.n_points if self.n_points is not None else ""
        row["n_vertices"] = self.n_vertices if self.n_vertices is not None else ""
        row["error"] = self.error
        return row


def iter_test_cases(path: Path):
    return sorted(
        (case for case in Path(path).iterdir()
         if case.is_dir() and case.name.startswith("NARCO_")
         and case.name.removeprefix("NARCO_").isdigit()),
        key=lambda case: int(case.name.removeprefix("NARCO_")),
    )


def write_results(results: list[CaseResult], output_name: str) -> Path:
    output_path = OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case", *BLOCKS, "total_seconds", "n_points", "n_vertices", "error"]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.as_row() for result in results)
    return output_path


def run_pipeline(path: Path = PATH_TEST_DATA) -> Path:
    """Run the pipeline matching the installed multimodars version over every NARCO_ case."""
    mm_version = get_multimodars_version()
    module_name = PIPELINES.get(mm_version)
    if module_name is None:
        raise SystemExit(f"No timed pipeline for multimodars {mm_version} (have: {', '.join(PIPELINES)})")

    print(f"Running multimodars pipeline for version {mm_version}")
    run_case = import_module(module_name).run_case
    results = []
    for case_path in iter_test_cases(path):
        print(f"Running {case_path.name}")
        results.append(run_case(case_path))

    output_path = write_results(results, f"fusion_{mm_version.replace('.', '_')}.csv")
    print(f"Wrote {len(results)} cases to {output_path}")
    return output_path
