import os

from importlib.metadata import PackageNotFoundError, version

PATH_TEST_DATA = "C:/Users/ansel/OneDrive/Dokumente/3_Research/data_holorama_test/anomalies"

def run_mutlimodars_pipeline():
    version = get_multimodars_version()
    if version == "0.3.4":
        print("Running multimodars pipeline for version 0.3.4")
    else:
        print("Running multimodars pipeline for version", version)


def get_multimodars_version() -> str:
    try:
        multimodars_version = version("multimodars")
    except PackageNotFoundError:
        raise RuntimeError("multimodars not installed")
    return multimodars_version


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = os.times()[4]
        result = func(*args, **kwargs)
        end_time = os.times()[4]
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
        return result
    return wrapper