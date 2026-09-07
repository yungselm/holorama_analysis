from importlib.metadata import PackageNotFoundError, version


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