import numpy as np
import pandas as pd

import multimodars_performance.mm_perf

def main():
    # stage 1: time tracking of different modules
    print("Hello, World!")
    multimodars_performance.mm_perf.run_mutlimodars_pipeline()

if __name__ == "__main__":
    main()