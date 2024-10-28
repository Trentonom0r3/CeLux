import pytest
import torch
import json
import logging
from pathlib import Path
import sys
import os

# Adjust the path to include celux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import celux_cuda as celux

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths to sample videos as strings
SAMPLE_VIDEO_CUDA = "tests/data/BigBuckBunny.mp4"  # 14315 frames
SAMPLE_VIDEO_CPU = "tests/data/BigBuckBunny.mp4"   # 14315 frames

@pytest.fixture(scope="module")
def video_properties():
    """
    Fixture to obtain video properties once for reuse in benchmarks.
    Also generates total_frames.json for benchmark reporting.
    """
    with celux.VideoReader(SAMPLE_VIDEO_CPU, device="cpu") as reader:
        props_cpu = reader.get_properties()
    
    with celux.VideoReader(SAMPLE_VIDEO_CUDA, device="cuda") as reader:
        props_cuda = reader.get_properties()
    
    total_frames = {
        "test_video_reader_cpu_benchmark": props_cpu['total_frames'],
        "test_video_reader_cuda_benchmark": props_cuda['total_frames']
    }

    # Write to total_frames.json
    total_frames_path = Path("tests/benchmarks/total_frames.json")
    with total_frames_path.open("w", encoding='utf-8') as f:
        json.dump(total_frames, f, indent=4)
    
    return {
        "cpu": props_cpu,
        "cuda": props_cuda
    }

def test_video_reader_cpu_benchmark(benchmark, video_properties):
    """
    Benchmark the VideoReader on CPU.
    """
    total_frames = video_properties["cpu"]['total_frames']
    
    def read_video():
        logger.info("Starting VideoReader CPU benchmark.")
        with celux.VideoReader(SAMPLE_VIDEO_CPU, device="cpu") as reader:
            for frame in reader:
                pass  # Process frame without storing
        logger.info("Completed VideoReader CPU benchmark.")
    
    benchmark(read_video)
    assert True  # Dummy assertion to ensure the test doesn't fail

def test_video_reader_cuda_benchmark(benchmark, video_properties):
    """
    Benchmark the VideoReader on CUDA.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for benchmarking.")
    
    total_frames = video_properties["cuda"]['total_frames']
    
    def read_video():
        logger.info("Starting VideoReader CUDA benchmark.")
        with celux.VideoReader(SAMPLE_VIDEO_CUDA, device="cuda") as reader:
            for frame in reader:
                pass  # Process frame without storing
        logger.info("Completed VideoReader CUDA benchmark.")
    
    benchmark(read_video)
    assert True  # Dummy assertion to ensure the test doesn't fail

#pytest tests\benchmarks\ --benchmark-only --benchmark-json=benchmark_results.json --html=benchmark_report.html
