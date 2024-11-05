import pytest
import torch
import json
import logging
from pathlib import Path
import sys
import os
import av  # PyAV
import cv2  # OpenCV

# Adjust the path to include CeLux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import celux_cuda as celux

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths to sample videos
SAMPLE_VIDEO_CUDA = "tests/data/BigBuckBunny.mp4"  # 14315 frames
SAMPLE_VIDEO_CPU = "tests/data/BigBuckBunny.mp4"   # 14315 frames

@pytest.fixture(scope="module")
def video_properties():
    """
    Fixture to obtain video properties once for reuse in benchmarks.
    Also generates total_frames.json for benchmark reporting.
    """
    num_threads = 16
    # CeLux VideoReader properties
    with celux.VideoReader(SAMPLE_VIDEO_CPU, device="cpu", num_threads=num_threads) as reader:
        props_cpu = reader.get_properties()
        props_cpu["num_threads"] = num_threads
        props_cpu["video_size"] = (props_cpu['width'], props_cpu['height'])
    
    with celux.VideoReader(SAMPLE_VIDEO_CUDA, device="cuda", num_threads=num_threads) as reader:
        props_cuda = reader.get_properties()
        props_cuda["num_threads"] = num_threads
        props_cuda["video_size"] = (props_cuda['width'], props_cuda['height'])

    # Calculate total frames for PyAV
    with av.open(SAMPLE_VIDEO_CPU) as container:
        total_frames_pyav = sum(1 for _ in container.decode(video=0))

    # Calculate total frames for OpenCV
    cap = cv2.VideoCapture(SAMPLE_VIDEO_CPU)
    total_frames_opencv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Create total_frames.json mapping
    total_frames = {
        "test_Celux_cpu_benchmark": {
            "total_frames": props_cpu['total_frames'],
            "num_threads": num_threads,
            "video_size": props_cpu["video_size"]
        },
        "test_Celux_cuda_benchmark": {
            "total_frames": props_cuda['total_frames'],
            "num_threads": num_threads,
            "video_size": props_cuda["video_size"]
        },
        "test_pyav_cpu_benchmark": {
            "total_frames": total_frames_pyav,
            "num_threads": 1,  # PyAV single-threaded by default
            "video_size": (props_cpu['width'], props_cpu['height'])
        },
        "test_opencv_cpu_benchmark": {
            "total_frames": total_frames_opencv,
            "num_threads": 1,  # OpenCV single-threaded by default
            "video_size": (props_cpu['width'], props_cpu['height'])
        }
    }

    # Write to total_frames.json
    total_frames_path = Path("tests/benchmarks/total_frames.json")
    with total_frames_path.open("w", encoding='utf-8') as f:
        json.dump(total_frames, f, indent=4)
    
    return {
        "cpu": props_cpu,
        "cuda": props_cuda
    }

# Benchmark tests for CeLux, PyAV, and OpenCV
def test_video_reader_cpu_benchmark(benchmark, video_properties):
    """
    Benchmark the VideoReader on CPU.
    """
    def read_video():
        with celux.VideoReader(SAMPLE_VIDEO_CPU, device="cpu") as reader:
            for frame in reader:
                pass  # Process frame without storing

    benchmark(read_video)

def test_video_reader_cuda_benchmark(benchmark, video_properties):
    """
    Benchmark the VideoReader on CUDA.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for benchmarking.")

    def read_video():
        with celux.VideoReader(SAMPLE_VIDEO_CUDA, device="cuda") as reader:
            for frame in reader:
                pass  # Process frame without storing

    benchmark(read_video)

def test_pyav_cpu_benchmark(benchmark, video_properties):
    """
    Benchmark video decoding using PyAV on CPU.
    """
    def read_video():
        with av.open(SAMPLE_VIDEO_CPU) as container:
            for frame in container.decode(video=0):
                pass  # Process frame without storing

    benchmark(read_video)

def test_opencv_cpu_benchmark(benchmark, video_properties):
    """
    Benchmark video decoding using OpenCV on CPU.
    """
    def read_video():
        cap = cv2.VideoCapture(SAMPLE_VIDEO_CPU)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Process frame without storing
        cap.release()

    benchmark(read_video)
    assert True

# Run pytest benchmarks with JSON output and HTML report
# pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark_results.json --html=benchmark_report.html
