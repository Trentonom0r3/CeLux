"""
This test is designed to help determine the optimal buffer size for the VideoReader class.
It measures the performance (FPS) of the VideoReader with various buffer sizes.
"""

import time
import argparse
import logging
import requests
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import celux_cuda as celux

from requests.exceptions import RequestException

# Configure logging for nicer output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def downloadVideo(url, outputPath):
    """
    Downloads a video from the given URL to the specified output path.

    Args:
        url (str): The URL of the video to download.
        outputPath (str): The path where the video will be saved.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(outputPath, "wb") as file:
            logging.info(f"Downloading {url} to {outputPath}")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    except RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise

def processVideo(videoPath, device, buffer_size):
    """
    Processes the video to count frames and measure performance.

    Args:
        videoPath (str): The path to the video file.
        device (str): The device to use ('cuda' or 'cpu').
        buffer_size (int): The buffer size to use.

    Returns:
        float: The frames per second achieved during processing.
    """
    try:
        frameCount = 0
        start = time.time()
        with celux.VideoReader(videoPath, device=device, d_type="uint8", buffer_size=buffer_size) as reader:
            for frame in reader:
                if frameCount == 0:
                    logging.info(f"Frame data: {frame.shape}, {frame.dtype}, {frame.device}")
                frameCount += 1
        end = time.time()
        total_time = end - start
        fps = frameCount / total_time
        logging.info(f"Device: {device.upper()}, Buffer Size: {buffer_size}")
        logging.info(f"Time taken: {total_time:.4f} seconds")
        logging.info(f"Total Frames: {frameCount}")
        logging.info(f"FPS: {fps:.2f}")
        return fps
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise

def main(args):
    if args.mode == "lite":
        videoUrl = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
        videoPath = os.path.join(os.getcwd(), "ForBiggerBlazes.mp4")
    else:
        videoUrl = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        videoPath = os.path.join(os.getcwd(), "BigBuckBunny.mp4")

    if not os.path.exists(videoPath):
        downloadVideo(videoUrl, videoPath)
    else:
        logging.info(f"Video already exists at {videoPath}")

    # Define the buffer sizes to test
    buffer_sizes = [1, 2, 5, 10, 20, 50, 100]

    # Test on CUDA device
    device = "cuda"
    logging.info(f"Processing video with {device.upper()}")
    fps_results_cuda = {}
    for buffer_size in buffer_sizes:
        logging.info(f"Testing buffer size: {buffer_size}")
        fps = processVideo(videoPath, device=device, buffer_size=buffer_size)
        fps_results_cuda[buffer_size] = fps

    print("")

    # Test on CPU device
    device = "cpu"
    logging.info(f"Processing video with {device.upper()}")
    fps_results_cpu = {}
    for buffer_size in buffer_sizes:
        logging.info(f"Testing buffer size: {buffer_size}")
        fps = processVideo(videoPath, device=device, buffer_size=buffer_size)
        fps_results_cpu[buffer_size] = fps

    # Output the results
    print("\nFPS Results for CUDA:")
    for buffer_size in buffer_sizes:
        fps = fps_results_cuda.get(buffer_size, 0)
        print(f"Buffer Size: {buffer_size}, FPS: {fps:.2f}")

    print("\nFPS Results for CPU:")
    for buffer_size in buffer_sizes:
        fps = fps_results_cpu.get(buffer_size, 0)
        print(f"Buffer Size: {buffer_size}, FPS: {fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add 'lite' for GitHub Actions and 'full' for local testing
    parser.add_argument(
        "--mode",
        type=str,
        default="lite",
        help="The mode to run the test in. 'lite' for GitHub Actions and 'full' for local testing.",
        choices=["lite", "full"],
    )
    args = parser.parse_args()

    main(args)
