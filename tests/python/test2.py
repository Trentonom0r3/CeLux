"""
This test is to check the performance of the VideoReader class with torch frames.
Should be useful inside Github Actions to keep track of the performance of the VideoReader class.
"""

import time
import sys
import subprocess
import os
import logging

# Nicer prints
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Try to import requests, if not found install it
try:
    import requests
except ImportError:
    logging.warning("Requests not found. Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests
    from requests.exceptions import RequestException


# Try to import torch, if not found install it
try:
    import torch
except ImportError:
    logging.warning("Torch not found. Installing torch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    import torch

# Importing at the bottom so FFMPY won't cry about it
import ffmpy


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


def getVideoUrl():
    """
    Returns the URL of a public domain video.

    Returns:
        str: The URL of the video.
    """
    videoUrls = [
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    ]
    return videoUrls[0]


def processVideo(videoPath):
    """
    Processes the video to count frames and measure performance.

    Args:
        videoPath (str): The path to the video file.
    """
    try:
        frameCount = 0
        start = time.time()
        with ffmpy.VideoReader(videoPath, as_numpy=False, dtype="uint8") as reader:
            for frame in reader:
                if frameCount == 0:
                    logging.info(
                        f"Frame data: {frame.shape, frame.dtype, frame.device}"
                    )
                frameCount += 1
        end = time.time()
        logging.info(f"Time taken: {end-start} seconds")
        logging.info(f"Total Frames: {frameCount}")
        logging.info(f"FPS: {frameCount/(end-start)}")
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise


def main():
    videoUrl = getVideoUrl()
    videoPath = os.path.join(os.getcwd(), "BigBuckBunny.mp4")

    if not os.path.exists(videoPath):
        downloadVideo(videoUrl, videoPath)
    else:
        logging.info(f"Video already exists at {videoPath}")

    processVideo(videoPath)


if __name__ == "__main__":
    main()
