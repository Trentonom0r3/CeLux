"""
This test is to check the performance of the VideoReader class with torch frames.
Should be useful inside Github Actions to keep track of the performance of the VideoReader class.
"""

import time
import os
import logging
import requests
import torch
import ffmpy

from requests.exceptions import RequestException

# Nicer prints
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


def getVideoUrl():
    """
    Returns the URL of a public domain video.

    Returns:
        str: The URL of the video.

    URL From:
        https://gist.github.com/jsturgis/3b19447b304616f18657
    """

    videoUrls = [
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    ]
    return videoUrls[0]


def processVideoTorch(videoPath):
    """
    Processes the video to count frames and measure performance.

    Args:
        videoPath (str): The path to the video file.
    """
    try:
        frameCount = 0
        start = time.time()
        with ffmpy.VideoReader(videoPath, as_numpy=False, d_type="uint8") as reader:
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


def processVideoNumPy(videoPath):
    """
    Processes the video to count frames and measure performance.

    Args:
        videoPath (str): The path to the video file.
    """
    try:
        frameCount = 0
        start = time.time()
        # Hardcoded to as_numpy false until fixed
        # Until then, this still decodes on GPU
        with ffmpy.VideoReader(videoPath, as_numpy=True, d_type="uint8") as reader:
            for frame in reader:
                if frameCount == 0:
                    logging.info(f"Frame data: {frame.shape, frame.dtype}")
                    # Just to make sure it's a numpy array
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

    logging.info("Processing video with torch frames")
    processVideoTorch(videoPath)

    print("")

    logging.info("Processing video with numpy frames")
    processVideoNumPy(videoPath)


if __name__ == "__main__":
    main()
