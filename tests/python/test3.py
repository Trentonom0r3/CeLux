"""
This verifies the metadata request.

Just to make sure that the metadata request is working as expected.

Expected Values are:

Total Frames: 14315
Width: 1280
Height: 720
Channels: 3
FPS: 24
Duration: +- 596 seconds
Audio: True
"""

import logging
import subprocess
import sys
import os

# Try to import stuff, if not found, pip install them
# Maybe not the best way to do this, but it works for now rather than having random .txt files laying around
try:
    import torch
except ImportError:
    logging.warning("Torch not found. Installing torch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    import torch  # noqa

try:
    import cv2
except ImportError:
    logging.warning("OpenCV not found. Installing opencv-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2  # noqa

try:
    import requests
except ImportError:
    logging.warning("Requests not found. Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests
    from requests.exceptions import RequestException

# Needs to be a wheel file
import ffmpy

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


def getMetadataCV2(videoPath):
    """
    Get metadata from the video file.

    Args:
        videoPath (str): The path to the video file.
    """
    video = cv2.VideoCapture(videoPath)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # channels = int(video.get(cv2.CAP_PROP_CHANNEL))
    fps = video.get(cv2.CAP_PROP_FPS)
    # duration = video.get(cv2.CAP_PROP_POS_MSEC)
    # audio = doesn't seem to be a property in OpenCV

    logging.info("CV2 Metadata:")
    logging.info(f"Total Frames: {totalFrames}")
    logging.info(f"Width: {width}")
    logging.info(f"Height: {height}")
    # logging.info(f"Channels: {channels}")
    logging.info(f"FPS: {fps}")
    # logging.info(f"Duration: {duration}")
    # logging.info(f"Audio: {audio}")

    video.release()

    metadata = {
        "totalFrames": totalFrames,
        "width": width,
        "height": height,
        # "channels": channels,
        "fps": fps,
        # "duration": duration,
        # "audio": audio,
    }

    return metadata


def getMetadataFFMPY(videoPath):
    """
    Get metadata from the video file using FFMPY.
    """

    video = ffmpy.VideoReader(videoPath)

    metadata = video.getProperties()

    logging.info("FFMPY Metadata:")
    logging.info(f"Total Frames: {metadata['totalFrames']}")
    logging.info(f"Width: {metadata['width']}")
    logging.info(f"Height: {metadata['height']}")
    # logging.info(f"Channels: {metadata['channels']}")
    logging.info(f"FPS: {metadata['fps']}")
    logging.info(f"Duration: {metadata['duration']}")

    return metadata


if __name__ == "__main__":
    videoUrl = getVideoUrl()
    videoPath = os.path.join(os.getcwd(), "BigBuckBunny.mp4")

    if not os.path.exists(videoPath):
        downloadVideo(videoUrl, videoPath)
    else:
        logging.info(f"Video already exists at {videoPath}")

    getMetadataCV2(videoPath)

    print("")

    getMetadataFFMPY(videoPath)
