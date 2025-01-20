"""
Compare Celux VideoReader with OpenCV for extracting frames over a given time range.

Usage:
    python test_range_comparison.py --start 5.83917250583917 --end 6.25625625625626
    (adjust times or use defaults)

Requirements:
    - Python 3.x
    - requests (if you want to auto-download the sample video)
    - numpy, cv2 (OpenCV)
    - torch (Celux uses PyTorch)
    - celux

Behavior:
    1) Optionally downloads a test video (BigBuckBunny.mp4) if missing.
    2) Reads frames from both OpenCV and Celux starting at `start_time` until `end_time`.
    3) Displays side-by-side frames in a single OpenCV window for quick comparison.
    4) Logs the number of frames retrieved by each library.
"""

import os
import sys
import cv2
import time
import math
import torch
import argparse
import logging
import requests
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import celux

# Set Celux log level for debugging
celux.set_log_level(celux.LogLevel.off)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def download_video(url: str, output_path: str) -> None:
    """
    Downloads a video file from the given URL and saves it to the output path.

    Args:
        url (str): URL of the video file (HTTP/HTTPS).
        output_path (str): Local file path where the video will be saved.
    """
    if os.path.exists(output_path):
        logging.info(f"Video already exists: {output_path}")
        return

    try:
        logging.info(f"Downloading from {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"Downloaded video to {output_path}")
    except Exception as e:
        logging.error(f"Failed to download video: {e}")
        raise


def get_frames_opencv(video_path: str, start_time: float, end_time: float) -> list:
    """
    Extract frames between start_time and end_time (in seconds) using OpenCV.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        A list of BGR frames (numpy arrays) from OpenCV within [start_time, end_time).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video with OpenCV.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Convert start_time/end_time to approximate frame indexes
    start_msec = start_time * 1000.0
    end_msec = end_time * 1000.0

    logging.info(f"[OpenCV] Using start_time={start_time}, end_time={end_time}, fps={fps}")

    # Seek to the approximate start time (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    frames = []
    while True:
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if current_msec < start_msec:
            # If we haven't reached the start yet, read and discard
            ret = cap.grab()
            if not ret:
                break
            continue

        if current_msec >= end_msec:
            # If we passed the end time, stop
            break

        ret, frame_bgr = cap.read()
        if not ret:
            # No more frames
            break

        frames.append(frame_bgr)

    cap.release()
    logging.info(f"[OpenCV] Retrieved {len(frames)} frames in range [{start_time}, {end_time}).")
    return frames


def get_frames_celux(video_path: str, start_time: float, end_time: float) -> list:
    """
    Extract frames between start_time and end_time (in seconds) using celux.VideoReader.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        A list of frames (numpy arrays in BGR order) from Celux within [start_time, end_time).
    """
    frames = []
    logging.info(f"[Celux] Using start_time={start_time}, end_time={end_time}")

    # We'll set a time range
    # Celux typically sets the range via either set_range(...) or calling with a list/tuple.
    # e.g.  reader.set_range(5.8, 6.2) or reader((5.8, 6.2))
    # We'll do the "call" approach as in your example.

    try:
        with celux.VideoReader(video_path, num_threads=8) as reader:
            # This calls __call__ which sets range, returning the same reader
            #  e.g. `[5.83917250583917, 6.25625625625626]`
            reader((start_time, end_time))

            for frame_idx, frame_tensor in enumerate(reader):
                # The frame is in Torch format (HWC, BGR).
                # Convert to CPU numpy array
                frame_cpu = frame_tensor.cpu().numpy()  # shape is [H, W, 3]
                frames.append(frame_cpu)
    except Exception as e:
        logging.error(f"Celux error: {e}")

    logging.info(f"[Celux] Retrieved {len(frames)} frames in range [{start_time}, {end_time}).")
    return frames

def fit_width(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    """
    Scales the given frame (BGR) to fit within max_width, preserving aspect ratio.
    """
    h, w, _ = frame_bgr.shape
    if w <= max_width:
        return frame_bgr  # No scaling needed

    scale = max_width / float(w)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def display_side_by_side(opencv_frames: list, celux_frames: list, max_width=900) -> None:
    """
    Displays frames side-by-side (OpenCV vs Celux) until one list ends or user presses 'q'.
    Scales each panel to fit within max_width. 
    """
    min_len = min(len(opencv_frames), len(celux_frames))
    logging.info(f"Displaying {min_len} frames side by side. Press 'q' to quit.")

    for i in range(min_len):
        frame_opencv = opencv_frames[i]
        frame_celux = celux_frames[i]

        if frame_opencv is None or frame_celux is None:
            continue

        # Resize both to fit within max_width
        frame_opencv = fit_width(frame_opencv, max_width)
        frame_celux = fit_width(frame_celux, max_width)

        # Make sure both frames have the same height (for nice stacking)
        h1, w1, _ = frame_opencv.shape
        h2, w2, _ = frame_celux.shape

        if h1 != h2:
            # We'll scale the second frame to match the first frame's height
            scale = h1 / float(h2)
            frame_celux = cv2.resize(frame_celux, (int(w2 * scale), h1), interpolation=cv2.INTER_AREA)

        # Combine side by side
        combined = np.hstack((frame_opencv, frame_celux))

        # Label each side
        cv2.putText(combined, "OpenCV", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(
            combined, "Celux", 
            (frame_opencv.shape[1] + 50, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, (255, 0, 0), 2
        )

        cv2.imshow("OpenCV vs Celux (press 'q' to quit)", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()




def main():
    parser = argparse.ArgumentParser(description="Compare Celux VideoReader with OpenCV in a specific time range.")
    parser.add_argument("--start", type=float, default=5.8, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=6.2, help="End time in seconds.")
    parser.add_argument("--video", type=str, default="BigBuckBunny.mp4", help="Local path for the video.")
    args = parser.parse_args()

    # Sample video from GCS
    sample_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

    # Download if missing
    if not os.path.exists(args.video):
        download_video(sample_url, args.video)

    logging.info(f"Comparing frames from {args.start} to {args.end} seconds in {args.video}...")

    # Extract frames using both methods
    t1 = time.time()
    opencv_frames = get_frames_opencv(args.video, args.start, args.end)
    t2 = time.time()
    logging.info(f"OpenCV extraction took {t2 - t1:.3f}s")

    t3 = time.time()
    celux_frames = get_frames_celux(args.video, args.start, args.end)
    t4 = time.time()
    logging.info(f"Celux extraction took {t4 - t3:.3f}s")

    # Display side-by-side
    display_side_by_side(opencv_frames, celux_frames)

    logging.info("Done. You can check visually if Celux is off by many frames.")


if __name__ == "__main__":
    main()
