#!/usr/bin/env python
"""
Compare Celux VideoReader with OpenCV for extracting frames over a given time range,
providing thorough testing of various possible time ranges and configurations.

Usage:
    python test_range_comparison.py --start 5.83917250583917 --end 6.25625625625626
    python test_range_comparison.py --video path/to/video.mp4 --start 0 --end 120 --compare

Requirements:
    - Python 3.x
    - requests (optional, only if auto-downloading BigBuckBunny sample is desired)
    - numpy, cv2 (OpenCV)
    - torch (Celux uses PyTorch)
    - celux

Behavior:
    1) Optionally downloads a test video if missing (BigBuckBunny.mp4).
    2) Checks, clamps, and logs the requested time range.
    3) Reads frames from both OpenCV and Celux starting at `start_time` until `end_time`.
    4) Displays side-by-side frames in a single OpenCV window for quick comparison, 
       unless user presses 'q'.
    5) (Optional) Computes and logs a simple numerical comparison (MSE) between frames at each index 
       to check if they closely match.
    6) Logs the number of frames retrieved by each library.
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

# If your local celux is in a parent folder, you can do:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import celux

# Set Celux log level for debugging
celux.set_log_level(celux.LogLevel.info)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def download_video(url: str, output_path: str) -> None:
    """
    Downloads a video file from the given URL and saves it to the output path.

    Args:
        url (str): URL of the video file (HTTP/HTTPS).
        output_path (str): Local file path where the video will be saved.
    """
    if os.path.exists(output_path):
        logging.info(f"Video already exists, skipping download: {output_path}")
        return

    try:
        logging.info(f"Downloading sample video from {url}")
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


def get_video_duration_opencv(video_path: str) -> float:
    """
    Gets the duration of the video (in seconds) using OpenCV.

    Args:
        video_path (str): Path to the video.

    Returns:
        float: Duration of the video in seconds, or 0.0 if unable to open.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps > 0:
        return float(frame_count / fps)
    else:
        return 0.0


def get_frames_opencv(video_path: str, start_time: float, end_time: float) -> list:
    """
    Extract frames between start_time and end_time (in seconds) using OpenCV.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        list of numpy.ndarray: 
            A list of BGR frames (numpy arrays) from OpenCV within [start_time, end_time).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video with OpenCV: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_msec = start_time * 1000.0
    end_msec = end_time * 1000.0

    logging.info(f"[OpenCV] Using start_time={start_time:.4f}, end_time={end_time:.4f}, fps={fps:.2f}")

    # Seek to the approximate start time (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    frames = []
    while True:
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

        # If we haven't reached the start yet, read and discard
        if current_msec < start_msec:
            ret = cap.grab()
            if not ret:
                break
            continue

        # If we've passed the end time, stop
        if current_msec >= end_msec:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            # No more frames
            break

        frames.append(frame_bgr)

    cap.release()
    logging.info(f"[OpenCV] Retrieved {len(frames)} frames in range [{start_time:.4f}, {end_time:.4f}).")
    return frames


def get_frames_celux(video_path: str, start_time: float, end_time: float) -> list:
    """
    Extract frames between start_time and end_time (in seconds) using celux.VideoReader.
    Celux returns frames in RGB; we convert them to BGR here so they match OpenCV's color order.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        A list of frames (numpy arrays in BGR order) from Celux within [start_time, end_time).
    """
    import celux
    frames = []
    logging.info(f"[Celux] Using start_time={start_time:.4f}, end_time={end_time:.4f}")

    try:
        with celux.VideoReader(video_path, num_threads=8) as reader:
            # Set range
            reader((start_time, end_time))

            for frame_idx, frame_tensor in enumerate(reader):
                # Celux returns frames in RGB (H, W, 3) on GPU
                frame_cpu_rgb = frame_tensor.cpu().numpy()  # shape [H, W, 3] in RGB
                frame_cpu_bgr = cv2.cvtColor(frame_cpu_rgb, cv2.COLOR_RGB2BGR)
                frames.append(frame_cpu_bgr)
                
    except Exception as e:
        logging.error(f"[Celux] Error: {e}")

    logging.info(f"[Celux] Retrieved {len(frames)} frames in range [{start_time:.4f}, {end_time:.4f}).")
    return frames

def fit_width(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    """
    Scales the given frame (BGR) to fit within max_width, preserving aspect ratio.

    Args:
        frame_bgr (np.ndarray): The input frame in BGR format.
        max_width (int): The maximum width allowed.

    Returns:
        np.ndarray: The resized (or original) frame.
    """
    h, w, _ = frame_bgr.shape
    if w <= max_width:
        return frame_bgr  # No scaling needed

    scale = max_width / float(w)
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def display_side_by_side(opencv_frames: list, celux_frames: list, max_width=600) -> None:
    """
    Displays frames side-by-side (OpenCV vs Celux) while showing Mean Squared Error (MSE).
    Press any key to advance to the next frame, or 'q' to quit immediately.

    Args:
        opencv_frames (list of np.ndarray): Frames from OpenCV (BGR).
        celux_frames (list of np.ndarray): Frames from Celux (BGR, after conversion).
        max_width (int): Maximum width for each sub-frame display (optional).
    """
    min_len = min(len(opencv_frames), len(celux_frames))
    logging.info(f"Displaying up to {min_len} frame pairs side by side. Press any key to step, 'q' to quit.")

    for i in range(min_len):
        frame_opencv = opencv_frames[i]
        frame_celux = celux_frames[i]

        if frame_opencv is None or frame_celux is None:
            continue

        # Resize both to fit within max_width
        frame_opencv = fit_width(frame_opencv, max_width)
        frame_celux = fit_width(frame_celux, max_width)

        # Match heights for proper side-by-side
        h1, w1, _ = frame_opencv.shape
        h2, w2, _ = frame_celux.shape
        if h1 != h2:
            scale = h1 / float(h2)
            frame_celux = cv2.resize(frame_celux, (int(w2 * scale), h1), interpolation=cv2.INTER_AREA)

        # Compute MSE for this frame
        mse = compute_mean_squared_error(frame_opencv, frame_celux)

        # Combine frames side by side
        combined = np.hstack((frame_opencv, frame_celux))

        # Overlay labels
        cv2.putText(combined, "OpenCV", (50, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.putText(combined, "Celux", (frame_opencv.shape[1] + 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Overlay MSE score in red at the bottom
        mse_text = f"MSE: {mse:.4f}"
        cv2.putText(combined, mse_text, (50, h1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)

        cv2.imshow("OpenCV vs Celux (MSE displayed) - Press any key to step, 'q' to quit", combined)
        key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


def compute_mean_squared_error(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes the mean squared error (MSE) between two images of the same shape.

    Args:
        img1 (np.ndarray): First image (H,W,3).
        img2 (np.ndarray): Second image (H,W,3).

    Returns:
        float: Mean Squared Error value.
    """
    if img1.shape != img2.shape:
        return 1e6  # Large value to indicate mismatched sizes

    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse = np.mean(diff ** 2)
    return mse



def compare_frames_numerically(opencv_frames: list, celux_frames: list) -> None:
    """
    Logs MSE for corresponding frames and calculates an average.

    Args:
        opencv_frames (list of np.ndarray): Frames from OpenCV.
        celux_frames (list of np.ndarray): Frames from Celux.
    """
    min_len = min(len(opencv_frames), len(celux_frames))
    if min_len == 0:
        logging.info("No frames to compare.")
        return

    mses = []
    for i in range(min_len):
        frame_cv = opencv_frames[i]
        frame_clx = celux_frames[i]

        if frame_cv.shape != frame_clx.shape:
            logging.warning(f"Frame {i} shape mismatch: OpenCV={frame_cv.shape}, Celux={frame_clx.shape}")
            continue

        mse = compute_mean_squared_error(frame_cv, frame_clx)
        mses.append(mse)
        logging.info(f"Frame {i}: MSE = {mse:.4f}")

    if mses:
        avg_mse = sum(mses) / len(mses)
        logging.info(f"🔹 Average MSE across frames: {avg_mse:.4f}")
    else:
        logging.info("No valid frames for numerical comparison.")


def clamp_time_range(start_time: float, end_time: float, video_duration: float) -> (float, float):
    """
    Clamps start_time and end_time to valid values within [0, video_duration].

    Args:
        start_time (float): Desired start time in seconds.
        end_time (float): Desired end time in seconds.
        video_duration (float): The total video duration in seconds.

    Returns:
        tuple: (clamped_start, clamped_end)
    """
    clamped_start = max(0.0, min(start_time, video_duration))
    clamped_end = max(0.0, min(end_time, video_duration))

    if clamped_end < clamped_start:
        # Just swap or force them equal if user gave an inverted range
        clamped_start, clamped_end = clamped_end, clamped_start

    return clamped_start, clamped_end


def main():
    parser = argparse.ArgumentParser(description="Compare Celux VideoReader with OpenCV in a specific time range.")
    parser.add_argument("--start", type=float, default=5.8, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=6.2, help="End time in seconds.")
    parser.add_argument("--video", type=str, default="BigBuckBunny.mp4", 
                        help="Local path for the video (MP4, AVI, etc.).")
    args = parser.parse_args()

    # Optionally auto-download a sample if user specifically gave BigBuckBunny.mp4 
    # and it doesn't exist
    sample_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    if args.video.lower().endswith("bigbuckbunny.mp4") and not os.path.exists(args.video):
        download_video(sample_url, args.video)

    # 1) Determine the video duration to clamp the time range
    duration = get_video_duration_opencv(args.video)
    if duration <= 0.0:
        logging.error("Could not determine video duration or open the video. Exiting.")
        sys.exit(1)

    # 2) Clamp start/end times to valid ranges
    start_time, end_time = clamp_time_range(args.start, args.end, duration)
    logging.info(f"Requested time range: [{args.start}, {args.end}]")
    logging.info(f"Clamped time range:   [{start_time}, {end_time}] of total {duration:.2f} seconds")

    # 3) Extract frames with OpenCV
    t1 = time.time()
    opencv_frames = get_frames_opencv(args.video, start_time, end_time)
    t2 = time.time()
    logging.info(f"[OpenCV] Extraction time: {t2 - t1:.3f}s")

    # 4) Extract frames with Celux
    t3 = time.time()
    celux_frames = get_frames_celux(args.video, start_time, end_time)
    t4 = time.time()
    logging.info(f"[Celux] Extraction time: {t4 - t3:.3f}s")

    logging.info(f"OpenCV frames: {len(opencv_frames)} | Celux frames: {len(celux_frames)}")

    # 5) Display frames side-by-side for manual/visual inspection
    if opencv_frames and celux_frames:
        for i in range(min(len(opencv_frames), len(celux_frames))):
            cv2.imwrite(f"opencv_frame_{i}.png", opencv_frames[i])
            cv2.imwrite(f"celux_frame_{i}.png", celux_frames[i])
        #display_side_by_side(opencv_frames, celux_frames, max_width=600)
    else:
        logging.info("Skipping side-by-side display as one or both frame lists are empty.")

    logging.info("Done. Check logs and visual output for discrepancies.")


if __name__ == "__main__":
    main()
