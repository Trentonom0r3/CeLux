"""
This script checks the performance of the VideoReader class with torch and numpy frames.
It provides real-time visual confirmation of frames using OpenCV, can write output to a video file,
and includes a frame range verification test against OpenCV's VideoCapture.
"""

import time
import argparse
import logging
import requests
import sys
import os
import cv2  # For visual confirmation
import numpy as np

# Adjust the path to include celux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import celux_cuda as celux  # Assuming celux_cuda contains VideoReader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_video(url, output_path):
    """
    Downloads a video from the specified URL.

    Args:
        url (str): The URL of the video.
        output_path (str): The local path to save the video.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as file:
            logging.info(f"Downloading {url} to {output_path}")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        logging.info(f"Download completed: {output_path}")
    except requests.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise

def read_frames_with_cv2(video_path, start_frame, end_frame):
    """
    Reads frames from start_frame to end_frame using OpenCV.

    Args:
        video_path (str): Path to the video file.
        start_frame (int): Starting frame index (inclusive).
        end_frame (int): Ending frame index (exclusive).

    Returns:
        list of numpy.ndarray: List of frames read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return []

    frames = []
    current_frame = 0

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Reached end of video at frame {current_frame}.")
            break

        if start_frame <= current_frame < end_frame:
            frames.append(frame.copy())

        current_frame += 1

    cap.release()
    logging.info(f"Read {len(frames)} frames with OpenCV from {start_frame} to {end_frame-1}.")
    return frames

def read_frames_with_video_reader(video_path, frame_range):
    """
    Reads frames using VideoReader for the specified frame range.

    Args:
        video_path (str): Path to the video file.
        frame_range (list or tuple): [start_frame, end_frame) indices.

    Returns:
        list of numpy.ndarray: List of frames read.
    """
    start_frame, end_frame = frame_range
    frames = []
    try:
        with celux.VideoReader(video_path, device="cuda", d_type="uint8")([start_frame, end_frame]) as reader:
            for frame in reader:
                # Assuming frame is a torch tensor; convert to numpy
                frame_np = frame.cpu().numpy()
                # Convert from RGB to BGR to match OpenCV's format
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
    except Exception as e:
        logging.error(f"Error reading frames with VideoReader: {e}")
        return []

    logging.info(f"Read {len(frames)} frames with VideoReader from {start_frame} to {end_frame-1}.")
    return frames

def combine_frames_side_by_side(frame1, frame2):
    """
    Combines two frames side by side for comparison.

    Args:
        frame1 (numpy.ndarray): First frame.
        frame2 (numpy.ndarray): Second frame.

    Returns:
        numpy.ndarray: Combined frame.
    """
    # Ensure both frames have the same height
    if frame1.shape[0] != frame2.shape[0]:
        # Resize frames to have the same height
        height = min(frame1.shape[0], frame2.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))

    combined = cv2.hconcat([frame1, frame2])
    return combined

def compare_frames(frames_cv2, frames_vr):
    """
    Compares two lists of frames and displays them side by side if mismatches occur.

    Args:
        frames_cv2 (list of numpy.ndarray): Frames read with OpenCV.
        frames_vr (list of numpy.ndarray): Frames read with VideoReader.

    Returns:
        bool: True if all frames match, False otherwise.
    """
    if len(frames_cv2) != len(frames_vr):
        logging.error(f"Number of frames mismatch: OpenCV={len(frames_cv2)}, VideoReader={len(frames_vr)}")
        return False

    for idx, (frame_cv2, frame_vr) in enumerate(zip(frames_cv2, frames_vr)):
        if frame_cv2.shape != frame_vr.shape:
            logging.error(f"Frame {idx} shape mismatch: OpenCV={frame_cv2.shape}, VideoReader={frame_vr.shape}")
            # Display the mismatched frames
            combined = combine_frames_side_by_side(frame_cv2, frame_vr)
            cv2.imshow("Frame Shape Mismatch", combined)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows()
            return False
        if not np.array_equal(frame_cv2, frame_vr):
            # Display the differing frames side by side
            combined = combine_frames_side_by_side(frame_cv2, frame_vr)
            cv2.imshow(f"Frame Content Mismatch at Frame {idx}", combined)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows()

            # Log the maximum difference
            diff = np.abs(frame_cv2.astype(int) - frame_vr.astype(int))
            max_diff = diff.max()
            logging.error(f"Frame {idx} content mismatch: Max pixel difference={max_diff}")
            return False

    logging.info("All compared frames match between OpenCV and VideoReader.")
    return True

def process_video_with_visualization(video_path, output_path=None):
    """
    Processes the video, showing frames in real-time and optionally writing to output.

    Args:
        video_path (str): The path to the input video file.
        output_path (str, optional): The path to save the output video. Defaults to None.
    """
    try:
        frame_count = 0
        start = time.time()

        with celux.VideoReader(video_path, device="cuda", d_type="uint8") as reader:
            writer = None
            if output_path:
                # Fetch video properties for the writer
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logging.error(f"Cannot open video file to get properties: {video_path}")
                    return
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                writer = celux.VideoWriter(output_path, width, height, fps, device="cuda")
                logging.info(f"Initialized VideoWriter to save output at {output_path}")

            for frame in reader:
                # Display the frame using OpenCV
                frame_np = frame.cpu().numpy()
                # Convert from RGB to BGR
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imshow("Video Frame", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Stopping early - 'q' pressed.")
                    break

                # Write frame to output if writer is enabled
                if writer:
                    writer(frame)

                frame_count += 1

            if writer:
                writer.close()
                logging.info(f"VideoWriter closed and saved output to {output_path}")

        end = time.time()
        logging.info(f"Time taken: {end - start} seconds")
        logging.info(f"Total Frames: {frame_count}")
        logging.info(f"FPS: {frame_count / (end - start)}")

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise
    finally:
        # Ensure OpenCV windows are closed properly
        cv2.destroyAllWindows()

def test_frame_range(video_path, start_frame, end_frame):
    """
    Tests whether VideoReader correctly reads the specified frame range by comparing with OpenCV.

    Args:
        video_path (str): Path to the video file.
        start_frame (int): Starting frame index (inclusive).
        end_frame (int): Ending frame index (exclusive).

    Returns:
        bool: True if frames match, False otherwise.
    """
    logging.info(f"Starting frame range test from {start_frame} to {end_frame-1}")
    # Read frames using OpenCV
    frames_cv2 = read_frames_with_cv2(video_path, start_frame, end_frame)

    # Read frames using VideoReader
    frames_vr = read_frames_with_video_reader(video_path, [start_frame, end_frame])

    # Compare the frames
    match = compare_frames(frames_cv2, frames_vr)
    if match:
        logging.info("Frame range test passed: VideoReader matches OpenCV.")
    else:
        logging.error("Frame range test failed: Discrepancies found.")
    return match

def main(args):
    # Determine video URLs and paths based on mode
    if args.mode == "lite":
        video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
        video_filename = "ForBiggerBlazes.mp4"
    else:
        video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        video_filename = "BigBuckBunny.mp4"

    video_path = os.path.join(os.getcwd(), video_filename)

    # Download video if it does not exist
    if not os.path.exists(video_path):
        logging.info(f"Video not found at {video_path}. Initiating download.")
        download_video(video_url, video_path)
    else:
        logging.info(f"Video already exists at {video_path}")

    # Perform frame range test by default
    # Define the frame range to test
    start_frame = 10
    end_frame = 21  # Exclusive; to include frame 20
    test_result = test_frame_range(video_path, start_frame, end_frame)
    if test_result:
        logging.info("Test Passed: VideoReader frame range matches OpenCV.")
    else:
        logging.error("Test Failed: Frame ranges do not match.")

    # Optional: Process video with visualization and optional output saving
    if args.process_video:
        logging.info("Processing video with visualization")
        output_path = "./output.mp4" if args.save_output else None
        process_video_with_visualization(video_path, output_path)

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Video processing script with visualization, optional output saving, and frame range testing.")

    parser.add_argument(
        "--mode",
        type=str,
        default="lite",
        choices=["lite", "full"],
        help="Choose 'lite' for GitHub Actions testing or 'full' for local testing."
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Enable this flag to save the processed video to an output file."
    )
    parser.add_argument(
        "--process-video",
        action="store_true",
        help="Enable this flag to process the video with visualization."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
