"""
This script checks the performance of the VideoReader class with torch and numpy frames.
It provides real-time visual confirmation of frames using OpenCV and writes frames to an output file.
"""

import time
import argparse
import logging
import requests
import sys
import os
import cv2  # For visual confirmation
import threading
from queue import Queue, Empty

# Adjust the path to include ffmpy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import ffmpy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_video(url, output_path):
    """
    Downloads a video from the specified URL.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as file:
            logging.info(f"Downloading {url} to {output_path}")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    except requests.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise

def frame_writer_thread(output_path, frame_queue, stop_event, as_numpy):
    """
    Writes frames from the queue to the output video file in a separate thread.
    """
    try:
        with ffmpy.VideoWriter(output_path, 1920, 1080, 24.0, as_numpy=as_numpy) as writer:
            while not stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=1)  # Timeout prevents blocking on join
                    if frame is None:
                        break
                    writer(frame)
                except Empty:
                    continue
    except Exception as e:
        logging.error(f"Error in writer thread: {e}")

def display_frames(video_path, frame_queue, stop_event, as_numpy):
    """
    Reads and displays frames in real-time using OpenCV.
    """
    try:
        with ffmpy.VideoReader(video_path, as_numpy=as_numpy, d_type="uint8") as reader:
            for frame in reader:
                if stop_event.is_set():
                    break

                # Display the frame
                cv2.imshow("Video Frame", frame.cpu().numpy() if not as_numpy else frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Stopping early - 'q' pressed.")
                    stop_event.set()
                    break

                # Put the frame in the queue for writing
                frame_queue.put(frame)

            frame_queue.put(None)  # Signal that reading is complete

    except Exception as e:
        logging.error(f"Error in display thread: {e}")
    finally:
        cv2.destroyAllWindows()

def threaded_processing(video_path, output_path, as_numpy):
    """
    Launches separate threads for frame display and writing.
    """
    frame_queue = Queue(maxsize=10)  # Limit queue size to control memory usage
    stop_event = threading.Event()

    # Start the writer thread if saving is enabled
    writer_thread = None
    if output_path:
        writer_thread = threading.Thread(
            target=frame_writer_thread, args=(output_path, frame_queue, stop_event, as_numpy)
        )
        writer_thread.start()

    # Start the display thread
    display_thread = threading.Thread(
        target=display_frames, args=(video_path, frame_queue, stop_event, as_numpy)
    )
    display_thread.start()

    logging.info("Starting threaded video processing...")
    start = time.time()

    # Wait for the display thread to finish
    display_thread.join()

    # If writer thread is running, wait for it to finish
    if writer_thread:
        writer_thread.join()

    end = time.time()
    logging.info(f"Total time for threaded processing: {end - start:.2f} seconds")

def main(args):
    if args.mode == "lite":
        video_url = r"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
        video_path = os.path.join(os.getcwd(), "ForBiggerBlazes.mp4")
    else:
        video_url = r"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        video_path = os.path.join(os.getcwd(), "BigBuckBunny.mp4")

    if not os.path.exists(video_path):
        download_video(video_url, video_path)
    else:
        logging.info(f"Video already exists at {video_path}")

    output_path = "./output.mp4" if args.save_output else None

    logging.info(f"Running threaded processing with as_numpy={args.as_numpy}...")
    threaded_processing(video_path, output_path, args.as_numpy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time video processing with threaded display and writing.")
    parser.add_argument(
        "--mode",
        type=str,
        default="lite",
        choices=["lite", "full"],
        help="Choose 'lite' for GitHub Actions testing or 'full' for local testing."
    )
    parser.add_argument(
        "--as-numpy",
        action="store_true",
        default=False,
        help="Use this flag to enable processing with NumPy arrays instead of torch tensors."
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Enable this flag to save the processed video to an output file."
    )
    args = parser.parse_args()

    main(args)
