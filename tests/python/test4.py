"""
This test checks the performance of the VideoReader class with torch and numpy frames.
It uses threading and multiprocessing to compare speeds, useful for benchmarking in GitHub Actions.
"""

import time
import argparse
import logging
import requests
import sys
import os
import threading
import queue
from multiprocessing import Process, Queue, Event

# Adjust the import path for ffmpy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import ffmpy  # Assuming ffmpy has VideoReader and VideoWriter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_video(url, output_path):
    """Downloads a video from the URL to the given output path."""
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

def frame_reader(video_path, frame_queue, stop_event, as_numpy):
    """Reads frames from the video and pushes them into a queue."""
    try:
        with ffmpy.VideoReader(video_path, as_numpy=as_numpy, d_type="uint8") as reader:
            for frame in reader:
                if stop_event.is_set():
                    break
                frame_queue.put(frame)
        frame_queue.put(None)  # Signal that reading is complete
    except Exception as e:
        logging.error(f"Error in frame reader: {e}")
        raise

def frame_processor(frame_queue, stop_event):
    """Processes frames from the queue."""
    frame_count = 0
    start = time.time()
    try:
        while not stop_event.is_set():
            frame = frame_queue.get()
            if frame is None:
                break  # End of frames
            frame_count += 1
    except Exception as e:
        logging.error(f"Error in frame processor: {e}")
    finally:
        end = time.time()
        logging.info(f"Processed {frame_count} frames in {end - start:.2f} seconds")
        logging.info(f"FPS: {frame_count / (end - start):.2f}")

def threaded_processing(video_path, as_numpy):
    """Runs frame reading and processing using threads."""
    frame_queue = queue.Queue(maxsize=10)  # Limit queue size
    stop_event = threading.Event()

    reader_thread = threading.Thread(target=frame_reader, args=(video_path, frame_queue, stop_event, as_numpy))
    processor_thread = threading.Thread(target=frame_processor, args=(frame_queue, stop_event))

    logging.info("Starting threaded processing...")
    start = time.time()

    reader_thread.start()
    processor_thread.start()

    reader_thread.join()
    processor_thread.join()

    end = time.time()
    logging.info(f"Total time for threaded processing: {end - start:.2f} seconds")

def multiprocessing_processing(video_path, as_numpy):
    """Runs frame reading and processing using multiprocessing."""
    frame_queue = Queue(maxsize=10)  # Use multiprocessing Queue
    stop_event = Event()

    reader_process = Process(target=frame_reader, args=(video_path, frame_queue, stop_event, as_numpy))
    processor_process = Process(target=frame_processor, args=(frame_queue, stop_event))

    logging.info("Starting multiprocessing...")
    start = time.time()

    reader_process.start()
    processor_process.start()

    reader_process.join()
    processor_process.join()

    end = time.time()
    logging.info(f"Total time for multiprocessing: {end - start:.2f} seconds")

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

    logging.info(f"Running threaded processing with as_numpy={args.as_numpy}...")
    threaded_processing(video_path, args.as_numpy)

    logging.info(f"Running multiprocessing processing with as_numpy={args.as_numpy}...")
    multiprocessing_processing(video_path, args.as_numpy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threaded and multiprocessing video processing performance test.")
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
    args = parser.parse_args()

    main(args)
