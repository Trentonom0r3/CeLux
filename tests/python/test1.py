"""
This script checks the performance of the VideoReader class with torch and numpy frames.
It provides real-time visual confirmation of frames using OpenCV and can write output to a video file.
"""

import time
import argparse
import logging
import requests
import sys
import os
import cv2  # For visual confirmation

# Adjust the path to include ffmpy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import ffmpy

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
    except requests.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise

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

        with ffmpy.VideoReader(video_path, as_numpy=True, d_type="uint8") as reader:
            writer = None
            if output_path:
                writer = ffmpy.VideoWriter(output_path, 1920, 1080, 24.0, as_numpy=True)

            for frame in reader:
                # Display the frame using OpenCV
                cv2.imshow("Video Frame", frame.numpy())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Stopping early - 'q' pressed.")
                    break

                # Write frame to output if writer is enabled
                if writer:
                    writer(frame)

                frame_count += 1

            if writer:
                writer.close()

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
    logging.info("Processing video with visualization")
    process_video_with_visualization(video_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing script with visualization and optional output saving.")
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
    args = parser.parse_args()

    main(args)
