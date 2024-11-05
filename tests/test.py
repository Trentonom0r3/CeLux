"""
This script tests the color similarity between Celux and OpenCV by processing video frames using both libraries.
It computes the Delta E (ΔE) value for each frame to quantify color differences and displays the frames scaled to fit within the window.
"""

import time
import argparse
import logging
import numpy as np
import requests
import sys
import os
import cv2
import torch

# Adjust the path to include celux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import celux_cuda as celux

celux.set_log_level(celux.LogLevel.debug)

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
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    file.write(chunk)
    except requests.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise

def calculate_delta_e(frame1_lab, frame2_lab):
    """
    Calculates the Delta E (ΔE) between two LAB frames.

    Args:
        frame1_lab (np.ndarray): First frame in LAB color space.
        frame2_lab (np.ndarray): Second frame in LAB color space.

    Returns:
        float: The average Delta E value for the frame.
    """
    delta_e = np.linalg.norm(frame1_lab.astype(float) - frame2_lab.astype(float), axis=2)
    average_delta_e = np.mean(delta_e)
    return average_delta_e

def resize_frames(frame1, frame2, max_width=1920, max_height=1080):
    """
    Resizes two frames proportionally to fit within the specified maximum dimensions.

    Args:
        frame1 (np.ndarray): First frame (e.g., Celux).
        frame2 (np.ndarray): Second frame (e.g., OpenCV).
        max_width (int, optional): Maximum width of the combined frame. Defaults to 1920.
        max_height (int, optional): Maximum height of the combined frame. Defaults to 1080.

    Returns:
        tuple: Resized frame1 and frame2.
    """
    # Determine the combined width and the maximum height
    combined_width = frame1.shape[1] + frame2.shape[1]
    combined_height = max(frame1.shape[0], frame2.shape[0])

    # Calculate scaling factor based on width
    scale_width = min(max_width / combined_width, 1.0)
    scale_height = min(max_height / combined_height, 1.0)
    scale = min(scale_width, scale_height)

    if scale < 1.0:
        new_width1 = int(frame1.shape[1] * scale)
        new_height1 = int(frame1.shape[0] * scale)
        new_width2 = int(frame2.shape[1] * scale)
        new_height2 = int(frame2.shape[0] * scale)

        frame1_resized = cv2.resize(frame1, (new_width1, new_height1), interpolation=cv2.INTER_AREA)
        frame2_resized = cv2.resize(frame2, (new_width2, new_height2), interpolation=cv2.INTER_AREA)
    else:
        frame1_resized = frame1
        frame2_resized = frame2

    return frame1_resized, frame2_resized

def process_video_color_similarity(video_path):
    """
    Processes the video using both Celux and OpenCV, computes color similarity between them.

    Args:
        video_path (str): The path to the input video file.
    """
    try:
        frame_count = 0
        delta_e_total = 0.0
        start = time.time()

        celux_reader = celux.VideoReader(video_path, device="cpu", num_threads=16)((0,100))

        # Initialize OpenCV VideoCapture
        opencv_cap = cv2.VideoCapture(video_path)
        if not opencv_cap.isOpened():
            logging.error(f"Failed to open video with OpenCV: {video_path}")
            return

        input("Press Enter to start processing video...")

        while True:
            # Read frame from Celux
            try:
                celux_frame = next(celux_reader)
            except StopIteration:
                logging.info("Celux has finished reading all frames.")
                break
            except Exception as e:
                logging.error(f"Error reading frame from Celux: {e}")
                break

            # Read frame from OpenCV
            ret, opencv_frame = opencv_cap.read()
            if not ret:
                logging.info("OpenCV has finished reading all frames.")
                break

            # Convert Celux frame (assumed to be torch.Tensor) to numpy
            if isinstance(celux_frame, torch.Tensor):
                frame_cpu = celux_frame.cpu().numpy()
                # Assume Celux outputs in RGB format; convert to BGR for OpenCV
                frame_celux_bgr = cv2.cvtColor(frame_cpu, cv2.COLOR_RGB2BGR)
            else:
                logging.error("Celux frame is not a torch.Tensor.")
                break

            # Ensure OpenCV frame is in BGR format
            frame_opencv_bgr = opencv_frame  # Already in BGR

            # Convert both frames to LAB color space
            frame_celux_lab = cv2.cvtColor(frame_celux_bgr, cv2.COLOR_BGR2LAB)
            frame_opencv_lab = cv2.cvtColor(frame_opencv_bgr, cv2.COLOR_BGR2LAB)

            # Calculate Delta E
            delta_e = calculate_delta_e(frame_celux_lab, frame_opencv_lab)
            delta_e_total += delta_e
            frame_count += 1

            logging.info(f"Frame {frame_count}: ΔE = {delta_e:.2f}")

            # Resize frames to fit within the window
            frame_celux_resized, frame_opencv_resized = resize_frames(frame_celux_bgr, frame_opencv_bgr, max_width=1920, max_height=1080)

            # Combine frames side by side
            combined_frame = np.hstack((frame_celux_resized, frame_opencv_resized))

            # Add Delta E text overlay
            cv2.putText(combined_frame, f'ΔE: {delta_e:.2f}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the combined frame
            cv2.imshow("Celux (Left) vs OpenCV (Right)", combined_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Processing interrupted by user.")
                break

        end = time.time()
        elapsed_time = end - start
        average_delta_e = delta_e_total / frame_count if frame_count else 0

        logging.info(f"Time taken: {elapsed_time:.2f} seconds")
        logging.info(f"Total Frames Processed: {frame_count}")
        logging.info(f"Average ΔE: {average_delta_e:.2f}")
        logging.info(f"FPS: {frame_count / elapsed_time:.2f}")

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise
    finally:
        # Release OpenCV resources
        opencv_cap.release()
        try:
            celux_reader.close()
        except:
            pass
        cv2.destroyAllWindows()

def main(args):
    if args.mode == "lite":
        video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
        video_path = os.path.join(os.getcwd(), "ForBiggerBlazes.mp4")
    else:
        video_url = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        video_path = os.path.join(os.getcwd(), "BigBuckBunny.mp4")

    if not os.path.exists(video_path):
        download_video(video_url, video_path)
    else:
        logging.info(f"Video already exists at {video_path}")

    logging.info("Starting color similarity testing between Celux and OpenCV.")
    process_video_color_similarity(video_path)
    logging.info("Color similarity testing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing script to test color similarity between Celux and OpenCV.")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["lite", "full"],
        help="Choose 'lite' for GitHub Actions testing or 'full' for local testing."
    )
    args = parser.parse_args()

    main(args)
