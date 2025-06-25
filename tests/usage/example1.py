import os
import sys
import time
import logging
import torch
import cv2

# Append the parent directory to system path to allow package import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Configure logging for clean output
logging.basicConfig(level=logging.INFO, format="%(message)s")

import celux

celux.set_log_level(celux.LogLevel.debug)
def test_video_encoding(input_video, output_video):
    """
    Test the VideoReader and VideoEncoder classes.

    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to save the encoded video.
    """
    if not os.path.exists(input_video):
        logging.error(f"❌ Missing test video: {input_video}")
        return

    try:
        logging.info(f"🔍 Loading video: {input_video}")

        # Initialize the video reader
        reader = celux.VideoReader(input_video)
        logging.info(f"📹 Video Properties: {reader.properties}")

        # Get the first frame for validation
        first_frame = next(iter(reader), None)
        if first_frame is None:
            logging.error("❌ Failed to decode first frame")
            return
        
        logging.info("✅ Video loaded successfully.")

        # Infer properties for encoding
        width, height = first_frame.shape[1], first_frame.shape[0]
        fps = reader.properties.get("fps", 30)  # Default to 30 FPS if not provided
        logging.info(f"📏 Frame Size: {width}x{height}, FPS: {fps}")

        # Initialize the video encoder
        encoder = celux.VideoEncoder(output_video, codec="libx264", width=width, height=height, fps=int(fps))
        logging.info(f"🎬 Encoding to: {output_video}")

        # Benchmark decoding and encoding speed
        num_frames = 100  # Adjust for accuracy
        start_time = time.time()

        for frame in reader:
            reader([0,100])
            if frame is None:
                logging.warning("⚠️ No more frames to read.")
                break
            encoder.encodeFrame(frame)


        elapsed_time = time.time() - start_time
        fps = num_frames / elapsed_time if elapsed_time > 0 else 0

        # Close the encoder
        encoder.close()

        logging.info(f"⚡ Benchmark: {fps:.2f} FPS for {num_frames} frames (Decoding & Encoding).")
        logging.info(f"✅ Encoding complete: {output_video}")

    except Exception as e:
        logging.error(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    # Allow passing video file paths as arguments or use default paths
    default_input = os.path.join(os.path.dirname(__file__), "..", "data", "default", "BigBuckBunny.mp4")
    default_output = os.path.join(os.path.dirname(__file__), "..", "data", "output", "encoded_video.mp4")

    input_video = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_video = sys.argv[2] if len(sys.argv) > 2 else default_output

    test_video_encoding(input_video, output_video)
