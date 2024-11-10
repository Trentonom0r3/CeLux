
import time
import argparse
import logging
import numpy as np
import sys
import os
import cv2
import torch

# Adjust the path to include celux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import celux_cuda as celux

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def print_info(path):
        
    reader = celux.VideoReader(path, device='cpu')
    logging.info(f"Total Frames: {reader['total_frames']}")
    logging.info(f"Width: {reader['width']}")
    logging.info(f"Height: {reader['height']}")
    #fps
    logging.info(f"FPS: {reader['fps']}")
    #duration
    logging.info(f"Duration: {reader['duration']} ms")
    #pixel format
    logging.info(f"Pixel Format: {reader['pixel_format']}")
    #codec 
    logging.info(f"Codec: {reader['codec']}")
    #bit_depth
    logging.info(f"Bit Depth: {reader['bit_depth']}")

def show_frame(path):
    reader = celux.VideoReader(path, device='cpu')
    reader((0.00, 2.00))
    for i, frame in enumerate(reader):
        print(f"Frame {i}")
        cv2.imshow("Frame", frame.numpy())
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    TEST_VIDEO_PATH = os.path.join(os.getcwd(), "BigBuckBunny.mp4")

    print_info(TEST_VIDEO_PATH)
    show_frame(TEST_VIDEO_PATH)