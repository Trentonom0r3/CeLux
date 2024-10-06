# example.py
import torch
import time
import cv2
import sys
sys.path.append(r"C:\Users\tjerf\source\repos\FFMPY\out\build\x64-release")
import ffmpy
def main():
    start = time.time()
    reader = ffmpy.VideoReader(r"C:\Users\tjerf\source\repos\FrameSmith\Input.mp4", useHardware=True, hwType="cuda", as_numpy=False)
    framecount = 0
    # Iterate over frames
    for frame in reader:
        framecount+=1
        # Process the frame
        pass
    end = time.time()
    print("Tensor Frames")
    print(f"Time taken: {end-start} seconds")
    print(f"Total Frames: {framecount}")
    print(f"FPS: {framecount/(end-start)}")
    # Access a specific frame
   # frame_100 = reader[100]
    framecount = 0
    # Use with context manager
    start = time.time()
    with ffmpy.VideoReader(r"C:\Users\tjerf\source\repos\FrameSmith\Input.mp4", useHardware=True, hwType="cuda", as_numpy=False) as reader_cm:
        for frame in reader_cm:
            framecount+=1
            # frame is a NumPy array
            # Process the frame
            pass
    end = time.time()
    print("Numpy Frames")
    print(f"Time taken: {end-start} seconds")
    print(f"Total Frames: {framecount}")
    print(f"FPS: {framecount/(end-start)}")
if __name__ == "__main__":
    main()
