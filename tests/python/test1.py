# example.py
import torch
import time
import cv2
import sys
sys.path.append(r"C:\Users\tjerf\source\repos\FFMPY\out\build\x64-release")
import ffmpy
def main():
    framecount = 0
    # Use with context manager
    start = time.time()
    with ffmpy.VideoReader(r"C:\Users\tjerf\source\repos\FrameSmith\Input.mp4", useHardware=True, hwType="cuda", as_numpy=False
                           ,dtype="float") as reader_cm:
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
