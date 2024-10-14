# example.py
import torch
import time
import cv2
import sys
#out/build/x64-release
sys.path.append(r"C:\Users\tjerf\source\repos\ffmpy\out\build\x64-release")
import ffmpy


def main():
    try:
        framecount = 0
        # Use with context manager
        start = time.time()
        with ffmpy.VideoReader(
            r"C:\Users\tjerf\source\repos\FrameSmith\Input.mp4",
            as_numpy=True,
            d_type="uint8",
        ) as reader_cm:
            #with ffmpy.VideoWriter("./output.mp4", 1920, 1080, 24.0, as_numpy=True) as writer:
            for frame in reader_cm:
               # writer(frame)
                framecount += 1

                pass
        end = time.time()
        print("Numpy Frames")
        print(f"Time taken: {end-start} seconds")
        print(f"Total Frames: {framecount}")
        print(f"FPS: {framecount/(end-start)}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
