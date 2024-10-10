# example.py
import torch
import time
import cv2
import sys
import ffmpy
def main():
    try:
        framecount = 0
        # Use with context manager
        start = time.time()
        with ffmpy.VideoReader(r"C:\Users\tjerf\source\repos\FrameSmith\Input.mp4", as_numpy=False ,dtype="uint8") as reader_cm:
            for frame in reader_cm:
                cv2.imshow("Frame", frame.cpu().numpy())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                framecount+=1

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
