import time
import sys

# Adjust the path to include CeLux
sys.path.append("D:/dev/Projects/Repos/CeLux/python")

from celux import VideoReader

VIDEO_PATH = r"D:\dev\Projects\Repos\CeLux\tests\data\color_fmts\output_yuv444p12le.mp4"


def benchmark_celux():
    reader = VideoReader(VIDEO_PATH)
    start = time.time()
    frame_count = 0
    for frame in reader:
        frame_count += 1
    return frame_count / (time.time() - start)

if __name__ == "__main__":
    print(f"CeLux FPS: {benchmark_celux():.2f}")
