#!/usr/bin/env python3
import sys
import subprocess
import time
import numpy as np
import cv2

# Add your CeLux build to the path
sys.path.append(r"D:/dev/Projects/Repos/CeLux/python")
import torch
from celux import VideoReader

VIDEO_PATH = r"C:\Users\tjerf\Downloads\1080.mp4"

# maximum display dimensions (e.g. your screen resolution)
MAX_W, MAX_H = 1920, 1080

def ffmpeg_rgb24_pipe(path):
    """
    Launch ffmpeg to decode `path` into raw RGB24 piped frames.
    Yields numpy arrays shaped (H, W, 3), dtype=uint8.
    """
    p = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", path
    ], capture_output=True, text=True)
    w, h = map(int, p.stdout.strip().split('x'))

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-map", "0:v:0",
        "-f", "rawvideo",
        "-color_range", "pc",
        "-pix_fmt", "rgb24",
        "-an",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_size = w * h * 3

    while True:
        data = proc.stdout.read(frame_size)
        if len(data) < frame_size:
            break
        yield np.frombuffer(data, np.uint8).reshape((h, w, 3))

    proc.stdout.close()
    proc.wait()

def compare_with_ffmpeg(path):
    ce_reader = VideoReader(path)
    ff_pipe   = ffmpeg_rgb24_pipe(path)

    for idx, celux_frame in enumerate(ce_reader):
        arr_celux = celux_frame.numpy().astype(np.uint8)

        try:
            arr_ff = next(ff_pipe)
        except StopIteration:
            print("FFmpeg ended early.")
            break

        if arr_ff.shape != arr_celux.shape:
            raise RuntimeError(
                f"Shape mismatch on frame {idx}: "
                f"{arr_celux.shape} vs {arr_ff.shape}"
            )

        diff = np.abs(arr_celux.astype(int) - arr_ff.astype(int))
        mad  = diff.mean()

        combined = np.hstack((arr_celux, arr_ff))

        # compute scale to fit in MAX_W x MAX_H
        h, w = combined.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        display = cv2.resize(combined, (disp_w, disp_h))

        cv2.putText(
            display,
            f"Frame {idx:03d}  MAD={mad:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1 * scale,
            (0, 255, 0),
            max(1, int(2 * scale))
        )

        cv2.imshow("CeLux (L) | FFmpeg (R)", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            print(f"[Frame {idx:03d}] MAD = {mad:.2f}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start = time.time()
    compare_with_ffmpeg(VIDEO_PATH)
    print(f"Done in {time.time() - start:.1f}s")
