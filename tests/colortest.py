import sys
import subprocess
import time
import numpy as np
import cv2
import torch

# Add your CeLux build to the path
sys.path.append(r"D:/dev/Projects/Repos/CeLux/celux")
from celux import VideoReader

VIDEO_PATH = r"C:\Users\tjerf\Downloads\1080.mp4"
MAX_W, MAX_H = 1920, 1080

def ffmpeg_rgb24_pipe(path):
    """
    Launch ffmpeg to decode `path` into raw RGB24 piped frames.
    Yields numpy arrays shaped (H, W, 3), dtype=uint8.
    """
    # probe width/height
    p = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", path
    ], capture_output=True, text=True)
    w, h = map(int, p.stdout.strip().split('x'))

    # spawn ffmpeg pipe
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-map", "0:v:0",
        "-f", "rawvideo",
        "-color_range", "pc",     # full range metadata
        "-pix_fmt", "rgb24",      # raw RGB triplets
        "-an",
        "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_size = w * h * 3

    # yield each frame as an RGB numpy array
    while True:
        data = proc.stdout.read(frame_size)
        if len(data) < frame_size:
            break
        yield np.frombuffer(data, np.uint8).reshape((h, w, 3))

    proc.stdout.close()
    proc.wait()


def compare_with_ffmpeg(path):
    """
    Compare every frame from Celux vs. an ffmpeg RGB24 pipe.
    Displays side-by-side in OpenCV and logs a sample pixel + MAD.
    """
    ce_reader = VideoReader(path)
    ff_pipe   = ffmpeg_rgb24_pipe(path)

    for idx, celux_frame in enumerate(ce_reader):
        # --- read & prep raw arrays ---
        arr_celux = celux_frame.numpy().astype(np.uint8)  # RGB
        try:
            arr_ff = next(ff_pipe)                       # RGB
        except StopIteration:
            print("FFmpeg pipe ended early.")
            break

        # shape check
        if arr_ff.shape != arr_celux.shape:
            raise RuntimeError(
                f"Shape mismatch on frame {idx}: "
                f"{arr_celux.shape} vs {arr_ff.shape}"
            )

        # compute Mean Absolute Difference
        diff = np.abs(arr_celux.astype(int) - arr_ff.astype(int))
        mad  = diff.mean()

        # --- logging ---
        h0, w0, _ = arr_celux.shape
        # pick center pixel for a quick sanity check
        yc, xc = h0 // 2, w0 // 2
        print(
            f"[Frame {idx:03d}] "
            f"Sample@({xc},{yc}) CeLux={arr_celux[yc,xc]} "
            f"FFmpeg={arr_ff[yc,xc]}  MAD={mad:.2f}"
        )

        # --- prepare for display (OpenCV wants BGR) ---
        arr_celux_bgr = cv2.cvtColor(arr_celux, cv2.COLOR_RGB2BGR)
        arr_ff_bgr    = cv2.cvtColor(arr_ff,    cv2.COLOR_RGB2BGR)

        combined = np.hstack((arr_celux_bgr, arr_ff_bgr))

        # scale down if too large
        h, w = combined.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        display = cv2.resize(combined, (disp_w, disp_h))

        # overlay text
        cv2.putText(
            display,
            f"Frame {idx:03d}  MAD={mad:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1 * scale,
            (0, 255, 0),
            max(1, int(2 * scale))
        )

        # show & handle keys
        cv2.imshow("CeLux (L) | FFmpeg (R)", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            # extra debug print if you really want
            print(f"[Frame {idx:03d}] MAD = {mad:.2f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start = time.time()
    compare_with_ffmpeg(VIDEO_PATH)
    print(f"Done in {time.time() - start:.1f}s")
