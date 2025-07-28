#!/usr/bin/env python3
import sys, time, torch

# Point at your built celux/python module
sys.path.append(r"D:/dev/Projects/Repos/CeLux/python")
from celux import VideoReader

VIDEO_PATH  = r"C:\Users\tjerf\Downloads\1080.mp4"
OUTPUT_PATH = r"C:\Users\tjerf\Downloads\1080_celux_with_audio.mp4"

def reencode_with_audio(in_path, out_path):
    reader = VideoReader(in_path)
    w, h, fps = reader.width, reader.height, reader.fps
    print(f"[Info] Opened {in_path}: {w}×{h} @ {fps:.3f}fps")

    # 1) encode video
    with reader.create_encoder(out_path) as enc:
        for frame in reader:
            enc.encode_frame(frame)

        # 2) if there's audio, hand it off in one go
        if reader.has_audio:
            pcm = reader.audio.tensor().to(torch.int16)  # full PCM
            print(f"[Info] Muxing audio: {pcm.numel()} samples")
            enc.encode_audio_frame(pcm)

    print(f"[Info] Done!  Output saved to: {out_path}")

if __name__ == "__main__":
    t0 = time.time()
    reencode_with_audio(VIDEO_PATH, OUTPUT_PATH)
    print(f"Total elapsed: {time.time() - t0:.1f}s")
