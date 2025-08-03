import sys
sys.path.append(r"D:\dev\Projects\Repos\CeLux")  # Adjust path to import celux from parent directory

import celux
import torch

reader = celux.VideoReader(r"./tests/data/default/BigBuckBunny.mp4")
with reader.create_encoder(r"./tests/data/default/demo_output.mp4") as enc:
    # 1) Re‑encode video frames
    for frame in reader:
        enc.encode_frame(frame)

    # 2) If there’s audio, hand off the entire PCM in one go:
    if reader.has_audio:
        pcm = reader.audio.tensor().to(torch.int16)
        enc.encode_audio_frame(pcm)

print("Done!")