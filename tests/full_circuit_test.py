import celux
import torch
from utils.video_downloader import get_video #utility to download open source test clips
import os

OUTPUT_PATH = r"./tests/data/default/demo_output.mp4"

def main(mode:str = "full", video_path:str = None, output_path:str = None):
    video_path = video_path if video_path and os.path.exists(video_path) else get_video(mode)  # download video if not provided
    output_path = output_path if output_path and os.path.exists(output_path) else OUTPUT_PATH  # use default output path if not provided

    reader = celux.VideoReader(video_path)  # create the celux reader
    with reader.create_encoder(output_path) as enc:  # create the celux encoder
        # Decode + Re-encode video frames
        for frame in reader:
            enc.encode_frame(frame)

        # 2) If there’s audio, encode it as well
        if reader.has_audio:
            pcm = reader.audio.tensor().to(torch.int16)
            enc.encode_audio_frame(pcm)

    print("Done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video files")

    parser.add_argument("--mode", choices=["lite", "full"], default="full", help="Video mode")
    parser.add_argument("--video_path", default=None, help="Path to the input video file")
    parser.add_argument("--output_path", default=None, help="Path to the output video file")

    args = parser.parse_args()

    main(mode=args.mode, video_path=args.video_path, output_path=args.output_path)