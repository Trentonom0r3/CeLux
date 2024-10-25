import argparse
import logging
import requests
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import celux_cuda as cx

HD_H264_8BIT = "https://repo.jellyfin.org/jellyfish/media/jellyfish-90-mbps-hd-h264.mkv" # 1080p, 90 Mbps, H.264, 8-bit
HD_HEVC_8BIT = "https://repo.jellyfin.org/jellyfish/media/jellyfish-90-mbps-hd-hevc.mkv" # 1080p, 90 Mbps, HEVC, 8-bit
HD_HEVC_10BIT = "https://repo.jellyfin.org/jellyfish/media/jellyfish-90-mbps-hd-hevc-10bit.mkv" # 1080p, 90 Mbps, HEVC, 10-bit

UHD_H264_8BIT = "https://repo.jellyfin.org/jellyfish/media/jellyfish-120-mbps-4k-uhd-h264.mkv" # 4K, 120 Mbps, H.264, 8-bit
UHD_HEVC_10BIT = "https://repo.jellyfin.org/jellyfish/media/jellyfish-120-mbps-4k-uhd-hevc-10bit.mkv" # 4K, 120 Mbps, HEVC, 10-bit

# Nicer prints
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

cx.set_log_level(cx.LogLevel.info)


def download_video(url, output_path):
    """
    Downloads a video from the specified URL.

    Args:
        url (str): The URL of the video.
        output_path (str): The local path to save the video.
    """
    try:
        success = check_if_exists(output_path)
        if success:
            return
        output_path = os.path.join(os.path.dirname(__file__), 'data', output_path)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as file:
            logging.info(f"Downloading {url} to {output_path}")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    except requests.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        raise
    
    
def print_video_information(video_path, hw = "cuda"):
    video_path = os.path.join(os.path.dirname(__file__), 'data', video_path)
    video = cx.VideoReader(video_path, hw)
    properties = video.get_properties()
    codec = properties['codec']
    pixel_format = properties['pixel_format']
    bit_depth = properties['bit_depth']
    logging.info(f"VideoReader Is Using {codec} Codec, with Bit Depth of {bit_depth}, and Pixel Format of {pixel_format}")
    
def check_if_exists(video_path):
    video_path = os.path.join(os.path.dirname(__file__), 'data', video_path)
    if not os.path.exists(video_path):
        logging.error(f"Video {video_path} does not exist.")
        return False
    return True

def main():
    
    logging.info("Downloading videos...")
    download_video(HD_H264_8BIT, "HD_H264_8BIT.mkv")    
    download_video(HD_HEVC_8BIT, "HD_HEVC_8BIT.mkv")
    download_video(HD_HEVC_10BIT, "HD_HEVC_10BIT.mkv")

    download_video(UHD_H264_8BIT, "UHD_H264_8BIT.mkv")
    download_video(UHD_HEVC_10BIT, "UHD_HEVC_10BIT.mkv")

    logging.info("Finished downloading videos.")
    
    logging.info("Printing video information, CPU...")
    
    print_video_information("HD_H264_8BIT.mkv", "cpu")
    print_video_information("HD_HEVC_8BIT.mkv", "cpu")
    print_video_information("HD_HEVC_10BIT.mkv", "cpu")
    
    print_video_information("UHD_H264_8BIT.mkv", "cpu")
    print_video_information("UHD_HEVC_10BIT.mkv", "cpu")
    
    logging.info("Printing video information, CUDA...")
    
    print_video_information("HD_H264_8BIT.mkv", "cuda")
    print_video_information("HD_HEVC_8BIT.mkv", "cuda")
    print_video_information("HD_HEVC_10BIT.mkv", "cuda")
    
    print_video_information("UHD_H264_8BIT.mkv", "cuda")
    print_video_information("UHD_HEVC_10BIT.mkv", "cuda")
    
if __name__ == "__main__":
    main()