# test_color_fmts.py
# ----------------------------------------------------------------------------
# Script used to test the color formats supported by Celux.
# Prints a summary of supported and unsupported formats after running the tests.
# ----------------------------------------------------------------------------

import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import celux
import logging
from tests.utils.generate_test_videos import generate_test_videos

# Set up logging with a cleaner format
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure test videos exist before running tests
@pytest.fixture(scope="session", autouse=True)
def setup_videos():
    """Generate test videos, but suppress duplicate logs."""
    generate_test_videos()

# Define the test video formats
VIDEO_FILES = [
    "output_yuv420p8le.mp4",
    "output_yuv420p10le.mp4",
    "output_yuv420p12le.mp4",
    "output_yuv422p8le.mp4",
    "output_yuv422p10le.mp4",
    "output_yuv422p12le.mp4",
    "output_yuv444p8le.mp4",
    "output_yuv444p10le.mp4",
    "output_yuv444p12le.mp4",
    "output_rgb24.mp4",
    "output_nv12.mp4",
    "output_prores422.mov",
    "output_prores4444.mov",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "color_fmts")

# Track results for a final summary
SUPPORTED_FORMATS = []
UNSUPPORTED_FORMATS = []

@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_video_format_loading(video_file):
    """Test whether CeLux can load different video color formats."""
    video_path = os.path.join(DATA_DIR, video_file)

    # Ensure file exists
    assert os.path.exists(video_path), f"❌ Missing test video: {video_file}"

    try:
        reader = celux.VideoReader(video_path)
        first_frame = next(iter(reader))
        assert first_frame is not None, "❌ Failed to decode first frame"

        # If successful, mark it as supported
        SUPPORTED_FORMATS.append(video_file)
    except Exception as e:
        UNSUPPORTED_FORMATS.append((video_file, str(e)))

@pytest.fixture(scope="session", autouse=True)
def print_summary(request):
    """Prints a summary of supported and unsupported formats after tests run."""
    def summary():
        logging.info("\n" + "=" * 60)
        logging.info("\033[92m✅ Supported Formats:\033[0m")  # Green color
        for fmt in SUPPORTED_FORMATS:
            #remove the output_ part of the filename
            fmt = fmt[7:]
            #remove the file extension
            fmt = fmt.split(".")[0]
            logging.info(f"    - {fmt}")

        if UNSUPPORTED_FORMATS:
            logging.info("\n\033[91m❌ Unsupported Formats:\033[0m")  # Red color
            for fmt, reason in UNSUPPORTED_FORMATS:
                fmt = fmt[7:]
                fmt = fmt.split(".")[0]
                logging.info(f"    - {fmt}")

        logging.info("=" * 60 + "\n")

    request.addfinalizer(summary)

if __name__ == "__main__":
    pytest.main([__file__])