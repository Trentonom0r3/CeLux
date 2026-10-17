"""Concatenated MPEG-TS must decode straight through the splice.

``cat a.ts b.ts > all.ts`` (and joined HLS segments, and broadcast captures)
restarts the continuity counters at the seam. Unless the counter happens to
wrap to exactly the expected value, FFmpeg's mpegts demuxer flags the packet
straddling the seam ``AV_PKT_FLAG_CORRUPT`` -- the payload is intact, and the
ffmpeg CLI decodes through it. nelux treated the flag as damage everywhere, so
whether a spliced file opened depended on 1-in-16 counter luck: the
``discontinuity_ts`` range fixture passed on one CI runner and raised
"Decoding failed part way through" on the next, whose x264 thread count gave
a different stream size.
"""

from __future__ import annotations

import subprocess

import pytest
import torch  # noqa: F401  (nelux needs torch first)

from nelux import VideoReader
from tests.range_tools import FFMPEG, FFPROBE, available

pytestmark = pytest.mark.skipif(
    not (available(FFMPEG) and available(FFPROBE)), reason="ffmpeg and ffprobe required"
)


def _run(args):
    return subprocess.run(args, capture_output=True, check=True, timeout=90)


def _segment(path, threads, vf):
    _run([FFMPEG, "-v", "error", "-y",
          "-f", "lavfi", "-i", "testsrc2=size=320x192:rate=24:duration=2",
          "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
          "-vf", vf, "-c:v", "libx264", "-threads", str(threads), "-pix_fmt", "yuv420p",
          "-bf", "3", "-g", "24", "-c:a", "aac", str(path)])


def _packet_flags(path):
    out = _run([FFPROBE, "-v", "error", "-show_entries", "packet=flags",
                "-of", "csv=p=0", str(path)]).stdout.decode()
    return out.split()


@pytest.fixture(scope="module")
def spliced(tmp_path_factory):
    """A concatenation whose seam the demuxer really flags as corrupt.

    The flag depends on the segment sizes, so try a few x264 thread counts
    until one produces it; asserting it is there keeps the test from passing
    on a fixture that never exercised the bug.
    """
    d = tmp_path_factory.mktemp("splice")
    for threads in range(1, 9):
        a, b, joined = d / f"a{threads}.ts", d / f"b{threads}.ts", d / f"all{threads}.ts"
        _segment(a, threads, "null")
        _segment(b, threads, "hflip")
        joined.write_bytes(a.read_bytes() + b.read_bytes())
        if any("C" in f for f in _packet_flags(joined)):
            frames = _run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                           "-count_frames", "-show_entries", "stream=nb_read_frames",
                           "-of", "csv=p=0", str(joined)]).stdout.decode().split()[0]
            # mpegts lists the stream once per program section too
            return joined, int(frames)
    pytest.skip("no x264 thread count produced a flagged splice with this FFmpeg")


@pytest.mark.parametrize("prefetch", [False, True])
def test_video_decodes_through_the_splice(spliced, prefetch):
    path, expected = spliced
    with VideoReader(str(path), prefetch=prefetch) as reader:
        n = sum(1 for _ in reader)
    assert n == expected

