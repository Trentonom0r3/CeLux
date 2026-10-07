"""Operation-level bench: nelux vs torchcodec beyond sequential decode.

Each (library, operation) pair runs in a fresh process, ``--reps`` times, in
alternating order; the table reports the median wall time of the operation
and its spread. Both libraries use their default settings and the same
FFmpeg build (``external/ffmpeg/bin``; torchcodec loads its FFmpeg-9 core).

torchcodec is usually installed in a separate environment from the nelux
build. Point ``--tc-python`` at that interpreter; it must also import torch.

    python tests/bench_ops_vs_torchcodec.py --tc-python path/to/python.exe

Writes tests/output/ops_vs_torchcodec/<tag>.json and .md.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE / "data"
FFBIN = REPO / "external" / "ffmpeg" / "bin"
OUT = HERE / "output" / "ops_vs_torchcodec"

LONG = DATA / "BigBuckBunny.mp4"  # 720p, 596 s, 14315 frames, AAC stereo
HD = DATA / "test_1080p.mp4"      # 1080p, 600 frames, B-frames

# name -> (description, unit) ; every op returns elapsed seconds.
OPS = {
    "open_first": ("Open + first frame (720p)", "ms"),
    "open_random32": ("Open + 32 random exact frames (10-min 720p)", "ms"),
    "random32_hd": ("Open + 32 random exact frames (1080p, 600 fr)", "ms"),
    "clips": ("Open + 8 clips x 16 frames (10-min 720p)", "ms"),
    "played_at": ("Open + 32 frames by timestamp (10-min 720p)", "ms"),
    "jpeg": ("Decode 1080p JPEG (per image)", "ms"),
    "png": ("Decode 1080p PNG (per image)", "ms"),
    "audio": ("Decode 10-min AAC track to float32", "ms"),
    "audio16k": ("... resampled to 16 kHz mono", "ms"),
    "encode": ("Encode 120 1080p frames, libx264 crf 23 medium", "ms"),
}


def _fixture_images():
    """A 1080p JPEG and PNG cut from the 1080p clip, created on first use."""
    fx = HERE / "output" / "ops_vs_torchcodec" / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    jpg, png = fx / "frame1080.jpg", fx / "frame1080.png"
    if not (jpg.exists() and png.exists()):
        ffmpeg = FFBIN / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        subprocess.run([str(ffmpeg), "-v", "error", "-y", "-ss", "5", "-i", str(HD),
                        "-frames:v", "1", "-q:v", "2", str(jpg),
                        "-frames:v", "1", str(png)], check=True)
    return jpg, png


def run_op(lib: str, op: str) -> float:
    """Runs inside the child process. Returns seconds for the timed region."""
    if FFBIN.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(FFBIN))
    import torch
    if lib == "nelux":
        sys.path.insert(0, str(REPO))
        import nelux
        from nelux import VideoReader

        def video(path):
            return VideoReader(str(path))
    else:
        from torchcodec.decoders import VideoDecoder

        def video(path):
            return VideoDecoder(str(path), dimension_order="NHWC")

    now = time.perf_counter
    if op == "open_first":
        t = now(); d = video(LONG)
        f = next(iter(d)) if lib == "nelux" else d[0]
        return now() - t
    if op in ("open_random32", "random32_hd"):
        path, count = (LONG, 14315) if op == "open_random32" else (HD, 600)
        rng = random.Random(1234)
        idx = [rng.randrange(count) for _ in range(32)]
        t = now(); b = video(path).get_frames_at(idx)
        assert b.data.shape[0] == 32
        return now() - t
    if op == "clips":
        t = now(); d = video(LONG)
        if lib == "nelux":
            from nelux.samplers import clips_at_regular_indices
        else:
            from torchcodec.samplers import clips_at_regular_indices
        c = clips_at_regular_indices(d, num_clips=8, num_frames_per_clip=16)
        assert c.data.shape[:2] == (8, 16)
        return now() - t
    if op == "played_at":
        rng = random.Random(7)
        ts = [rng.uniform(0, 590) for _ in range(32)]
        t = now(); b = video(LONG).get_frames_played_at(ts)
        assert b.data.shape[0] == 32
        return now() - t
    if op in ("jpeg", "png"):
        jpg, png = _fixture_images()
        data = (jpg if op == "jpeg" else png).read_bytes()
        if lib == "nelux":
            from nelux import images
            fn = lambda: images.decode_image(data)
        else:
            from torchcodec.decoders import decode_jpeg, decode_png
            raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            fn = (lambda: decode_jpeg(raw)) if op == "jpeg" else (lambda: decode_png(raw))
        fn()
        t = now()
        for _ in range(30):
            fn()
        return (now() - t) / 30
    if op in ("audio", "audio16k"):
        opts = {"sample_rate": 16000, "num_channels": 1} if op == "audio16k" else {}
        t = now()
        if lib == "nelux":
            from nelux.audio import AudioReader
            s = AudioReader(str(LONG), **opts).get_all_samples()
        else:
            from torchcodec.decoders import AudioDecoder
            s = AudioDecoder(str(LONG), **opts).get_all_samples()
        assert s.data.shape[1] > 0
        return now() - t
    if op == "encode":
        if lib == "nelux":
            frames = VideoReader(str(HD)).get_batch_range(0, 120)
        else:
            frames = VideoDecoder(str(HD), dimension_order="NHWC").get_frames_in_range(0, 120).data
        frames = frames.contiguous()
        dst = OUT / f"encode_{lib}_{os.getpid()}.mp4"
        t = now()
        if lib == "nelux":
            from nelux.encoding import encode_video
            encode_video(frames, dst, codec="libx264", fps=30, preset="medium",
                         pixel_format="yuv420p", options={"crf": "23", "g": "250", "bf": "3"})
        else:
            from torchcodec.encoders import VideoEncoder
            VideoEncoder(frames.permute(0, 3, 1, 2), frame_rate=30).to_file(
                str(dst), codec="libx264", crf=23, preset="medium", pixel_format="yuv420p")
        el = now() - t
        dst.unlink()
        return el
    raise ValueError(op)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc-python", required=True, help="interpreter with torchcodec installed")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="current")
    ap.add_argument("--ops", nargs="*", default=list(OPS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pythons = {"nelux": sys.executable, "torchcodec": args.tc_python}
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    rows = {}
    for op in args.ops:
        times = {"nelux": [], "torchcodec": []}
        for rep in range(args.reps):
            order = ["nelux", "torchcodec"] if rep % 2 == 0 else ["torchcodec", "nelux"]
            for lib in order:
                p = subprocess.run([pythons[lib], __file__, "--child", lib, op],
                                   capture_output=True, text=True, env=env, timeout=900)
                line = [l for l in p.stdout.splitlines() if l.startswith("SECONDS ")]
                if p.returncode or not line:
                    raise RuntimeError(f"{lib} {op} failed:\n{p.stderr[-1500:]}")
                times[lib].append(float(line[0].split()[1]))
        med = {k: statistics.median(v) for k, v in times.items()}
        rows[op] = {"times_s": times, "median_s": med, "speedup": med["torchcodec"] / med["nelux"]}
        print(f"{op:14s} nelux {med['nelux']*1e3:9.2f} ms  torchcodec {med['torchcodec']*1e3:9.2f} ms"
              f"  nelux speedup {rows[op]['speedup']:.2f}x", flush=True)
    meta = {"tag": args.tag, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "reps": args.reps,
            "python": sys.version, "argv": sys.argv}
    (OUT / f"{args.tag}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    lines = ["| Operation | nelux | torchcodec | nelux speedup |", "|---|---:|---:|---:|"]
    for op, r in rows.items():
        lines.append(f"| {OPS[op][0]} | {r['median_s']['nelux']*1e3:.1f} ms | "
                     f"{r['median_s']['torchcodec']*1e3:.1f} ms | {r['speedup']:.2f}x |")
    (OUT / f"{args.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        print(f"SECONDS {run_op(sys.argv[2], sys.argv[3])!r}")
        sys.exit(0)
    sys.exit(main())
