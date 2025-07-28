[![Release and Benchmark Tests](https://github.com/Trentonom0r3/CeLux/actions/workflows/createRelease.yaml/badge.svg)](https://github.com/Trentonom0r3/CeLux/actions/workflows/createRelease.yaml)
[![License](https://img.shields.io/badge/license-AGPL%203.0-blue.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/celux)](https://pypi.org/project/celux/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/celux)](https://pypi.org/project/celux/)
[![Python Versions](https://img.shields.io/pypi/pyversions/celux)](https://pypi.org/project/celux/)
[![Discord](https://img.shields.io/discord/1041502781808328704.svg?label=Join%20Us%20on%20Discord&logo=discord&colorB=7289da)](https://discord.gg/hFSHjGyp4p)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-coff.ee%2Fspigonvidsu-yellow.svg?logo=buy-me-a-coffee&logoColor=white)](https://coff.ee/spigonvidsu)

# CeLux

**CeLux** is a high‑performance Python library for video processing, leveraging the power of FFmpeg. It delivers some of the fastest decode times for full‑HD videos globally, enabling efficient video decoding directly into PyTorch tensors—and now simplified, one‑call audio muxing straight from a tensor.

The name **CeLux** comes from the Latin words _celer_ (speed) and _lux_ (light), reflecting its commitment to speed and efficiency.

# [Check out the latest changes](docs/CHANGELOG.md#version-063)
- 🎶 **Simplified Audio Encoding**: Call `encode_audio_tensor()` with a full PCM tensor—CeLux handles chunking, float‑conversion, and PTS automatically.
- **Reduced Complexity of API, Adjusted Color Conversion** for more accurate HWC workflows.

## 📚 Documentation

- [📝 Changelog](docs/CHANGELOG.md)
- [🍎 Audio & Muxing Guide](docs/FAQ.md#audio)
- [📊 Benchmarks](https://github.com/NevermindNilas/python-decoders-benchmarks/blob/main/1280x720_diagram.png)
- [❓ FAQ](docs/FAQ.md)

## 🚀 Features

- ⚡ **Ultra‑Fast Video Decoding:** Lightning‑fast decode times for full‑HD videos using hardware acceleration.
- 🔗 **Direct Decoding to Tensors:** Frames come out as PyTorch tensors (`HWC` layout by default).
- 🔊 **Simplified Audio Encoding:** One call to `encode_audio_tensor()` streams your raw PCM into the encoder.
- 🔄 **Easy Integration:** Drop‑in replacement for your existing Python + PyTorch workflows.

## ⚡ Quick Start

```bash
pip install celux
```

```python
from celux import VideoReader
import torch

reader = VideoReader("/path/to/input.mp4")
with reader.create_encoder("/path/to/output.mp4") as enc:
    # 1) Re‑encode video frames
    for frame in reader:
        enc.encode_frame(frame)

    # 2) If there’s audio, hand off the entire PCM in one go:
    if reader.has_audio:
        pcm = reader.audio.tensor().to(torch.int16)
        enc.encode_audio_tensor(pcm)

print("Done!")
```

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[FFmpeg](https://ffmpeg.org/):** The backbone of video processing in CeLux.
- **[PyTorch](https://pytorch.org/):** For tensor operations and CUDA support.
- **[Vcpkg](https://github.com/microsoft/vcpkg):** Simplifies cross‑platform dependency management.
- **[@NevermindNilas](https://github.com/NevermindNilas):** For assistance with testing, API suggestions, and more.
