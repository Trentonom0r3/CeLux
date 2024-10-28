[![License](https://img.shields.io/badge/license-AGPL%203.0-blue.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/celux)](https://pypi.org/project/celux/)
[![PyPI Version CUDA](https://img.shields.io/pypi/v/celux-cuda)](https://pypi.org/project/celux-cuda/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/celux)](https://pypi.org/project/celux/)
[![Python Versions](https://img.shields.io/pypi/pyversions/celux)](https://pypi.org/project/celux/)
[![Discord](https://img.shields.io/discord/1041502781808328704.svg?label=Join%20Us%20on%20Discord&logo=discord&colorB=7289da)](https://discord.gg/hFSHjGyp4p)


# CeLux

**CeLux** is a high-performance Python library for video processing, leveraging the power of FFmpeg. It delivers some of the fastest decode times for full HD videos globally, enabling efficient and seamless video decoding directly into PyTorch tensors.

The name **CeLux** is derived from the Latin words `celer` (speed) and `lux` (light), reflecting its commitment to speed and efficiency.

## 📚 Documentation

- [📝 Changelog](docs/CHANGELOG.md)
- [📦 Installation Instructions](docs/INSTALLATION.md)
  - [🤖 PIP installation](docs/INSTALLATION.md#pip-installation)
  - [🛠️ Building from Source](docs/INSTALLATION.md#building-from-source)
- [🚀 Getting Started](docs/GETTINGSTARTED.md)
- [📊 Benchmarks](docs/BENCHMARKS.md)
- [🤝 Contributing Guide](docs/CONTRIBUTING.md)
- [❓ FAQ](docs/FAQ.md)

## 🚀 Features

- **⚡ Ultra-Fast Video Decoding:** Achieve lightning-fast decode times for full HD videos using hardware acceleration.
- **🔗 Direct Decoding to Tensors:** Decode video frames directly into PyTorch tensors for immediate processing.
- **🖥️ Hardware Acceleration Support:** Utilize CUDA for GPU-accelerated decoding, significantly improving performance.
- **🔄 Easy Integration:** Seamlessly integrates with existing Python workflows, making it easy to incorporate into your projects.

## ⚡ Quick Start

```sh
pip install celux  # cpu only version
```
```sh
pip install celux-cuda  # cuda+cpu
```
```py
from celux import VideoReader
#import celux as cx
reader = VideoReader("/path/to/video.ext", device = "cuda")
for frame in reader:
# do something
```

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[FFmpeg](https://ffmpeg.org/):** The backbone of video processing in CeLux.
- **[PyTorch](https://pytorch.org/):** For tensor operations and CUDA support.
- **[Vcpkg](https://github.com/microsoft/vcpkg):** Simplifies cross-platform dependency management.
- **[@NevermindNilas](https://github.com/NevermindNilas):** For assistance with testing, API suggestions, and more.


