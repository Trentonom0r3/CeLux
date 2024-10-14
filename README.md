
# ffmpy

ffmpy is a high-performance Python library for video processing built on top of FFmpeg. It offers some of the fastest decode times for full HD video in the world, allowing for efficient and seamless video decoding directly into NumPy arrays or PyTorch tensors.

## Features

- **Ultra-Fast Video Decoding:** Achieve lightning-fast decode times for full HD videos using hardware acceleration.
- **Direct Decoding to NumPy/Tensors:** Decode video frames directly into NumPy arrays or PyTorch tensors for immediate processing.
- **Hardware Acceleration Support:** Utilize CUDA for GPU-accelerated decoding, significantly improving performance.
- **Easy Integration:** Seamlessly integrates with existing Python workflows, making it easy to incorporate into your projects.
- **Supports Multiple Data Types:** Handle video frames in `uint8`, `float32`, or `float16` data types.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Building from Source](#building-from-source)
- [Usage](#usage)
  - [Basic Example](#basic-example)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [FAQ](#faq)
- [Roadmap](#roadmap)

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **FFmpeg:** Required for audio/video processing functionalities.
- **PyTorch with CUDA Support:** Required for tensor operations and GPU acceleration.
- **LibTorch:** The PyTorch C++ library, necessary for building the project.
- **PyBind11:** Used for creating Python bindings for C++ code.
- **C++17 Compiler:** The project utilizes C++17 features.
- **CMake (Version 3.12 or higher):** Used for building and managing the project configuration.
- **Vcpkg (Optional):** For managing dependencies like FFmpeg on Windows.

### Installation

Currently, ffmpy can be installed by downloading the latest release package. Please follow these steps:

1. **Download the Latest Release:**

   - Visit the [Releases](https://github.com/Trentonom0r3/ffmpy/releases) page and download the most recent `.zip` file, which includes all dependencies.

2. **Extract the Package:**

   - Extract the contents of the `.zip` file to your desired location.

3. **Update Python Path:**

   - Append the directory to your Python path to ensure Python can find the ffmpy module.
   - **Note:** Make sure to import `torch` before importing `ffmpy` in your Python scripts.


### Building from Source

If you prefer to build ffmpy from source, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Trentonom0r3/ffmpy.git
   cd ffmpy
   ```

2. **Configure the Project with CMake:**

   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
   ```

   - **Windows Users:** If using Vcpkg, include the toolchain file:

     ```bash
     cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake
     ```

3. **Build the Project:**

   ```bash
   cmake --build build --config Release
   ```

4. **Install the Package:**

   ```bash
   cd build
   cmake --install .
   ```

5. **Set Up Environment Variables:**

   - Ensure that the FFmpeg binaries and any other dependencies are in your system's PATH.
   - Set the `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` on Unix systems if necessary.

## Usage

### Basic Example

Here's a simple example demonstrating how to use ffmpy to read video frames and process them:

```python
import torch  # Import torch first
import ffmpy

def process_frame(frame):
    # Implement your frame processing logic here
    pass

# Ensure torch is imported before ffmpy
with ffmpy.VideoReader(
    "path/to/input/video.mp4",
    as_numpy=False,
    d_type="uint8"
) as reader:
    for frame in reader:
        # Frame will be in HWC format, uint8 data type, values in [0, 255]
        process_frame(frame)
```

**Parameters:**

- `useHardware` (bool): Enable hardware acceleration (default: `True`).
- `hwType` (str): Type of hardware acceleration to use (e.g., `"cuda"`).
- `as_numpy` (bool): Return frames as NumPy arrays if `True`, else as PyTorch tensors (default: `False`).
- `dtype` (str): Data type of the output frames (`"uint8"`, `"float"`, or `"half"`).

**Note:** If you set `dtype` to `"float"` or `"half"`, the frame values will be normalized between `0.0` and `1.0`.

## Contributing

Contributions are welcome! Please follow these steps to contribute to the project:

1. **Fork the Repository:**

   - Click the "Fork" button at the top right of the repository page to create your own fork.

2. **Clone Your Fork:**

   ```bash
   git clone https://github.com/your-username/ffmpy.git
   cd ffmpy
   ```

3. **Create a New Branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes:**

   - Implement your feature or bugfix.

5. **Commit Your Changes:**

   ```bash
   git commit -am "Add your commit message here"
   ```

6. **Push to Your Fork:**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request:**

   - Go to the original repository and click on "Pull Requests," then "New Pull Request."

## Changelog
### Version 0.2.2 (2024-10-14)
- **Pre-Release Update:**
  - Fixed a few small issues.
  - Made `VideoReader` and `VideoWriter` callable.
  - Created BGR onversions.
  - Added frame range (in/out) args.
  ```py
  with VideoReader('input.mp4')([10, 20]) as reader:
    for frame in reader:
        print(f"Processing frame {frame}")
  ```
### Version 0.2.1 (2024-10-13)

- **Pre-Release Update:**
  - Adjusted Python bindings to use snake_case.
  - Added `.pyi` stub files to `.whl`.
  - Adjusted  `d_type` args to (`uint8`, `float32`, `float16`).
  - Added github actions for new releases.
  - Added HW Accel Encoder support, direct encoding from numpy/Tensors.
  - Added `has_audio` property to `VideoReader.get_properties()`.

### Version 0.1.1 (2024-10-06)

- **Pre-Release Update:**
  - Implemented support for multiple data types (`uint8`, `float`, `half`).
  - Provided example usage and basic documentation.


## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **FFmpeg:** This project is built on top of the powerful [FFmpeg](https://ffmpeg.org/) library.
- **PyTorch:** Utilizes [PyTorch](https://pytorch.org/) for tensor operations and CUDA support.
- **Vcpkg:** Simplifies cross-platform dependency management.
- **[NevermindNilas](https://github.com/NevermindNilas)** - for help testing, API suggestions + more. 

## FAQ

### Q: Why do I need to import `torch` before `ffmpy`?

**A:** ffmpy depends on PyTorch, and importing `torch` first ensures that all the necessary CUDA context and resources are correctly initialized before ffmpy uses them.

### Q: Can I use ffmpy without CUDA or GPU acceleration?

**A:** Yes, you can set `useHardware=False` when initializing `VideoReader` to use CPU decoding. However, performance may be significantly slower compared to GPU-accelerated decoding.

### Q: What video formats are supported?

**A:** ffmpy's goal is to support all video formats and codecs that are supported by FFmpeg. However, hardware-accelerated decoding may only be available for specific codecs like H.264 and HEVC. Currently, H264/H265/HEVC are the only codecs tested.

### Q: How do I report a bug or request a feature?

**A:** Please open an issue on the [GitHub Issues](https://github.com/Trentonom0r3/ffmpy/issues) page with detailed information about the bug or feature request.

## Roadmap
- **Additional Conversion Support:**
  - Create additional conversion modules:
    - NV12ToBGR, BGRToNV12, NV12ToNV12(no change converter)

- **Audio Processing:**
  - Introduce capabilities for audio extraction and processing.

- **Performance Enhancements:**
  - Further optimize decoding performance and memory usage.

- **Cross-Platform Support:**
  - Improve compatibility with different operating systems and hardware configurations.

- **Support for Additional Codecs:**
  - Expand the range of supported video codecs.




