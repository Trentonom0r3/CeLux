
# ffmpy

ffmpy is a python video processing library built on top of FFMPEG. 
ffmpy boasts some of the, if not the, fastest decode times for full HD video *in the world!*

## Features
- **Video Decoding Directly to Numpy/Tensors:** Decode video directly into numpy arrays or torch tensors.

## Getting Started

### Prerequisites
- **FFmpeg:** Required for audio/video processing functionalities.
- **LibTorch & Pytorch with cuda support:** Required for torch::tensor and cuda processing.
- **Pybind11:** For python bindings.
- **C++17:** The project utilizes C++17 features and standard libraries.
- **CMake:** Used for building and managing project configuration.
- **Vcpkg:** (Optional) For managing dependencies like FFmpeg on Windows.

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Trentonom0r3/ffmpy.git
   cd FFMPyLib
   ```

2. Configure the project with CMake:
   ```bash
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake
   ```

3. Build the project:
   ```bash
   cmake --build build --config Debug
   ```

4. Run the tests (optional):
   ```bash
   cd build
   ctest
   ```

### Running the Project
- Download the most recent release.
  - .zip will contain all dependencies.
- Append directory to python path, and use as normal! (Torch MUST be imported first)

### Usage Example

```py
import ffmpy
import torch

def process_frame(frame):
    pass

    with ffmpy.VideoReader("path/to/input/video.mp4", useHardware=True, hwType="cuda", as_numpy=False) as reader:
        for frame in reader: #Frame will be in HWC Format, uint8, [0,255]
            process_frame(frame)

```

### Testing

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for any bugs or feature suggestions.

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Make your changes.
4. Submit a pull request.

## License
This project is licensed under the AGPL3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **FFmpeg:** This project is built on top of the powerful FFmpeg library.
- **Vcpkg:** Simplifies cross-platform dependency management.
