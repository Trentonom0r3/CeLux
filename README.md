
# FFMPy

FFMPy is a C++ audio/video processing library built on top of FFmpeg. It provides convenient wrappers around FFmpeg's core functionalities, including media format handling, decoding, and audio resampling. 
The library is designed to make it easier to integrate advanced media processing tasks into your projects.

## Features
- **Media Format Handling:** Easily open and manage media files, retrieve stream information, and select specific audio/video streams.
- **Audio/Video Decoding:** Decode audio and video streams using FFmpeg's powerful codec support.
- **Audio Resampling:** Seamlessly convert audio data between different sample rates, formats, and channel layouts.
- **Error Handling:** Comprehensive error management using custom exception classes for better control over media processing flows.

## Getting Started

### Prerequisites
- **FFmpeg:** Ensure that FFmpeg is installed on your system and accessible through the project's CMake configuration.
- **C++17 Compiler:** The project is built with C++17, so make sure your compiler supports this standard.
- **CMake:** Use CMake (version 3.15 or higher) to build the project.
- **Vcpkg:** (For Windows users) Use vcpkg for managing dependencies, including FFmpeg.

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FFMPyLib.git
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

### Usage Example

Here’s an example of how to use FFMPyLib for resampling audio from a media file:

```cpp
#include "MediaFormat.hpp"
#include "MediaDecoder.hpp"
#include "MediaResampler.hpp"

int main() {
    try {
        std::string filePath = "path/to/your/audiofile.mp4";

        // Initialize MediaFormat
        FFmpeg::MediaFormat mediaFormat(filePath);
        mediaFormat.open();

        // Initialize MediaDecoder for the audio stream
        FFmpeg::MediaDecoder decoder(mediaFormat);
        decoder.initializeAudioDecoder(0); // Assuming stream index 0 is audio

        // Initialize Resampler
        AVChannelLayout outputChannelLayout;
        av_channel_layout_default(&outputChannelLayout, 1); // Mono output
        decoder.initializeResampler(44100, AV_SAMPLE_FMT_S16, outputChannelLayout);

        // Resample audio frames
        AVFrame* outputFrame = av_frame_alloc();
        while (decoder.decodeAndResampleAudioFrame(outputFrame)) {
            // Process the resampled output frame
            std::cout << "Resampled " << outputFrame->nb_samples << " samples." << std::endl;
        }

        av_frame_free(&outputFrame);
    } catch (const FFmpeg::Error::FFException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
```

### Testing
The project includes unit tests for media decoding and resampling. Tests are located in the `tests/` directory. You can run these tests after building the project to verify its functionality.

### Dependencies
- **FFmpeg:** Required for audio/video processing functionalities.
- **C++17:** The project utilizes C++17 features and standard libraries.
- **CMake:** Used for building and managing project configuration.
- **Vcpkg:** (Optional) For managing dependencies like FFmpeg on Windows.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for any bugs or feature suggestions.

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Make your changes.
4. Submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **FFmpeg:** This project is built on top of the powerful FFmpeg library.
- **Vcpkg:** Simplifies cross-platform dependency management.
