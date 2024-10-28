
## ❓ FAQ

### Q: What video formats are supported?

**A:** **CeLux** aims to support all video formats and codecs that are supported by FFmpeg. Currently, hardware-accelerated decoding is available for specific codecs like H.264 and HEVC, which have been thoroughly tested. Additionally, the library supports conversions between various pixel formats, including:

- YUV420P
- YUV420P10LE
- BGR24
- RGB24
- P010LE

This enables compatibility with a wide range of codecs, including:

- H.264 (AVC)
- H.265 (HEVC)
- VP8/VP9
- AV1
- MPEG-2
- ProRes
- DNxHD/DNxHR
- DV (Digital Video)
- Uncompressed RGB
- While these formats and codecs are supported theoretically, testing has primarily focused on H.264 and HEVC.

### Q: How do I report a bug or request a feature?

**A:** Please open an issue on the [GitHub Issues](https://github.com/Trentonom0r3/celux/issues) page with detailed information about the bug or feature request.

## 🚤 Roadmap

- **Support for Additional Codecs:** 
  - Expand the range of supported video codecs.
  
- **Audio Processing:**
  - Introduce capabilities for audio extraction and processing.

- **Performance Enhancements:**
  - Further optimize decoding performance and memory usage.

- **Cross-Platform Support:**
  - Improve compatibility with different operating systems and hardware configurations.

