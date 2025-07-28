# FAQ

## ❓ FAQ

### Q: What video formats are supported?

**A:** **CeLux** aims to support all video formats and codecs that FFmpeg can handle. Hardware‑accelerated decoding is available (and tested) for codecs like H.264 (AVC) and H.265 (HEVC), and the library supports a wide range of pixel‑format conversions, including:

- YUV420P  
- YUV420P10LE  
- BGR24  
- RGB24  
- P010LE  

This enables compatibility with most container/codec combos, such as:

- H.264 (AVC)  
- H.265 (HEVC)  
- VP8/VP9  
- AV1  
- MPEG‑2  
- ProRes  
- DNxHD/DNxHR  
- DV (Digital Video)  
- Uncompressed RGB  

While everything above “just works,” our regression tests focus primarily on H.264 and HEVC.

---

### Q: Does CeLux support audio muxing?

**A:** Yes! As of **v0.6.3**, you can re‑encode or pass through audio directly in Python via the same `VideoEncoder`:

```py
# assume `pcm` is a torch.int16 tensor of interleaved PCM samples,
# and `enc` is a VideoEncoder (from reader.create_encoder)
enc.encode_frame(video_frame)           # video
enc.encode_audio_frame(pcm[start:end])  # audio, in 1024‑sample chunks
```

Or, if you prefer a single‑call API (v0.6.3+):

```py
# pcm: torch.int16 CPU tensor
enc.encode_audio_tensor(pcm)  # handles chunking, float conversion & muxing
```

Your output file will have both video and aac audio tracks correctly interleaved.

---

### Q: How do I report a bug or request a feature?

**A:** Open an issue on our [GitHub Issues](https://github.com/Trentonom0r3/celux/issues) with as much detail as you can (FFmpeg version, platform, repro steps, etc.).

## 🚤 Roadmap

- **Support for Additional Codecs:**  
  - Expand hardware‑accelerated decoding/muxing support to VP9, AV1, etc.  
- **Audio Filters & Effects:**  
  - Add simple audio‑only filters (gain, resample, stereo panning).  
- **Advanced Muxing Options:**  
  - Expose more container parameters (subtitle tracks, chapters).  
- **Cross‑Platform CI:**  
  - Ensure Windows, macOS, Linux builds all pass full audio+video tests.
