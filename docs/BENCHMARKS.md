# 📈 **Benchmarks**
<!-- BENCHMARKS_START -->

### 🖥️ **System Specifications**

| Specification         | Details                                 |
|-----------------------|-----------------------------------------|
| **Processor**         | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel |
| **Architecture**      | AMD64 |
| **Python Version**    | 3.12.7 (CPython) |
| **Python Build**      | tags/v3.12.7:0b05ead Oct  1 2024 03:06:41 |
| **Operating System**  | Windows 11 |
| **CPU Brand**         | 12th Gen Intel(R) Core(TM) i7-12700H |
| **CPU Frequency**     | 2.3000 GHz |
| **L2 Cache Size**     | 11776 KB |
| **L3 Cache Size**     | 24576 KB |
| **Number of Cores**   | 20 |
| **GPU #1**           | NVIDIA GeForce RTX 3060 Laptop GPU (6.00 GB) |


| Benchmark                      | Video Size  | Threads | Mean Time (s) | Std Dev (s) | FPS    |
|--------------------------------|-------------|---------|---------------|-------------|--------|
| Test Video Reader Cpu Benchmark | [1280, 720] | 16     | 10.67          | 2.72       | 1341.77 |
| Test Video Reader Cuda Benchmark | [1280, 720] | 16     | 7.95          | 0.03       | 1799.69 |


### 📊 **Benchmark Visualizations**

![FPS Comparison](scripts/benchmarks/fps_comparison.png)

![Mean Time Comparison](scripts/benchmarks/mean_time_comparison.png)



<!-- BENCHMARKS_END -->