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
| Test Celux Cpu Benchmark | [1280, 720] | 16 | 9.41 | 0.16 | 1520.75 |
| Test Celux Cuda Benchmark | [1280, 720] | 16 | 8.36 | 0.28 | 1710.85 |
| Test Pyav Cpu Benchmark | [1280, 720] | 1 | 40.83| 8.14 | 350.58|
| Test Opencv Cpu Benchmark | [1280, 720] | 1 | 31.49 | 2.49 | 454.44 |


### 📊 **Benchmark Visualizations**

![FPS Comparison](scripts/benchmarks/fps_comparison.png)

![Mean Time Comparison](scripts/benchmarks/mean_time_comparison.png)



<!-- BENCHMARKS_END -->