
## 📦 Installation
 
### 🤖 PIP installation

**CeLux** offers two installation options tailored to your system's capabilities:

1. **CPU-Only Version:** For systems without CUDA-capable GPUs.
2. **CUDA (GPU) Version:** For systems with NVIDIA GPUs supporting CUDA.

### 🖥️ CPU-Only Installation

Install the CPU version of **CeLux** using `pip`:

```bash
pip install celux
```

**Note:** The CPU version **only** supports CPU operations. Attempting to use GPU features with this version will result in an error.

### 🖥️ CUDA (GPU) Installation

Install the CUDA version of **CeLux** using `pip`:

```bash
pip install celux-cuda
```

**Note:** The CUDA version **requires** a CUDA-capable GPU and the corresponding Torch-Cuda installation.

### 🔄 Both Packages Import as `celux`

Regardless of the installation choice, both packages are imported using the same module name:

```python
import celux #as cx
```

This design ensures a seamless transition between CPU and CUDA versions without changing your import statements.

## 🛠️ Building from Source

While **CeLux** is easily installable via `pip`, you might want to build it from source for customization or contributing purposes.

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Trentonom0r3/celux.git
    cd celux
    ```

2. **Install Dependencies:**

    Ensure all prerequisites are installed. You can use `vcpkg` for managing dependencies on Windows.

3. **Configure the Project with CMake:**

    ```bash
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
    ```

    **Windows Users:** If using Vcpkg, include the toolchain file:

    ```bash
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>/scripts/buildsystems/vcpkg.cmake
    ```

4. **Build the Project:**

    ```bash
    cmake --build build --config Release
    ```

5. **Install the Package:**

    ```bash
    cmake --install build
    ```

6. **Set Up Environment Variables:**

    Ensure FFmpeg binaries and other dependencies are in your system's `PATH`. On Unix systems, you might need to set `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`.