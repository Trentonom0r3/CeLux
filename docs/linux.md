
# Install vcpkg and Dependencies

```bash
# Clone vcpkg repository
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg


# Update and install required system packages
sudo apt-get update
sudo apt-get install curl zip unzip tar nasm

# Bootstrap vcpkg
./bootstrap-vcpkg.sh
```

# Install CMake v3.30.1

```bash
# Download and extract CMake
wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1-linux-x86_64.tar.gz
tar -zxvf cmake-3.30.1-linux-x86_64.tar.gz

# Move CMake to /opt and create a symlink
sudo mv cmake-3.30.1-linux-x86_64 /opt/cmake-3.30.1
sudo ln -s /opt/cmake-3.30.1/bin/cmake /usr/local/bin/cmake
```
# Install Cuda ToolKit

```bash
#install cuda toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

# Install Project Dependencies with vcpkg

```bash
sudo apt-get install pkg-config
# Install FFmpeg with specific features
./vcpkg install "ffmpeg[avcodec,avformat,swscale,avfilter,nvcodec, x264, x265]:x64-linux"

# Install additional libraries
./vcpkg install pybind11
./vcpkg install spdlog
./vcpkg install fmt
```

### Download and extract CPU and CUDA versions of libtorch

```bash
# Download and extract CPU and CUDA versions of libtorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip -O libtorch-cpu.zip && \
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip -O libtorch-cu124.zip && \
# Extract using sudo to avoid permission issues in /opt
sudo unzip libtorch-cpu.zip -d /opt/libtorch_cpu
sudo unzip libtorch-cu124.zip -d /opt/libtorch_cuda
rm libtorch-cpu.zip libtorch-cu124.zip
```