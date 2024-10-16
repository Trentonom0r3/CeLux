cmake_minimum_required(VERSION 3.15)

# ------------------------------
# 1. Project Setup
# ------------------------------

# Use vcpkg toolchain if on Windows
if (WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/tjerf/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

# Enable C++ and CUDA languages
project(FFMPy LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------
# 2. Specify CUDA Paths Directly
# ------------------------------

# Define CUDA toolkit root path
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")

# Include CUDA directories
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")

# ------------------------------
# 3. Manually Specify FFmpeg Static Libraries (A)
# ------------------------------

# Define the vcpkg installation path
set(VCPKG_INSTALLED_DIR "C:/Users/tjerf/vcpkg/installed/x64-windows")

# Specify the path to FFmpeg include directories
set(FFMPEG_INCLUDE_DIRS "${VCPKG_INSTALLED_DIR}/include")

# List the static FFmpeg libraries you want to link against
set(FFMPEG_STATIC_LIBS
    "${VCPKG_INSTALLED_DIR}/lib/avutil.lib"
    "${VCPKG_INSTALLED_DIR}/lib/avcodec.lib"
    "${VCPKG_INSTALLED_DIR}/lib/avformat.lib"
    "${VCPKG_INSTALLED_DIR}/lib/avdevice.lib"
    "${VCPKG_INSTALLED_DIR}/lib/swscale.lib"
    "${VCPKG_INSTALLED_DIR}/lib/swresample.lib"
    "${VCPKG_INSTALLED_DIR}/lib/avfilter.lib"  # Add avfilter library
    "${VCPKG_INSTALLED_DIR}/lib/libx264.lib"
)

# ------------------------------
# 4. Keep Torch/CUDA as Dynamic Libraries (B)
# ------------------------------

# Set the path to the root of the libtorch installation
set(TORCH_INSTALL_PREFIX "C:/Users/tjerf/Downloads/libtorch-win-shared-with-deps-2.4.0+cu121/libtorch")
set(CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")

# Find Torch
find_package(Torch REQUIRED)

# ------------------------------
# 5. Find Windows Media Foundation Libraries
# ------------------------------

# Define the Windows SDK lib path (adjust the version as needed)
set(WINDOWS_SDK_LIB_PATH "C:/Program Files (x86)/Windows Kits/10/Lib")

# Find mf.lib
find_library(MF_LIB
    NAMES mf
    PATHS "${WINDOWS_SDK_LIB_PATH}/*/um/x64"
    NO_DEFAULT_PATH
)

if(NOT MF_LIB)
    message(FATAL_ERROR "mf.lib not found. Ensure the Windows SDK is installed.")
endif()

# Find mfuuid.lib
find_library(MFUUUID_LIB
    NAMES mfuuid
    PATHS "${WINDOWS_SDK_LIB_PATH}/*/um/x64"
    NO_DEFAULT_PATH
)

if(NOT MFUUUID_LIB)
    message(FATAL_ERROR "mfuuid.lib not found. Ensure the Windows SDK is installed.")
endif()

# Find ole32.lib
find_library(OLE32_LIB
    NAMES ole32
    PATHS "${WINDOWS_SDK_LIB_PATH}/*/um/x64"
    NO_DEFAULT_PATH
)

if(NOT OLE32_LIB)
    message(FATAL_ERROR "ole32.lib not found. Ensure the Windows SDK is installed.")
endif()

# ------------------------------
# 6. Manually Specify Python 3.12 Paths (C)
# ------------------------------

# Manually set Python paths
set(PYTHON_EXECUTABLE "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/python.exe")
set(PYTHON_INCLUDE_DIR "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/include")
set(PYTHON_LIBRARY "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/libs/python312.lib")

# Ensure that the Python executable exists
if(NOT EXISTS "${PYTHON_EXECUTABLE}")
    message(FATAL_ERROR "Python executable not found at ${PYTHON_EXECUTABLE}")
endif()

# Ensure that the Python include directory exists
if(NOT EXISTS "${PYTHON_INCLUDE_DIR}")
    message(FATAL_ERROR "Python include directory not found at ${PYTHON_INCLUDE_DIR}")
endif()

# Ensure that the Python library exists
if(NOT EXISTS "${PYTHON_LIBRARY}")
    message(FATAL_ERROR "Python library not found at ${PYTHON_LIBRARY}")
endif()

# ------------------------------
# 7. Find pybind11 using vcpkg
# ------------------------------
find_package(pybind11 REQUIRED)

# ------------------------------
# 8. Find Torch Python Library
# ------------------------------

# Find the path to the installed torch module
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT TORCH_PYTHON_PATH)
    message(FATAL_ERROR "Could not find torch in the current Python environment.")
endif()

# Find Torch Python library (torch_python)
find_library(
    TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${TORCH_PYTHON_PATH}/lib" "${TORCH_PYTHON_PATH}"
    NO_DEFAULT_PATH
)

if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Torch Python library not found.")
endif()

# ------------------------------
# 9. Add Source Files
# ------------------------------
file(GLOB_RECURSE SOURCES "src/FFMPY/*.cpp" "src/FFMPY/conversion/*.cpp" "src/FFMPY/cuda/*.cu" "src/FFMPY/backends/*.cpp")

# Create a static library target for FFMPyLib
add_library(FFMPyLib STATIC ${SOURCES})

# Specify include directories for FFMPyLib
target_include_directories(FFMPyLib PUBLIC
    ${CMAKE_SOURCE_DIR}/include/FFMPY
    ${CMAKE_SOURCE_DIR}/include/FFMPY/conversion
    ${CMAKE_SOURCE_DIR}/include/FFMPY/conversion/cpu
    ${CMAKE_SOURCE_DIR}/include/FFMPY/error
    ${CMAKE_SOURCE_DIR}/include/FFMPY/backends
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
    ${FFMPEG_INCLUDE_DIRS}
    "${CUDA_TOOLKIT_ROOT_DIR}/include"  # Add CUDA include directories
    ${PYTHON_INCLUDE_DIR}
)

# Set properties for FFMPyLib
set_target_properties(FFMPyLib PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Link FFmpeg static libraries, CUDA, and WMF libraries to FFMPyLib
target_link_libraries(FFMPyLib PUBLIC
    ws2_32
    bcrypt
    secur32
    mfplat
    strmiids
    ${MFUUUID_LIB}
    ${MF_LIB}
    ${OLE32_LIB}
    ${FFMPEG_STATIC_LIBS}
    ${TORCH_LIBRARIES}
    CUDA::cudart
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib"  # CUDA driver library
    #"${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvcuvid.lib"  # CUDA Video Decoder library
)

# ---------------------------
# ------------------------------
# 11. Add Pybind11 Bindings as a Separate Module (Manual Handling)
# ------------------------------

# Instead of using pybind11_add_module, we'll manually create the Python module
add_library(ffmpy MODULE  ${CMAKE_SOURCE_DIR}/src/FFMPY/Python/Bindings.cpp)

target_compile_options(ffmpy PRIVATE /O2 /Ob2 /Oi /Ot /GL)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

# Specify include directories for the bindings
target_include_directories(ffmpy PUBLIC
    ${CMAKE_SOURCE_DIR}/include/FFMPY
    ${CMAKE_SOURCE_DIR}/include/FFMPY/python
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include  # Include PyTorch API headers
    ${FFMPEG_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
)

# Define the module properties to match Python's expectations
set_target_properties(ffmpy PROPERTIES
    PREFIX ""  # Typically empty on Windows
    SUFFIX ".pyd"  # Typically .pyd on Windows
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Ensure the output directory exists
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/ffmpy")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")  # Directory for .lib and .exp files

# Set the output directory for the .pyd and .dll files to FFMPY/ffmpy
set_target_properties(ffmpy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/ffmpy"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/ffmpy"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"  # Redirect .lib files
    # Note: .exp files are typically placed alongside .lib; CMake doesn't provide a direct property to change their location
)

# Link libraries to the bindings
target_link_libraries(ffmpy PUBLIC
    FFMPyLib
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}          # Link against torch_python
    CUDA::cudart
    ${FFMPEG_STATIC_LIBS}
    ${PYTHON_LIBRARY}
)

# ------------------------------
# 12. Optional: Clean Up Other Output Directories
# ------------------------------

# Set FFMPyLib output directory for static libraries
set_target_properties(FFMPyLib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
