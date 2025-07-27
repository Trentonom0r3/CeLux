cmake_minimum_required(VERSION 3.23)

# ======================================================
# 1. User-configurable options
# ======================================================
option(CELUX_ENABLE_CUDA   "Build with CUDA acceleration" OFF)
option(CELUX_BUILD_PYTHON  "Build the Python extension (requires pybind11)" ON)
option(CELUX_BUILD_SHARED  "Build CeLux as a shared library" OFF)

if (NOT CELUX_ENABLE_CUDA)
    set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit ON)
endif()

# ======================================================
# 2. Project declaration
# ======================================================
if (CELUX_ENABLE_CUDA)
    project(CeLux VERSION 1.0 LANGUAGES CXX CUDA)
else()
    project(CeLux VERSION 1.0 LANGUAGES CXX)
endif()
include_directories("C:/Program Files/Python313/include")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ======================================================
# 3. Build acceleration (optional)
# ======================================================
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    message(STATUS "Enabling ccache")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    if (CELUX_ENABLE_CUDA)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    endif()
endif()

# ======================================================
# 4. Dependencies
# ======================================================
find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

if (NOT TARGET fmt::fmt-header_only AND TARGET fmt::fmt)
    add_library(fmt::fmt-header_only ALIAS fmt::fmt)
endif()

# ---- FFmpeg ----
find_package(FFMPEG REQUIRED COMPONENTS avcodec avformat avutil swscale swresample)

# ---- Python + pybind11 ----
if (CELUX_BUILD_PYTHON)
    if (NOT DEFINED Python3_EXECUTABLE OR NOT DEFINED Python3_ROOT_DIR)
        message(FATAL_ERROR
          "Missing Python config. Set Python3_EXECUTABLE and Python3_ROOT_DIR in your preset or environment."
        )
    endif()
# ==================== Hardcoded Python 3.13 =====================
# Set Python root and executable directly BEFORE any find_package(Python...)
set(Python3_ROOT_DIR "C:/Program Files/Python313" CACHE PATH "Python root directory")
set(Python3_EXECUTABLE "C:/Program Files/Python313/python.exe" CACHE FILEPATH "Python executable")
set(Python3_LIBRARIES "C:/Program Files/Python313/Lib" CACHE PATH "Lib dir")
# These help make sure CMake doesn't grab from registry or vcpkg
set(Python_FIND_REGISTRY NEVER)
set(Python_FIND_STRATEGY LOCATION)


    # Use FindPython (new module) to get your 3.13 dev files
    find_package(
      Python 3.13 REQUIRED
      COMPONENTS Development.Module Interpreter
    )

    if (Python_VERSION VERSION_LESS 3.13)
        message(FATAL_ERROR "Found Python ${Python_VERSION}, but CeLux requires ≥ 3.13.")
    endif()

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[1])"
        OUTPUT_VARIABLE Python3_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Detected site‑packages at: ${Python3_SITE_PACKAGES}")
        link_directories("${Python3_SITE_PACKAGES}/torch/lib")

    set(PYBIND11_PIP_INCLUDE
        "${Python3_SITE_PACKAGES}/pybind11/include"
    )
    if (NOT EXISTS "${PYBIND11_PIP_INCLUDE}/pybind11/pybind11.h")
        message(FATAL_ERROR
          "pybind11.h not found in ${PYBIND11_PIP_INCLUDE}. "
          "Run: pip install pybind11"
        )
    endif()

    include_directories(${PYBIND11_PIP_INCLUDE})
endif()



# ---- Torch ----
if (NOT DEFINED Torch_DIR)
    message(FATAL_ERROR "Torch_DIR not set. Please define it via environment or CMakePresets.json.")
endif()
find_package(Torch REQUIRED)

if (NOT TARGET Torch::Torch)
    if (TARGET torch)
        add_library(Torch::Torch ALIAS torch)
    elseif (TARGET torch_cpu)
        add_library(Torch::Torch ALIAS torch_cpu)
    else()
        message(FATAL_ERROR "LibTorch found but no usable target (torch / torch_cpu) exported.")
    endif()
endif()

if (NOT CELUX_ENABLE_CUDA)
    if (TARGET torch_cuda OR TORCH_CUDA_FOUND)
        message(FATAL_ERROR "CPU-only build requested but GPU-enabled libtorch detected. Use a CPU-only LibTorch build.")
    endif()
endif()
# ======================================================
# 5. Source Files
# ======================================================
file(GLOB_RECURSE CELUX_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/conversion/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/backends/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/filters/*.cpp)

if (CELUX_ENABLE_CUDA)
    file(GLOB_RECURSE CELUX_CUDA_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/cuda/*.cu)
    list(APPEND CELUX_SOURCES ${CELUX_CUDA_SOURCES})
endif()

# ======================================================
# 6. CeLux core library
# ======================================================
set(CELUX_LIB_TYPE STATIC)
if (CELUX_BUILD_SHARED)
    set(CELUX_LIB_TYPE SHARED)
endif()

add_library(CeLuxLib ${CELUX_LIB_TYPE} ${CELUX_SOURCES})

target_include_directories(CeLuxLib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${FFMPEG_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/error>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/conversion>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/conversion/cpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/backends>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/backends/cpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/filters>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/filters/generated>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/python>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/python>
)

set(FFMPEG_LIBS ${FFMPEG_LIBRARIES})

target_link_libraries(CeLuxLib PUBLIC
    ${FFMPEG_LIBS}
    Torch::Torch
    spdlog::spdlog_header_only
    fmt::fmt-header_only)

if (WIN32)
    target_link_libraries(CeLuxLib PUBLIC ws2_32 bcrypt)
endif()

if (CELUX_ENABLE_CUDA)
    target_compile_definitions(CeLuxLib PUBLIC CELUX_WITH_CUDA)
    set_property(TARGET CeLuxLib PROPERTY CUDA_STANDARD 17)
endif()


get_target_property(_celuxlib_incs CeLuxLib INCLUDE_DIRECTORIES)
if(NOT _celuxlib_incs)
  message(FATAL_ERROR "Could not read CeLuxLib include directories!")
endif()

# ======================================================
# 7. Python bindings
# ======================================================
if (CELUX_BUILD_PYTHON)
    # Build the Python extension module
    add_library(celux MODULE src/CeLux/Python/Bindings.cpp)

    # Includes: pybind11, Python, and your C++ headers
    target_include_directories(celux PRIVATE
        ${_celuxlib_incs}
        ${PYBIND11_PIP_INCLUDE}
        ${Python3_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/python
    )

# Tell the linker where to find python313.lib
link_directories("${Python3_ROOT_DIR}/libs")


    # Link against:
    # 1) your core C++ lib,
    # 2) python3.lib,
    # 3) the PyTorch “python” library and its dependencies
    target_link_libraries(celux PRIVATE
        CeLuxLib
        "${Python3_ROOT_DIR}/libs/python313.lib"

        # PyTorch C++ core (from vcpkg-find_package(Torch) or manually)
        Torch::Torch

        # PyTorch Python extension support (pip-installed)
        torch_python
        torch
        c10
        torch_cpu
        asmjit
        fbgemm
        fmt
        cpuinfo
        pthreadpool
        XNNPACK
    )

    # Emit as celux.pyd into python/ folder
    set_target_properties(celux PROPERTIES
        PREFIX ""
        SUFFIX ".pyd"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        OUTPUT_NAME "celux"
    )
endif()


# ======================================================
# 8. Install rules
# ======================================================
include(GNUInstallDirs)

install(TARGETS CeLuxLib EXPORT CeLuxTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT CeLuxTargets NAMESPACE CeLux:: FILE CeLuxTargets.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeLux)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/CeLuxConfig.cmake.in")
    configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/CeLuxConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeLux)
else()
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfig.cmake
        "# Auto-generated minimal config\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/CeLuxTargets.cmake\")\n")
endif()

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeLux)

# ======================================================
# 9. Build summary
# ======================================================
message(STATUS "\n================= CeLux build summary =================")
message(STATUS "CUDA support         : ${CELUX_ENABLE_CUDA}")
message(STATUS "Python bindings      : ${CELUX_BUILD_PYTHON}")
message(STATUS "Torch_DIR            : ${Torch_DIR}")
message(STATUS "Install prefix       : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "======================================================\n")
