cmake_minimum_required(VERSION 3.23)
project(CeLux VERSION 1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CELUX_ENABLE_CUDA   "Enable CUDA support in CeLux" OFF)
option(CELUX_BUILD_PYTHON "Build Python bindings"     ON)

# --- Compiler cache (ccache) ---
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  if(CELUX_ENABLE_CUDA)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  endif()
endif()

# --- Dependencies ---
find_package(spdlog CONFIG REQUIRED)
find_package(fmt    CONFIG REQUIRED)
if(NOT TARGET fmt::fmt-header_only AND TARGET fmt::fmt)
  add_library(fmt::fmt-header_only ALIAS fmt::fmt)
endif()

find_package(FFMPEG REQUIRED
  COMPONENTS avcodec avformat avutil swscale swresample
)
set(CEFLUX_FFMPEG_LIBS ${FFMPEG_LIBRARIES})
set(CEFLUX_FFMPEG_INCS ${FFMPEG_INCLUDE_DIRS})

if(CELUX_BUILD_PYTHON)
  find_package(Python3 3.13 REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "Python3: ${Python3_EXECUTABLE}")
  message(STATUS "Python3 includes: ${Python3_INCLUDE_DIRS}")

  # -- Find pybind11 --
  find_package(pybind11 CONFIG)
  if(pybind11_FOUND)
    message(STATUS "pybind11 (CMake): ${pybind11_VERSION}")
    set(CEFLUX_PYBIND11_INCS ${pybind11_INCLUDE_DIRS})
    set(CEFLUX_PYBIND11_LIBS pybind11::module)
  else()
    message(STATUS "pybind11 not found via CMake; falling back to pip headers")
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
              -c "import site; print(site.getsitepackages()[0])"
      OUTPUT_VARIABLE Python3_SITE_PACKAGES
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python site-packages: ${Python3_SITE_PACKAGES}")

    if(WIN32)
      set(CEFLUX_PYBIND11_INCS
          "${Python3_SITE_PACKAGES}/Lib/site-packages/pybind11/include")
    else()
      set(CEFLUX_PYBIND11_INCS
          "${Python3_SITE_PACKAGES}/pybind11/include")
    endif()
    set(CEFLUX_PYBIND11_LIBS "")
    if(NOT EXISTS "${CEFLUX_PYBIND11_INCS}/pybind11/pybind11.h")
      message(FATAL_ERROR "pybind11 headers missing; run: pip install pybind11")
    endif()
  endif()

  # -- PyTorch paths (platform-specific) --
  if(WIN32)
    set(CEFLUX_PYTORCH_LIBDIR "${Python3_SITE_PACKAGES}/Lib/site-packages/torch/lib")
    set(CEFLUX_PYTORCH_INCLUDEDIR "${Python3_SITE_PACKAGES}/Lib/site-packages/torch/include")
  else()
    set(CEFLUX_PYTORCH_LIBDIR "${Python3_SITE_PACKAGES}/torch/lib")
    set(CEFLUX_PYTORCH_INCLUDEDIR "${Python3_SITE_PACKAGES}/torch/include")
  endif()

  # Main headers: torch/include and torch/include/torch/csrc/api/include
  include_directories(
    ${CEFLUX_PYTORCH_INCLUDEDIR}
    ${CEFLUX_PYTORCH_INCLUDEDIR}/torch/csrc/api/include
    ${CEFLUX_PYTORCH_INCLUDEDIR}
  )
  # Add the lib directory to the linker
  link_directories(${CEFLUX_PYTORCH_LIBDIR})

  message(STATUS "Using Python site-packages torch headers: ${CEFLUX_PYTORCH_INCLUDEDIR}")
  message(STATUS "Using Python site-packages torch libs:    ${CEFLUX_PYTORCH_LIBDIR}")
endif()

file(GLOB_RECURSE CEFLUX_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/conversion/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CeLux/backends/*.cpp
)

add_library(CeLuxLib STATIC ${CEFLUX_SOURCES})

target_include_directories(CeLuxLib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/backends>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/python>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/conversion>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/conversion/cpu>
    $<INSTALL_INTERFACE:include>
    ${CEFLUX_FFMPEG_INCS}
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:${Python3_INCLUDE_DIRS}>
)

target_link_libraries(CeLuxLib PUBLIC
  ${CEFLUX_FFMPEG_LIBS}
  spdlog::spdlog_header_only
  fmt::fmt-header_only
  $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch_cpu>
  $<$<BOOL:${CELUX_BUILD_PYTHON}>:c10>
  $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch>
  $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch_python>
)
if(WIN32)
  target_link_libraries(CeLuxLib PUBLIC ws2_32 bcrypt)
endif()
# If you want to add Linux-specific libs:
#if(UNIX AND NOT APPLE)
#  target_link_libraries(CeLuxLib PUBLIC pthread dl rt)
#endif()

if(CELUX_BUILD_PYTHON)
  add_library(celux MODULE src/CeLux/python/Bindings.cpp)

  target_include_directories(celux PRIVATE
    ${CEFLUX_PYBIND11_INCS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CeLux/python
  )

  target_link_libraries(celux PRIVATE
    CeLuxLib
    Python3::Python
    ${CEFLUX_PYBIND11_LIBS}
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch_cpu>
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:c10>
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch>
    $<$<BOOL:${CELUX_BUILD_PYTHON}>:torch_python>
  )

  set_target_properties(celux PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/celux
    OUTPUT_NAME "celux"
  )
  if(WIN32)
    set_target_properties(celux PROPERTIES SUFFIX ".pyd")
  else()
    set_target_properties(celux PROPERTIES SUFFIX ".so")
  endif()

  # RPATH for Linux (optional, to run .so without LD_LIBRARY_PATH)
  if(UNIX AND NOT APPLE)
    set_target_properties(celux PROPERTIES
      INSTALL_RPATH "$ORIGIN"
      BUILD_WITH_INSTALL_RPATH TRUE
    )
  endif()
endif()

if(WIN32 AND CELUX_BUILD_PYTHON)
  # DLL copy logic for Windows only
  if(DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_BIN "$ENV{VCPKG_ROOT}/installed/x64-windows/bin")
  else()
    set(VCPKG_BIN "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin")
  endif()
  set(CEFLUX_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/celux")
  file(GLOB CEFLUX_DLLS
    "${VCPKG_BIN}/av*.dll"
    "${VCPKG_BIN}/sw*.dll"
    "${VCPKG_BIN}/ffmpeg*.dll"
    "${VCPKG_BIN}/libx264*.dll"
    "${VCPKG_BIN}/fmt.dll"
  )
  foreach(dll ${CEFLUX_DLLS})
    add_custom_command(TARGET celux POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "${CEFLUX_DLL_DIR}"
      COMMENT "Copying dependency DLL: ${dll} → ${CEFLUX_DLL_DIR}"
    )
  endforeach()
endif()

include(GNUInstallDirs)
install(TARGETS CeLuxLib EXPORT CeLuxTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT CeLuxTargets NAMESPACE CeLux::
  FILE CeLuxTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeLux
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfigVersion.cmake
  VERSION     ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cmake/CeLuxConfig.cmake.in")
  configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/CeLuxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeLux
  )
else()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfig.cmake"
"@PACKAGE_INIT@\ninclude(\"${CMAKE_CURRENT_LIST_DIR}/CeLuxTargets.cmake\")\n")
endif()

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CeLuxConfigVersion.cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/celux/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/celux/py.typed
        # ${CMAKE_CURRENT_SOURCE_DIR}/celux/celux.pyi   # If you have a .pyi file!
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CeLux
  )
# debug dump of the variables we care about
message(STATUS "---- CMake Debug Dump ----")
message(STATUS "  CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "  VCPKG_ROOT (env)       = $ENV{VCPKG_ROOT}")
message(STATUS "  VCPKG_TARGET_TRIPLET   = $ENV{VCPKG_TARGET_TRIPLET}")
message(STATUS "  CMAKE_PREFIX_PATH      = ${CMAKE_PREFIX_PATH}")
message(STATUS "  CMAKE_MODULE_PATH      = ${CMAKE_MODULE_PATH}")
message(STATUS "  Python3_EXECUTABLE     = ${Python3_EXECUTABLE}")
message(STATUS "  Python3_ROOT_DIR       = ${Python3_ROOT_DIR}")
message(STATUS "----------------------------")

