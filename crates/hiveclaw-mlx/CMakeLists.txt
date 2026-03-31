cmake_minimum_required(VERSION 3.21)
project(hiveclaw_mlx_ext LANGUAGES CXX OBJCXX)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))"
  OUTPUT_VARIABLE _NB_CMAKE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _NB_RES)
if(NOT _NB_RES EQUAL 0 OR _NB_CMAKE STREQUAL "")
  message(FATAL_ERROR "nanobind not found for this Python; pip install nanobind")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_NB_CMAKE}")
find_package(nanobind CONFIG REQUIRED)

if(NOT DEFINED ENV{MLX_DIR})
  message(FATAL_ERROR "MLX_DIR is not set (mlx.extension CMakeBuild sets it)")
endif()
find_package(MLX REQUIRED PATHS "$ENV{MLX_DIR}")

set(SOURCES
    src/slab_bridge.mm
    src/slab_primitives.cpp
    src/slab_kernels.cpp
    src/bindings.cpp)

set_source_files_properties(
    src/slab_bridge.mm PROPERTIES
    COMPILE_LANGUAGE OBJCXX
    COMPILE_OPTIONS "-fobjc-arc")

nanobind_add_module(
    hiveclaw_mlx_ext
    STABLE_ABI
    NB_DOMAIN mlx
    ${SOURCES})

target_link_libraries(hiveclaw_mlx_ext PRIVATE mlx)
target_link_libraries(
    hiveclaw_mlx_ext PRIVATE
    "-framework IOSurface"
    "-framework Metal"
    "-framework Foundation")

set_target_properties(
    hiveclaw_mlx_ext PROPERTIES
    CXX_STANDARD 17
    OBJCXX_STANDARD 17
    MACOSX_RPATH ON)
