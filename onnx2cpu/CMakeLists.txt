cmake_minimum_required(VERSION 3.14)
project(onnx2cpu)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(src)

# 添加头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 设置 ONNX-MLIR 的第三方库路径
set(ONNX_MLIR_DIR ~/APP/onnx-mlir)

# 添加第三方库路径
link_directories(${ONNX_MLIR_DIR}/lib)
