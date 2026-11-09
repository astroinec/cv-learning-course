cmake_minimum_required(VERSION 3.16)
project(ch15_cpp_vision CXX)

# 用法（需要先 brew install cmake）：
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/ch15_tests
# 想对照 OpenCV：brew install opencv 后加 -DWITH_OPENCV=ON

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(ch15 STATIC "15-1_图像基础.cpp")
target_include_directories(ch15 PUBLIC cpp)
target_compile_options(ch15 PRIVATE -Wall -Wextra)

add_executable(ch15_tests tests/test_main.cpp)
target_link_libraries(ch15_tests PRIVATE ch15)

# Debug 时打开地址和未定义行为检查，越界访问会立刻报出具体位置。
option(WITH_SANITIZERS "开启 AddressSanitizer 和 UBSan" OFF)
if(WITH_SANITIZERS)
  foreach(t ch15 ch15_tests)
    target_compile_options(${t} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${t} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

option(WITH_OPENCV "编译 OpenCV 对照程序" OFF)
if(WITH_OPENCV)
  find_package(OpenCV REQUIRED)
  add_executable(ch15_opencv_compare cpp/opencv_compare.cpp)
  target_link_libraries(ch15_opencv_compare PRIVATE ch15 ${OpenCV_LIBS})
endif()
