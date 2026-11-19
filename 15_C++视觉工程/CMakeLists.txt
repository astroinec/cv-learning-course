cmake_minimum_required(VERSION 3.16)
project(ch15_cpp_vision CXX)

# 用法（在本章目录下运行；CMake 和编译器的安装见 docs/环境安装.md）。
#
# Windows，Visual Studio 生成工具（在“x64 Native Tools Command Prompt”里运行）：
#   cmake -S . -B build
#   cmake --build build --config Release
#   build\Release\ch15_tests.exe
#
# Windows，MinGW-w64 的 g++（要另外装 Ninja 或 make，下面以 Ninja 为例）：
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   build\ch15_tests.exe
#
# macOS 和 Linux：
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ./build/ch15_tests
#
# ch15_tests 只输出英文的状态和代码（PASS|key|code|...），中文说明由 tests/check_progress.py 给出。
# 想对照 OpenCV：装好 C++ 版 OpenCV 后加 -DWITH_OPENCV=ON。

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 源文件是 UTF-8 编码，里面有中文注释和字符串。GCC 和 Clang 默认按 UTF-8 读，MSVC 要加 /utf-8。
if(MSVC)
  add_compile_options(/utf-8)
endif()

add_library(ch15 STATIC "15-1_图像基础.cpp")
target_include_directories(ch15 PUBLIC cpp)
if(MSVC)
  target_compile_options(ch15 PRIVATE /W4)
else()
  target_compile_options(ch15 PRIVATE -Wall -Wextra)
endif()

add_executable(ch15_tests tests/test_main.cpp)
target_link_libraries(ch15_tests PRIVATE ch15)

# Debug 时打开地址和未定义行为检查，越界访问会立刻报出具体位置。
# 这组选项是 GCC 和 Clang 的写法，只在非 MSVC 的编译器下添加。
option(WITH_SANITIZERS "开启 AddressSanitizer 和 UBSan" OFF)
if(WITH_SANITIZERS)
  if(MSVC)
    message(WARNING "WITH_SANITIZERS 只支持 GCC 和 Clang，MSVC 下已忽略这个选项。")
  else()
    foreach(t ch15 ch15_tests)
      target_compile_options(${t} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${t} PRIVATE -fsanitize=address,undefined)
    endforeach()
  endif()
endif()

option(WITH_OPENCV "编译 OpenCV 对照程序" OFF)
if(WITH_OPENCV)
  find_package(OpenCV REQUIRED)
  add_executable(ch15_opencv_compare cpp/opencv_compare.cpp)
  target_link_libraries(ch15_opencv_compare PRIVATE ch15 ${OpenCV_LIBS})
endif()
