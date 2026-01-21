# CMakeList.txt : cppmathutils の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。
#
cmake_minimum_required (VERSION 3.8)

# サポートされている場合は、MSVC コンパイラのホット リロードを有効にします。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("cppmathutils" LANGUAGES CXX)

# C++20標準を設定
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# インクルードディレクトリを追加
include_directories(${CMAKE_SOURCE_DIR}/include)

# ソースファイル
set(SOURCES
  src/main.cpp
)

# 実行可能ファイルを作成
add_executable (cppmathutils ${SOURCES})

# OpenMPの検出とリンク
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found, version ${OpenMP_CXX_VERSION}")
  target_link_libraries(cppmathutils PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found. Performance may be degraded.")
endif()

# コンパイラ固有の設定
if (MSVC)
  # MSVC用の設定
  target_compile_options(cppmathutils PRIVATE /W4)
else()
  # GCC/Clang用の設定
  target_compile_options(cppmathutils PRIVATE -Wall -Wextra -pedantic)
endif()

# TODO: テストを追加し、必要な場合は、ターゲットをインストールします。
