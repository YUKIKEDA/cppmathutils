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

# GoogleTestの設定
enable_testing()

# GoogleTestをvcpkgから取得
find_package(GTest REQUIRED)

# テスト実行可能ファイル
add_executable(tensordot_test
  tests/linalg/tensordot_test.cpp
)

target_link_libraries(tensordot_test
  PRIVATE
    GTest::gtest_main
    GTest::gtest
    OpenMP::OpenMP_CXX
)

target_include_directories(tensordot_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# テストを登録
include(GoogleTest)
gtest_discover_tests(tensordot_test)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(tensordot_test PRIVATE /W4)
else()
  target_compile_options(tensordot_test PRIVATE -Wall -Wextra -pedantic)
endif()

# TODO: 必要な場合は、ターゲットをインストールします。
