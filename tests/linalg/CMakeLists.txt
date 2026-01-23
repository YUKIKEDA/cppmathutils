# tensordot_test実行可能ファイル
add_executable(tensordot_test
  ${CMAKE_CURRENT_SOURCE_DIR}/tensordot_test.cpp
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
gtest_discover_tests(tensordot_test)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(tensordot_test PRIVATE /W4)
else()
  target_compile_options(tensordot_test PRIVATE -Wall -Wextra -pedantic)
endif()
