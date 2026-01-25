# simple_linear_regression_test実行可能ファイル
add_executable(simple_linear_regression_test
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_linear_regression_test.cpp
)

target_link_libraries(simple_linear_regression_test
  PRIVATE
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(simple_linear_regression_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# テストを登録
gtest_discover_tests(simple_linear_regression_test)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(simple_linear_regression_test PRIVATE /W4)
else()
  target_compile_options(simple_linear_regression_test PRIVATE -Wall -Wextra -pedantic)
endif()
