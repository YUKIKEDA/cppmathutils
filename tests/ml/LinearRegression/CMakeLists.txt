# linear_regression_test実行可能ファイル
add_executable(linear_regression_test
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_regression_test.cpp
)

target_link_libraries(linear_regression_test
  PRIVATE
    GTest::gtest_main
    GTest::gtest
    Eigen3::Eigen
)

target_include_directories(linear_regression_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# テストを登録
gtest_discover_tests(linear_regression_test)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(linear_regression_test PRIVATE /W4)
else()
  target_compile_options(linear_regression_test PRIVATE -Wall -Wextra -pedantic)
endif()
