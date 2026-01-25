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

# multiple_linear_regression_test実行可能ファイル
add_executable(multiple_linear_regression_test
  ${CMAKE_CURRENT_SOURCE_DIR}/multiple_linear_regression_test.cpp
)

target_link_libraries(multiple_linear_regression_test
  PRIVATE
    GTest::gtest_main
    GTest::gtest
    Eigen3::Eigen
)

target_include_directories(multiple_linear_regression_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# テストを登録
gtest_discover_tests(multiple_linear_regression_test)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(multiple_linear_regression_test PRIVATE /W4)
else()
  target_compile_options(multiple_linear_regression_test PRIVATE -Wall -Wextra -pedantic)
endif()

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

# ridge_regression_test実行可能ファイル
add_executable(ridge_regression_test
  ${CMAKE_CURRENT_SOURCE_DIR}/ridge_regression_test.cpp
)

target_link_libraries(ridge_regression_test
  PRIVATE
    GTest::gtest_main
    GTest::gtest
    Eigen3::Eigen
)

target_include_directories(ridge_regression_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# テストを登録
gtest_discover_tests(ridge_regression_test)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(ridge_regression_test PRIVATE /W4)
else()
  target_compile_options(ridge_regression_test PRIVATE -Wall -Wextra -pedantic)
endif()

# GaussianProcessRegressionサブディレクトリを追加
add_subdirectory(GaussianProcessRegression)
