# gp_regression_test実行可能ファイル
add_executable(gp_regression_test
  ${CMAKE_CURRENT_SOURCE_DIR}/gp_regression_test.cpp
)

target_link_libraries(gp_regression_test
  PRIVATE
    GTest::gtest_main
    GTest::gtest
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

target_include_directories(gp_regression_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

# テストを登録（作業ディレクトリをプロジェクトルートに設定）
gtest_discover_tests(gp_regression_test
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(gp_regression_test PRIVATE /W4)
else()
  target_compile_options(gp_regression_test PRIVATE -Wall -Wextra -pedantic)
endif()
