# adaptive_metropolis_test実行可能ファイル
add_executable(adaptive_metropolis_test
  ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_metropolis_test.cpp
)

target_link_libraries(adaptive_metropolis_test
  PRIVATE
    GTest::gtest_main
    GTest::gtest
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

target_include_directories(adaptive_metropolis_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

# テストを登録（作業ディレクトリをプロジェクトルートに設定）
gtest_discover_tests(adaptive_metropolis_test
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(adaptive_metropolis_test PRIVATE /W4)
else()
  target_compile_options(adaptive_metropolis_test PRIVATE -Wall -Wextra -pedantic)
endif()
