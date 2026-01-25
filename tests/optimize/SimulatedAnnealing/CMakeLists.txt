# simulated_annealing_test実行可能ファイル
add_executable(simulated_annealing_test
  ${CMAKE_CURRENT_SOURCE_DIR}/simulated_annealing_test.cpp
)

target_link_libraries(simulated_annealing_test
  PRIVATE
    GTest::gtest_main
    GTest::gtest
    nlohmann_json::nlohmann_json
)

target_include_directories(simulated_annealing_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

# テストを登録（作業ディレクトリをプロジェクトルートに設定）
gtest_discover_tests(simulated_annealing_test
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# コンパイラ固有の設定（テスト用）
if (MSVC)
  target_compile_options(simulated_annealing_test PRIVATE /W4)
else()
  target_compile_options(simulated_annealing_test PRIVATE -Wall -Wextra -pedantic)
endif()
