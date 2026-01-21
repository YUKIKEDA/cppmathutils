#include "../include/linalg/tensordot.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace linalg;

/**
 * @brief tensordot_upper_triangularのテスト
 */
void test_upper_triangular()
{
  std::cout << "=== tensordot_upper_triangularテスト ===" << std::endl;

  // サイズ設定
  const size_t K = 90;
  const size_t M = 5'000;
  const size_t N = 5'000;

  std::cout << "メモリ確保中..." << std::endl;
  // メモリ確保 (Aは巨大なのでスタックではなくヒープ/vectorを使う)
  std::vector<float> A(K * M * N, 1.0f);
  std::vector<float> B(K, 0.5f);
  std::vector<float> Result(M * N);

  std::cout << "計算中..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  // 関数呼び出し
  tensordot_upper_triangular(A, B, Result, K, M, N);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "計算時間: " << std::fixed << std::setprecision(3) << elapsed.count() << " 秒"
            << std::endl;

  // 結果確認
  std::cout << "Result[0][0]: " << Result[0] << " (期待値: 45.0)" << std::endl;

  // 下三角部分が0であることを確認
  if (M > 1)
  {
    std::cout << "Result[1][0]: " << Result[1 * N + 0] << " (期待値: 0.0)" << std::endl;
  }

  // 上三角部分のいくつかの値を確認
  std::cout << "Result[0][1]: " << Result[0 * N + 1] << " (期待値: 45.0)" << std::endl;
  if (M > 1 && N > 1)
  {
    std::cout << "Result[1][1]: " << Result[1 * N + 1] << " (期待値: 45.0)" << std::endl;
  }
}

int main()
{
  std::cout << "=== Tensordot Upper Triangular テストプログラム ===" << std::endl;

  try
  {
    // 上三角行列専用のtensordotのテスト
    test_upper_triangular();

    std::cout << "\n=== テスト完了 ===" << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cerr << "エラー: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
