#include <chrono>
#include <iocsv/csv_writer.hpp>
#include <iomanip>
#include <iostream>
#include <linalg/tensordot.hpp>
#include <vector>

using namespace linalg;
using namespace iocsv;

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
  // 期待値の計算: Result[i][j] = Σ(k=0 to K-1) A[k][i][j] * B[k]
  // A[k][i][j] = 1.0, B[k] = 0.5 より、各項は 0.5
  // K=90 なので、合計 = 0.5 × 90 = 45.0
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

  // CSV形式で結果を出力
  std::cout << "結果をCSVファイルに出力中..." << std::endl;
  if (write_csv(Result, M, N, "result.csv"))
  {
    std::cout << "結果を result.csv に出力しました。" << std::endl;
  }
  else
  {
    std::cerr << "CSVファイルの書き込みに失敗しました。" << std::endl;
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
