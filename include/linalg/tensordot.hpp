#pragma once

#include <cstddef>
#include <omp.h>
#include <vector>

namespace linalg
{

  // TODO：一般化したtensordotを実装する

  /**
   * @brief (K, M, N) のテンソル A と (K) のベクトル B を縮約し、
   *        (M, N) の行列の上三角部分のみを計算して Result に格納する関数。
   *
   * 計算式: Result[i][j] = sum_k( A[k][i][j] * B[k] )  (ただし j >= i のみ)
   *
   * @tparam T float または double
   * @param A      入力テンソルデータ (フラット配列: サイズ K*M*N)
   * @param B      入力ベクトルデータ (サイズ K)
   * @param Result 出力用バッファ (サイズ M*N, 事前に確保が必要)
   * @param K      縮約する次元のサイズ
   * @param M      行数
   * @param N      列数
   */
  template <typename T>
  void contract_3d_tensor_vector_upper_triangular(
      const T* A, const T* B, T* Result, size_t K, size_t M, size_t N)
  {
    // 行(i)ごとに並列処理
    // dynamicスケジュールにより、計算量が多い行(上の方)と少ない行(下の方)の負荷分散を行う
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(M); ++i)
    {
      // 結果行列の行 i の先頭ポインタ
      T* res_row = &Result[static_cast<size_t>(i) * N];

      // 上三角部分のみを0で初期化
      for (size_t j = static_cast<size_t>(i); j < N; ++j)
      {
        res_row[j] = static_cast<T>(0);
      }

      // 縮約次元 K のループ
      for (size_t k = 0; k < K; ++k)
      {
        T b_val = B[k];

        // Aの該当データ位置: k枚目のスライスの、i行目
        // オフセット計算: k * (M * N) + i * N
        // 注意: 20億を超えるため必ず size_t で計算する
        const T* a_row = &A[k * M * N + static_cast<size_t>(i) * N];

        // 列(j)のループ: 上三角 (j >= i) のみ計算
        // コンパイラの自動ベクトル化(SIMD)を最大限効かせるポイント
        for (size_t j = static_cast<size_t>(i); j < N; ++j)
        {
          res_row[j] += a_row[j] * b_val;
        }
      }
    }
  }

  /**
   * @brief std::vector を受け取るオーバーロード（使いやすくするためのラッパー）
   */
  template <typename T>
  void contract_3d_tensor_vector_upper_triangular(
      const std::vector<T>& A,
      const std::vector<T>& B,
      std::vector<T>& Result,
      size_t K,
      size_t M,
      size_t N)
  {
    contract_3d_tensor_vector_upper_triangular(A.data(), B.data(), Result.data(), K, M, N);
  }

}  // namespace linalg
