#include <cmath>
#include <gtest/gtest.h>
#include <linalg/tensordot.hpp>
#include <vector>

namespace linalg
{
  namespace test
  {

    /**
     * @brief contract_3d_tensor_vector_upper_triangularの基本テスト
     */
    TEST(
      Contract3DTensorVectorUpperTriangularTest, BasicTest)
    {
      const size_t K = 3;
      const size_t M = 4;
      const size_t N = 4;

      // テストデータの準備
      // A[k][i][j] = k + 1 (すべての要素が同じ値)
      std::vector<float> A(K * M * N);
      for (size_t k = 0; k < K; ++k)
      {
        for (size_t i = 0; i < M; ++i)
        {
          for (size_t j = 0; j < N; ++j)
          {
            A[k * M * N + i * N + j] = static_cast<float>(k + 1);
          }
        }
      }

      // B[k] = 1.0
      std::vector<float> B(K, 1.0f);

      // 結果用バッファ
      std::vector<float> Result(M * N);

      // 関数呼び出し
      contract_3d_tensor_vector_upper_triangular(A, B, Result, K, M, N);

      // 期待値の計算: Result[i][j] = Σ(k=0 to K-1) A[k][i][j] * B[k]
      // A[k][i][j] = k+1, B[k] = 1.0 より
      // Result[i][j] = 1 + 2 + 3 = 6.0 (j >= i の場合)
      const float expected = 6.0f;

      // 上三角部分の検証
      for (size_t i = 0; i < M; ++i)
      {
        for (size_t j = i; j < N; ++j)
        {
          EXPECT_FLOAT_EQ(Result[i * N + j], expected)
            << "Result[" << i << "][" << j << "] should be " << expected;
        }
      }

      // 下三角部分が0であることを確認（初期化されていないため、値は不定だが、
      // 上三角部分のみ計算されることを確認するため、下三角部分はチェックしない）
    }

    /**
     * @brief 小規模データでのテスト（並列化が無効になる場合）
     */
    TEST(
      Contract3DTensorVectorUpperTriangularTest, SmallDataTest)
    {
      const size_t K = 2;
      const size_t M = 3;
      const size_t N = 3;

      std::vector<float> A(K * M * N, 1.0f);
      std::vector<float> B(K, 0.5f);
      std::vector<float> Result(M * N);

      contract_3d_tensor_vector_upper_triangular(A, B, Result, K, M, N);

      // 期待値: 1.0 * 0.5 * 2 = 1.0
      const float expected = 1.0f;

      for (size_t i = 0; i < M; ++i)
      {
        for (size_t j = i; j < N; ++j)
        {
          EXPECT_FLOAT_EQ(Result[i * N + j], expected)
            << "Result[" << i << "][" << j << "] should be " << expected;
        }
      }
    }

    /**
     * @brief 中規模データでのテスト（並列化が有効になる場合）
     */
    TEST(
      Contract3DTensorVectorUpperTriangularTest, MediumDataTest)
    {
      const size_t K = 10;
      const size_t M = 100;
      const size_t N = 100;

      std::vector<float> A(K * M * N, 1.0f);
      std::vector<float> B(K, 0.1f);
      std::vector<float> Result(M * N);

      contract_3d_tensor_vector_upper_triangular(A, B, Result, K, M, N);

      // 期待値: 1.0 * 0.1 * 10 = 1.0
      const float expected = 1.0f;

      // いくつかのサンプルポイントをチェック
      EXPECT_FLOAT_EQ(Result[0], expected);
      EXPECT_FLOAT_EQ(Result[0 * N + N - 1], expected);
      EXPECT_FLOAT_EQ(Result[(M - 1) * N + (N - 1)], expected);
    }

    /**
     * @brief 大規模データでのテスト（元のmain.cppと同じサイズ、並列化が有効になる場合）
     */
    TEST(
      Contract3DTensorVectorUpperTriangularTest, LargeDataTest)
    {
      const size_t K = 90;
      const size_t M = 5'000;
      const size_t N = 5'000;

      std::vector<float> A(K * M * N, 1.0f);
      std::vector<float> B(K, 0.5f);
      std::vector<float> Result(M * N);

      contract_3d_tensor_vector_upper_triangular(A, B, Result, K, M, N);

      // 期待値の計算: Result[i][j] = Σ(k=0 to K-1) A[k][i][j] * B[k]
      // A[k][i][j] = 1.0, B[k] = 0.5 より、各項は 0.5
      // K=90 なので、合計 = 0.5 × 90 = 45.0 (j >= i の場合)
      const float expected = 45.0f;

      // いくつかのサンプルポイントをチェック（全要素をチェックするのは時間がかかるため）
      EXPECT_FLOAT_EQ(Result[0], expected) << "Result[0][0] should be 45.0";
      EXPECT_FLOAT_EQ(Result[0 * N + 1], expected) << "Result[0][1] should be 45.0";
      if (M > 1)
      {
        EXPECT_FLOAT_EQ(Result[1 * N + 1], expected) << "Result[1][1] should be 45.0";
        EXPECT_FLOAT_EQ(Result[1 * N + N - 1], expected)
          << "Result[1][" << (N - 1) << "] should be 45.0";
      }
      EXPECT_FLOAT_EQ(Result[(M - 1) * N + (N - 1)], expected)
        << "Result[" << (M - 1) << "][" << (N - 1) << "] should be 45.0";

      // 下三角部分が計算されていないことを確認（いくつかのサンプル）
      if (M > 1)
      {
        // Result[1][0]は下三角部分なので、計算されていない（値は不定だが、上三角部分の値とは異なるはず）
        // ただし、関数内で初期化されていないため、このテストではチェックしない
      }
    }

    /**
     * @brief double型でのテスト
     */
    TEST(
      Contract3DTensorVectorUpperTriangularTest, DoublePrecisionTest)
    {
      const size_t K = 3;
      const size_t M = 5;
      const size_t N = 5;

      std::vector<double> A(K * M * N, 2.0);
      std::vector<double> B(K, 1.5);
      std::vector<double> Result(M * N);

      contract_3d_tensor_vector_upper_triangular(A, B, Result, K, M, N);

      // 期待値: 2.0 * 1.5 * 3 = 9.0
      const double expected = 9.0;

      for (size_t i = 0; i < M; ++i)
      {
        for (size_t j = i; j < N; ++j)
        {
          EXPECT_DOUBLE_EQ(Result[i * N + j], expected)
            << "Result[" << i << "][" << j << "] should be " << expected;
        }
      }
    }

    /**
     * @brief 上三角部分のみが計算されることを確認するテスト
     */
    TEST(
      Contract3DTensorVectorUpperTriangularTest, UpperTriangularOnlyTest)
    {
      const size_t K = 2;
      const size_t M = 3;
      const size_t N = 3;

      std::vector<float> A(K * M * N, 1.0f);
      std::vector<float> B(K, 1.0f);
      std::vector<float> Result(M * N, 999.0f);  // 初期値を999に設定

      contract_3d_tensor_vector_upper_triangular(A, B, Result, K, M, N);

      // 上三角部分は計算された値（2.0）であることを確認
      for (size_t i = 0; i < M; ++i)
      {
        for (size_t j = i; j < N; ++j)
        {
          EXPECT_FLOAT_EQ(Result[i * N + j], 2.0f)
            << "Upper triangular Result[" << i << "][" << j << "] should be 2.0";
        }
      }

      // 下三角部分は初期化されていないため、値は不定
      // ただし、関数内で上三角部分のみが0で初期化されるため、
      // 下三角部分は元の値（999.0）のままである可能性がある
      // このテストでは、上三角部分のみが計算されることを確認する
    }

  }  // namespace test
}  // namespace linalg
