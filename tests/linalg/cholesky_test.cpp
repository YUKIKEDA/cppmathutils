#include <cmath>
#include <gtest/gtest.h>
#include <linalg/cholesky.hpp>
#include <vector>

namespace linalg
{
  namespace test
  {

    /**
     * @brief cholesky_decompositionの基本テスト（小規模行列）
     */
    TEST(CholeskyDecompositionTest, BasicTest)
    {
      const size_t N = 3;

      // 対称正定値行列を作成
      // A = [4, 2, 0]
      //     [2, 5, 1]
      //     [0, 1, 3]
      std::vector<double> A = {4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0};

      std::vector<double> L(N * N);

      // コレスキー分解を実行
      bool success = cholesky_decomposition(A, L, N);

      EXPECT_TRUE(success) << "Cholesky decomposition should succeed for positive definite matrix";

      // 分解結果を検証: A ≈ L * L^T
      bool verified = verify_cholesky_decomposition(A, L, N, 1e-10);
      EXPECT_TRUE(verified) << "L * L^T should reconstruct the original matrix A";
    }

    /**
     * @brief 単位行列のコレスキー分解テスト
     */
    TEST(CholeskyDecompositionTest, IdentityMatrixTest)
    {
      const size_t N = 4;

      // 単位行列を作成
      std::vector<double> A(N * N, 0.0);
      for (size_t i = 0; i < N; ++i)
      {
        A[i * N + i] = 1.0;
      }

      std::vector<double> L(N * N);

      bool success = cholesky_decomposition(A, L, N);
      EXPECT_TRUE(success) << "Cholesky decomposition should succeed for identity matrix";

      // 単位行列のコレスキー分解は単位行列自身
      for (size_t i = 0; i < N; ++i)
      {
        for (size_t j = 0; j < N; ++j)
        {
          if (i == j)
          {
            EXPECT_NEAR(L[i * N + j], 1.0, 1e-10) << "Diagonal element should be 1.0";
          }
          else if (j < i)
          {
            EXPECT_NEAR(L[i * N + j], 0.0, 1e-10) << "Lower triangular off-diagonal should be 0.0";
          }
          else
          {
            EXPECT_NEAR(L[i * N + j], 0.0, 1e-10) << "Upper triangular should be 0.0";
          }
        }
      }
    }

    /**
     * @brief float型でのテスト
     */
    TEST(CholeskyDecompositionTest, FloatPrecisionTest)
    {
      const size_t N = 3;

      std::vector<float> A = {9.0f, 3.0f, 0.0f, 3.0f, 10.0f, 2.0f, 0.0f, 2.0f, 8.0f};

      std::vector<float> L(N * N);

      bool success = cholesky_decomposition(A, L, N);
      EXPECT_TRUE(success) << "Cholesky decomposition should succeed";

      bool verified = verify_cholesky_decomposition(A, L, N, 1e-5f);
      EXPECT_TRUE(verified) << "L * L^T should reconstruct the original matrix A";
    }

    /**
     * @brief 大規模行列でのテスト
     */
    TEST(CholeskyDecompositionTest, LargeMatrixTest)
    {
      const size_t N = 100;

      // 対称正定値行列を生成: A = B * B^T + I
      // これにより正定値性が保証される
      std::vector<double> B(N * N);
      for (size_t i = 0; i < N; ++i)
      {
        for (size_t j = 0; j < N; ++j)
        {
          B[i * N + j] = static_cast<double>(i + j + 1) / static_cast<double>(N);
        }
      }

      // A = B * B^T + I を計算
      std::vector<double> A(N * N, 0.0);
      for (size_t i = 0; i < N; ++i)
      {
        for (size_t j = 0; j < N; ++j)
        {
          double sum = 0.0;
          for (size_t k = 0; k < N; ++k)
          {
            sum += B[i * N + k] * B[j * N + k];
          }
          A[i * N + j] = sum + (i == j ? 1.0 : 0.0);
        }
      }

      std::vector<double> L(N * N);

      bool success = cholesky_decomposition(A, L, N);
      EXPECT_TRUE(success)
        << "Cholesky decomposition should succeed for large positive definite matrix";

      bool verified = verify_cholesky_decomposition(A, L, N, 1e-8);
      EXPECT_TRUE(verified) << "L * L^T should reconstruct the original matrix A";
    }

    /**
     * @brief エラーハンドリングテスト（nullポインタ）
     */
    TEST(CholeskyDecompositionTest, NullPointerTest)
    {
      const size_t N = 3;
      std::vector<double> A(N * N, 1.0);
      std::vector<double> L(N * N);

      EXPECT_THROW(cholesky_decomposition(static_cast<const double*>(nullptr), L.data(), N),
        std::invalid_argument);
      EXPECT_THROW(
        cholesky_decomposition(A.data(), static_cast<double*>(nullptr), N), std::invalid_argument);
    }

    /**
     * @brief エラーハンドリングテスト（サイズ不一致）
     */
    TEST(CholeskyDecompositionTest, SizeMismatchTest)
    {
      const size_t N = 3;
      std::vector<double> A(N * N, 1.0);
      std::vector<double> L_wrong_size(N * N - 1);  // サイズが間違っている

      EXPECT_THROW(cholesky_decomposition(A, L_wrong_size, N), std::invalid_argument);
    }

    /**
     * @brief エラーハンドリングテスト（N=0）
     */
    TEST(CholeskyDecompositionTest, ZeroSizeTest)
    {
      std::vector<double> A(1);
      std::vector<double> L(1);

      EXPECT_THROW(cholesky_decomposition(A.data(), L.data(), 0), std::invalid_argument);
    }

    /**
     * @brief 下三角行列の構造を確認するテスト
     */
    TEST(CholeskyDecompositionTest, LowerTriangularStructureTest)
    {
      const size_t N = 5;

      // 対称正定値行列を作成
      std::vector<double> A(N * N);
      for (size_t i = 0; i < N; ++i)
      {
        for (size_t j = 0; j < N; ++j)
        {
          if (i == j)
          {
            A[i * N + j] = static_cast<double>(N + i + 1);  // 対角要素を大きくする
          }
          else
          {
            A[i * N + j] = static_cast<double>(std::min(i, j) + 1);  // 対称行列
          }
        }
      }

      std::vector<double> L(N * N);

      bool success = cholesky_decomposition(A, L, N);
      EXPECT_TRUE(success);

      // 上三角部分が0であることを確認
      for (size_t i = 0; i < N; ++i)
      {
        for (size_t j = i + 1; j < N; ++j)
        {
          EXPECT_NEAR(L[i * N + j], 0.0, 1e-10)
            << "Upper triangular element L[" << i << "][" << j << "] should be 0.0";
        }
      }

      // 下三角部分（対角含む）が非ゼロであることを確認（少なくともいくつか）
      bool has_nonzero = false;
      for (size_t i = 0; i < N; ++i)
      {
        if (std::abs(L[i * N + i]) > 1e-10)
        {
          has_nonzero = true;
          break;
        }
      }
      EXPECT_TRUE(has_nonzero) << "At least one diagonal element should be non-zero";
    }

  }  // namespace test
}  // namespace linalg
