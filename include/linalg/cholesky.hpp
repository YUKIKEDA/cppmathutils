#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace linalg
{

  /**
   * @brief Eigen + MKLバックエンドを使用してコレスキー分解を実行する関数
   *
   * 対称正定値行列 A を A = L * L^T の形に分解します。
   * ここで L は下三角行列です。
   *
   * @tparam T float または double
   * @param A       入力行列データ (行優先順序、サイズ N*N)
   * @param L       出力用バッファ (下三角行列、行優先順序、サイズ N*N、事前に確保が必要)
   * @param N       行列のサイズ (N×N)
   * @return bool   分解が成功した場合 true、失敗した場合 false
   */
  template <typename T>
  bool cholesky_decomposition(const T* A, T* L, size_t N)
  {
    if (A == nullptr || L == nullptr)
    {
      throw std::invalid_argument("Input pointers cannot be null");
    }

    if (N == 0)
    {
      throw std::invalid_argument("Matrix size N must be greater than 0");
    }

    // Eigen::MatrixXd または Eigen::MatrixXf に変換
    using MatrixType =
      std::conditional_t<std::is_same_v<T, double>, Eigen::MatrixXd, Eigen::MatrixXf>;
    using LLTType = std::conditional_t<std::is_same_v<T, double>,
      Eigen::LLT<Eigen::MatrixXd>,
      Eigen::LLT<Eigen::MatrixXf>>;

    // 入力行列をEigen形式に変換（行優先から列優先へ）
    MatrixType A_eigen(N, N);
    for (size_t i = 0; i < N; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        A_eigen(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = A[i * N + j];
      }
    }

    // コレスキー分解を実行
    LLTType llt(A_eigen);

    // 分解が成功したかチェック
    if (llt.info() != Eigen::Success)
    {
      return false;
    }

    // 下三角行列 L を取得
    MatrixType L_eigen = llt.matrixL();

    // 結果を行優先順序で出力バッファにコピー
    for (size_t i = 0; i < N; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        if (j <= i)
        {
          // 下三角部分（対角含む）
          L[i * N + j] = L_eigen(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
        else
        {
          // 上三角部分は0
          L[i * N + j] = static_cast<T>(0);
        }
      }
    }

    return true;
  }

  /**
   * @brief std::vector を受け取るオーバーロード（使いやすくするためのラッパー）
   */
  template <typename T>
  bool cholesky_decomposition(const std::vector<T>& A, std::vector<T>& L, size_t N)
  {
    if (A.size() != N * N)
    {
      throw std::invalid_argument("Input matrix A size must be N*N");
    }
    if (L.size() != N * N)
    {
      throw std::invalid_argument("Output matrix L size must be N*N");
    }

    return cholesky_decomposition(A.data(), L.data(), N);
  }

  /**
   * @brief コレスキー分解の結果を検証する関数
   *        A ≈ L * L^T であることを確認します
   *
   * @tparam T float または double
   * @param A       元の行列データ (行優先順序、サイズ N*N)
   * @param L       コレスキー分解の結果 (下三角行列、行優先順序、サイズ N*N)
   * @param N       行列のサイズ
   * @param tolerance 許容誤差
   * @return bool   検証が成功した場合 true
   */
  template <typename T>
  bool verify_cholesky_decomposition(
    const T* A, const T* L, size_t N, T tolerance = static_cast<T>(1e-5))
  {
    if (A == nullptr || L == nullptr)
    {
      throw std::invalid_argument("Input pointers cannot be null");
    }

    using MatrixType =
      std::conditional_t<std::is_same_v<T, double>, Eigen::MatrixXd, Eigen::MatrixXf>;

    // L をEigen形式に変換
    MatrixType L_eigen(N, N);
    for (size_t i = 0; i < N; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        L_eigen(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = L[i * N + j];
      }
    }

    // L * L^T を計算
    MatrixType reconstructed = L_eigen * L_eigen.transpose();

    // A と比較
    for (size_t i = 0; i < N; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        T diff = std::abs(
          reconstructed(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) - A[i * N + j]);
        if (diff > tolerance)
        {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * @brief std::vector を受け取るオーバーロード
   */
  template <typename T>
  bool verify_cholesky_decomposition(
    const std::vector<T>& A, const std::vector<T>& L, size_t N, T tolerance = static_cast<T>(1e-5))
  {
    return verify_cholesky_decomposition(A.data(), L.data(), N, tolerance);
  }

}  // namespace linalg
