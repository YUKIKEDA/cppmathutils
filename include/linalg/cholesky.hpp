#pragma once

#include <Eigen/Dense>
#include <concepts>
#include <stdexcept>
#include <type_traits>

namespace linalg
{

  /**
   * @brief コレスキー分解（Cholesky Decomposition）を用いて連立方程式 Ax = b を解く
   *
   * 正定値行列 A に対して、コレスキー分解 A = LL^T を用いて連立方程式 Ax = b を解きます。
   * コレスキー分解は数値的に安定で、正定値行列に対して効率的な解法を提供します。
   *
   * 計算量: O(n³)（n は行列のサイズ）
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   * @param A 正定値行列（n × n）
   * @param b 右辺ベクトル（n × 1）
   * @return 解ベクトル x（n × 1）
   * @throws std::invalid_argument A と b のサイズが一致しない場合
   * @throws std::runtime_error コレスキー分解が失敗した場合（行列が正定値でない可能性）
   *
   * @example
   * ```cpp
   * Eigen::MatrixXd A(3, 3);
   * A << 4, 2, 1,
   *      2, 5, 2,
   *      1, 2, 6;
   * Eigen::VectorXd b(3);
   * b << 1, 2, 3;
   * Eigen::VectorXd x = linalg::cholesky_solve(A, b);
   * ```
   */
  template <std::floating_point T>
  Eigen::Matrix<T, Eigen::Dynamic, 1> cholesky_solve(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& b)
  {
    static_assert(std::is_floating_point_v<T>,
      "T must be a floating-point type (float, double, or long double)");

    if (A.rows() != A.cols())
    {
      throw std::invalid_argument("Matrix A must be square");
    }
    if (A.rows() != b.rows())
    {
      throw std::invalid_argument("Matrix A and vector b must have compatible sizes");
    }

    // コレスキー分解: A = LL^T
    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> llt(A);
    if (llt.info() != Eigen::Success)
    {
      throw std::runtime_error(
        "Cholesky decomposition failed. Matrix may not be positive definite.");
    }

    // 連立方程式 Ax = b を解く
    return llt.solve(b);
  }

  /**
   * @brief コレスキー分解（Cholesky Decomposition）を用いて連立方程式 AX = B を解く
   *
   * 正定値行列 A に対して、コレスキー分解 A = LL^T を用いて連立方程式 AX = B を解きます。
   * 複数の右辺ベクトルに対して同時に解を求める場合に使用します。
   *
   * 計算量: O(n³ + n²m)（n は行列のサイズ、m は右辺ベクトルの数）
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   * @param A 正定値行列（n × n）
   * @param B 右辺行列（n × m、各列が1つの右辺ベクトル）
   * @return 解行列 X（n × m、各列が1つの解ベクトル）
   * @throws std::invalid_argument A と B のサイズが一致しない場合
   * @throws std::runtime_error コレスキー分解が失敗した場合（行列が正定値でない可能性）
   *
   * @example
   * ```cpp
   * Eigen::MatrixXd A(3, 3);
   * A << 4, 2, 1,
   *      2, 5, 2,
   *      1, 2, 6;
   * Eigen::MatrixXd B(3, 2);
   * B << 1, 2,
   *      2, 3,
   *      3, 4;
   * Eigen::MatrixXd X = linalg::cholesky_solve(A, B);
   * ```
   */
  template <std::floating_point T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> cholesky_solve(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B)
  {
    static_assert(std::is_floating_point_v<T>,
      "T must be a floating-point type (float, double, or long double)");

    if (A.rows() != A.cols())
    {
      throw std::invalid_argument("Matrix A must be square");
    }
    if (A.rows() != B.rows())
    {
      throw std::invalid_argument("Matrix A and matrix B must have compatible sizes");
    }

    // コレスキー分解: A = LL^T
    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> llt(A);
    if (llt.info() != Eigen::Success)
    {
      throw std::runtime_error(
        "Cholesky decomposition failed. Matrix may not be positive definite.");
    }

    // 連立方程式 AX = B を解く
    return llt.solve(B);
  }

  /**
   * @brief コレスキー分解（Cholesky Decomposition）を行い、下三角行列 L を返す
   *
   * 正定値行列 A に対して、コレスキー分解 A = LL^T を行い、下三角行列 L を返します。
   *
   * 計算量: O(n³)（n は行列のサイズ）
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   * @param A 正定値行列（n × n）
   * @return 下三角行列 L（n × n、A = LL^T）
   * @throws std::invalid_argument A が正方行列でない場合
   * @throws std::runtime_error コレスキー分解が失敗した場合（行列が正定値でない可能性）
   *
   * @example
   * ```cpp
   * Eigen::MatrixXd A(3, 3);
   * A << 4, 2, 1,
   *      2, 5, 2,
   *      1, 2, 6;
   * Eigen::MatrixXd L = linalg::cholesky_decompose(A);
   * ```
   */
  template <std::floating_point T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> cholesky_decompose(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A)
  {
    static_assert(std::is_floating_point_v<T>,
      "T must be a floating-point type (float, double, or long double)");

    if (A.rows() != A.cols())
    {
      throw std::invalid_argument("Matrix A must be square");
    }

    // コレスキー分解: A = LL^T
    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> llt(A);
    if (llt.info() != Eigen::Success)
    {
      throw std::runtime_error(
        "Cholesky decomposition failed. Matrix may not be positive definite.");
    }

    // 下三角行列 L を返す
    return llt.matrixL();
  }

}  // namespace linalg
