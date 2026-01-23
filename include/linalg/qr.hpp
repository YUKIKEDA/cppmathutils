#pragma once

#include <Eigen/Dense>
#include <concepts>
#include <stdexcept>
#include <type_traits>

namespace linalg
{

  /**
   * @brief QR分解（QR Decomposition）を用いて連立方程式 Ax = b を解く
   *
   * 任意の行列 A に対して、QR分解 A = QR（Q は直交行列、R は上三角行列）を用いて
   * 連立方程式 Ax = b を解きます。
   * QR分解は数値的に安定で、特に最小二乗問題を解く際に有効です。
   *
   * 計算量: O(n²m)（n は行数、m は列数）
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   * @param A 係数行列（n × m）
   * @param b 右辺ベクトル（n × 1）
   * @return 解ベクトル x（m × 1）
   * @throws std::invalid_argument A と b のサイズが一致しない場合
   *
   * @example
   * ```cpp
   * Eigen::MatrixXd A(3, 2);
   * A << 1, 2,
   *      3, 4,
   *      5, 6;
   * Eigen::VectorXd b(3);
   * b << 1, 2, 3;
   * Eigen::VectorXd x = linalg::qr_solve(A, b);
   * ```
   */
  template <std::floating_point T>
  Eigen::Matrix<T, Eigen::Dynamic, 1> qr_solve(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& b)
  {
    static_assert(std::is_floating_point_v<T>,
      "T must be a floating-point type (float, double, or long double)");

    if (A.rows() != b.rows())
    {
      throw std::invalid_argument("Matrix A and vector b must have compatible sizes");
    }

    // QR分解: A = QR（Householder変換を使用）
    Eigen::HouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> qr(A);

    // 連立方程式 Ax = b を解く
    return qr.solve(b);
  }

  /**
   * @brief QR分解（QR Decomposition）を用いて連立方程式 AX = B を解く
   *
   * 任意の行列 A に対して、QR分解 A = QR（Q は直交行列、R は上三角行列）を用いて
   * 連立方程式 AX = B を解きます。
   * 複数の右辺ベクトルに対して同時に解を求める場合に使用します。
   *
   * 計算量: O(n²m + nmk)（n は行数、m は列数、k は右辺ベクトルの数）
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   * @param A 係数行列（n × m）
   * @param B 右辺行列（n × k、各列が1つの右辺ベクトル）
   * @return 解行列 X（m × k、各列が1つの解ベクトル）
   * @throws std::invalid_argument A と B のサイズが一致しない場合
   *
   * @example
   * ```cpp
   * Eigen::MatrixXd A(3, 2);
   * A << 1, 2,
   *      3, 4,
   *      5, 6;
   * Eigen::MatrixXd B(3, 2);
   * B << 1, 2,
   *      2, 3,
   *      3, 4;
   * Eigen::MatrixXd X = linalg::qr_solve(A, B);
   * ```
   */
  template <std::floating_point T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> qr_solve(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B)
  {
    static_assert(std::is_floating_point_v<T>,
      "T must be a floating-point type (float, double, or long double)");

    if (A.rows() != B.rows())
    {
      throw std::invalid_argument("Matrix A and matrix B must have compatible sizes");
    }

    // QR分解: A = QR（Householder変換を使用）
    Eigen::HouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> qr(A);

    // 連立方程式 AX = B を解く
    return qr.solve(B);
  }

}  // namespace linalg
