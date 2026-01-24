#pragma once

#include <concepts>
#include <vector>

namespace stats
{

  /**
   * @brief ユークリッド距離の二乗を計算
   *
   * 2つのベクトル間のユークリッド距離の二乗を計算します。
   * ||x_i - x_j||² = Σ_d (x_id - x_jd)²
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   * @param x_i 第1のベクトル
   * @param x_j 第2のベクトル
   * @return ユークリッド距離の二乗 ||x_i - x_j||²
   * @throws std::invalid_argument x_i と x_j のサイズが異なる場合
   */
  template <std::floating_point T>
  T squared_euclidean_distance(const std::vector<T>& x_i, const std::vector<T>& x_j)
  {
    if (x_i.size() != x_j.size())
    {
      throw std::invalid_argument("x_i and x_j must have the same size");
    }

    T squared_distance = T(0);
    for (size_t d = 0; d < x_i.size(); ++d)
    {
      const T diff = x_i[d] - x_j[d];
      squared_distance += diff * diff;
    }
    return squared_distance;
  }

}  // namespace stats
