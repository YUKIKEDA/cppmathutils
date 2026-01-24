#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ml/GaussianProcessRegression/kernel_base.hpp>
#include <stats/distance.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ml
{

  /**
   * @brief RBFカーネル（Radial Basis Function Kernel / Gaussian Kernel）
   *
   * RBFカーネルは、無限次元の特徴空間に対応するカーネル関数です。
   * 長さスケール ℓ は、入力空間における「類似性」の範囲を制御します。
   *
   * カーネル関数: k(x_i, x_j) = σ_f² * exp(-||x_i - x_j||² / (2ℓ²))
   *
   * @see <a
   * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#rbf-kernel-gaussian-kernel">RBF
   * Kernel Documentation</a>
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class RBFKernel: public KernelBase<T>
  {
    public:
      static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (float, double, or long double)");

      /**
       * @brief コンストラクタ
       *
       * @param sigma_f 信号分散の平方根（signal standard deviation、デフォルト: 1.0）
       * @param length_scale 長さスケール（length scale、デフォルト: 1.0）
       */
      explicit RBFKernel(T sigma_f = T(1.0), T length_scale = T(1.0)):
          sigma_f_squared_(sigma_f * sigma_f),
          length_scale_(length_scale)
      {
        if (sigma_f <= T(0))
        {
          throw std::invalid_argument("sigma_f must be positive");
        }
        if (length_scale <= T(0))
        {
          throw std::invalid_argument("length_scale must be positive");
        }
      }

      /**
       * @brief カーネル関数の評価
       *
       * @param x_i 第1の入力ベクトル
       * @param x_j 第2の入力ベクトル
       * @return カーネル関数の値 k(x_i, x_j)
       * @throws std::invalid_argument x_i と x_j のサイズが異なる場合
       */
      T operator()(const std::vector<T>& x_i, const std::vector<T>& x_j) const
      {
        // ユークリッド距離の二乗を計算
        const T squared_distance = stats::squared_euclidean_distance(x_i, x_j);

        // RBFカーネル: k(x_i, x_j) = σ_f² * exp(-||x_i - x_j||² / (2ℓ²))
        return sigma_f_squared_
          * std::exp(-squared_distance / (T(2) * length_scale_ * length_scale_));
      }

      /**
       * @brief 信号分散の平方根を取得
       *
       * @return 信号分散の平方根 σ_f
       */
      T get_sigma_f() const
      {
        return std::sqrt(sigma_f_squared_);
      }

      /**
       * @brief 長さスケールを取得
       *
       * @return 長さスケール ℓ
       */
      T get_length_scale() const
      {
        return length_scale_;
      }

      /**
       * @brief 信号分散の平方根を設定
       *
       * @param sigma_f 信号分散の平方根（sigma_f > 0）
       */
      void set_sigma_f(T sigma_f)
      {
        if (sigma_f <= T(0))
        {
          throw std::invalid_argument("sigma_f must be positive");
        }
        sigma_f_squared_ = sigma_f * sigma_f;
      }

      /**
       * @brief 長さスケールを設定
       *
       * @param length_scale 長さスケール（length_scale > 0）
       */
      void set_length_scale(T length_scale)
      {
        if (length_scale <= T(0))
        {
          throw std::invalid_argument("length_scale must be positive");
        }
        length_scale_ = length_scale;
      }

      /**
       * @brief カーネルのクローンを作成
       *
       * @return カーネルのコピーへのshared_ptr
       */
      std::shared_ptr<KernelBase<T>> clone() const override
      {
        return std::make_shared<RBFKernel<T>>(*this);
      }

    private:
      /**
       * @brief 信号分散（signal variance）σ_f²
       */
      T sigma_f_squared_;

      /**
       * @brief 長さスケール（length scale）ℓ
       */
      T length_scale_;
  };

}  // namespace ml
