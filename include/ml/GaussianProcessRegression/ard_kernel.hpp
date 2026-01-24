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
   * @brief ARDカーネル（Automatic Relevance Determination Kernel）
   *
   * ARDカーネルは、RBFカーネルを拡張し、各入力次元に異なる長さスケールを持つカーネルです。
   * これにより、各次元の関連性（relevance）を自動的に決定できます。
   *
   * カーネル関数: k(x_i, x_j) = σ_f² * exp(-0.5 * Σ_d (x_id - x_jd)² / ℓ_d²)
   *
   * ここで：
   * - σ_f²: 信号分散（signal variance）
   * - ℓ_d: d番目の次元の長さスケール（length scale for dimension d）
   * - x_id: i番目の入力点のd番目の次元
   *
   * @see <a
   * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#ard-kernel-automatic-relevance-determination-kernel">ARD
   * Kernel Documentation</a>
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class ARDKernel: public KernelBase<T>
  {
    public:
      static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (float, double, or long double)");

      /**
       * @brief コンストラクタ
       *
       * すべての次元で同じ長さスケールを使用して初期化します。
       *
       * @param sigma_f 信号分散の平方根（signal standard deviation、デフォルト: 1.0）
       * @param length_scales 各次元の長さスケールのベクトル（デフォルト: すべて1.0）
       *                       サイズが0の場合、入力次元に応じて自動的に設定されます
       */
      explicit ARDKernel(
        T sigma_f = T(1.0), const std::vector<T>& length_scales = std::vector<T>()):
          sigma_f_squared_(sigma_f * sigma_f),
          length_scales_(length_scales)
      {
        if (sigma_f <= T(0))
        {
          throw std::invalid_argument("sigma_f must be positive");
        }
        if (!length_scales_.empty())
        {
          for (size_t d = 0; d < length_scales_.size(); ++d)
          {
            if (length_scales_[d] <= T(0))
            {
              throw std::invalid_argument("All length_scales must be positive");
            }
          }
        }
      }

      /**
       * @brief カーネル関数の評価
       *
       * @param x_i 第1の入力ベクトル
       * @param x_j 第2の入力ベクトル
       * @return カーネル関数の値 k(x_i, x_j)
       * @throws std::invalid_argument x_i と x_j
       * のサイズが異なる場合、または長さスケールのサイズが入力次元と一致しない場合
       */
      T operator()(const std::vector<T>& x_i, const std::vector<T>& x_j) const
      {
        if (x_i.size() != x_j.size())
        {
          throw std::invalid_argument("x_i and x_j must have the same size");
        }

        const size_t dim = x_i.size();
        if (dim == 0)
        {
          return sigma_f_squared_;
        }

        // 長さスケールが設定されていない場合、すべて1.0として扱う
        if (length_scales_.empty())
        {
          // すべての次元で同じ長さスケール1.0を使用（RBFカーネルと等価）
          const T squared_distance = stats::squared_euclidean_distance(x_i, x_j);
          return sigma_f_squared_ * std::exp(-squared_distance / T(2));
        }

        if (length_scales_.size() != dim)
        {
          throw std::invalid_argument(
            "Size of length_scales must match input dimension or be empty");
        }

        // ARDカーネル: k(x_i, x_j) = σ_f² * exp(-0.5 * Σ_d (x_id - x_jd)² / ℓ_d²)
        T sum_squared_scaled_distance = T(0);
        for (size_t d = 0; d < dim; ++d)
        {
          const T diff = x_i[d] - x_j[d];
          const T scaled_diff = diff / length_scales_[d];
          sum_squared_scaled_distance += scaled_diff * scaled_diff;
        }

        return sigma_f_squared_ * std::exp(-T(0.5) * sum_squared_scaled_distance);
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
       * @brief 長さスケールのベクトルを取得
       *
       * @return 各次元の長さスケールのベクトル
       */
      const std::vector<T>& get_length_scales() const
      {
        return length_scales_;
      }

      /**
       * @brief 指定された次元の長さスケールを取得
       *
       * @param dim 次元のインデックス（0から始まる）
       * @return 指定された次元の長さスケール（設定されていない場合は1.0を返す）
       * @throws std::out_of_range dim が範囲外の場合
       */
      T get_length_scale(size_t dim) const
      {
        if (length_scales_.empty())
        {
          return T(1.0);
        }
        if (dim >= length_scales_.size())
        {
          throw std::out_of_range("Dimension index out of range");
        }
        return length_scales_[dim];
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
       * @brief 長さスケールのベクトルを設定
       *
       * @param length_scales 各次元の長さスケールのベクトル（すべての要素が正の値である必要がある）
       */
      void set_length_scales(const std::vector<T>& length_scales)
      {
        for (size_t d = 0; d < length_scales.size(); ++d)
        {
          if (length_scales[d] <= T(0))
          {
            throw std::invalid_argument("All length_scales must be positive");
          }
        }
        length_scales_ = length_scales;
      }

      /**
       * @brief 指定された次元の長さスケールを設定
       *
       * @param dim 次元のインデックス（0から始まる）
       * @param length_scale 長さスケール（length_scale > 0）
       * @throws std::invalid_argument length_scale が正でない場合
       */
      void set_length_scale(size_t dim, T length_scale)
      {
        if (length_scale <= T(0))
        {
          throw std::invalid_argument("length_scale must be positive");
        }
        if (length_scales_.size() <= dim)
        {
          length_scales_.resize(dim + 1, T(1.0));
        }
        length_scales_[dim] = length_scale;
      }

      /**
       * @brief カーネルのクローンを作成
       *
       * @return カーネルのコピーへのshared_ptr
       */
      std::shared_ptr<KernelBase<T>> clone() const override
      {
        return std::make_shared<ARDKernel<T>>(*this);
      }

    private:
      /**
       * @brief 信号分散（signal variance）σ_f²
       */
      T sigma_f_squared_;

      /**
       * @brief 各次元の長さスケール（length scales）ℓ_d
       *        空の場合は、すべての次元で1.0を使用（RBFカーネルと等価）
       */
      std::vector<T> length_scales_;
  };

}  // namespace ml
