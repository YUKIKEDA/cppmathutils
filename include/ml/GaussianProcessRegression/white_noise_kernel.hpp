#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ml/GaussianProcessRegression/kernel_base.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ml
{

  /**
   * @brief ホワイトノイズカーネル（White Noise Kernel）
   *
   * ホワイトノイズカーネルは、独立なノイズをモデル化するためのカーネルです。
   * 観測ノイズや測定誤差を表現するために使用されます。
   *
   * カーネル関数: k(x_i, x_j) = σ² * δ_ij
   *
   * ここで：
   * - σ²: ノイズ分散（noise variance）
   * - δ_ij: クロネッカーのデルタ（Kronecker delta）
   *   - δ_ij = 1 if i = j
   *   - δ_ij = 0 if i ≠ j
   *
   * 注意: ホワイトノイズカーネルは、入力ベクトルの値に依存せず、同じ入力点（i = j）か
   * 異なる入力点（i ≠ j）かのみに依存します。通常、単独で使用されることはなく、
   * 他のカーネル（RBFカーネル、Matérnカーネルなど）と加法的に組み合わせて使用されます。
   *
   * @see <a
   * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#white-noise-kernel">White
   * Noise Kernel Documentation</a>
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class WhiteNoiseKernel: public KernelBase<T>
  {
    public:
      static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (float, double, or long double)");

      /**
       * @brief コンストラクタ
       *
       * @param noise_variance ノイズ分散 σ²（デフォルト: 0.1）
       */
      explicit WhiteNoiseKernel(T noise_variance = T(0.1)):
          noise_variance_(noise_variance)
      {
        if (noise_variance_ <= T(0))
        {
          throw std::invalid_argument("noise_variance must be positive");
        }
      }

      /**
       * @brief カーネル関数の評価
       *
       * 注意: このメソッドは常に0を返します。ノイズ分散は`GaussianProcessRegression`が
       * 対角要素に直接追加するため、このメソッドは使用されません。
       *
       * @param x_i 第1の入力ベクトル
       * @param x_j 第2の入力ベクトル
       * @return 常に0
       */
      T operator()(const std::vector<T>& /* x_i */, const std::vector<T>& /* x_j */) const override
      {
        return T(0);
      }

      /**
       * @brief ノイズ分散を取得
       *
       * @return ノイズ分散 σ²
       */
      T get_noise_variance() const
      {
        return noise_variance_;
      }

      /**
       * @brief ノイズ分散を設定
       *
       * @param noise_variance ノイズ分散 σ²（noise_variance > 0）
       */
      void set_noise_variance(T noise_variance)
      {
        if (noise_variance <= T(0))
        {
          throw std::invalid_argument("noise_variance must be positive");
        }
        noise_variance_ = noise_variance;
      }

      /**
       * @brief WhiteNoiseKernelの場合、ノイズ分散を返す
       *
       * @return ノイズ分散
       */
      T get_noise_variance_if_white_noise() const override
      {
        return noise_variance_;
      }

      /**
       * @brief WhiteNoiseKernelかどうかを判定
       *
       * @return true
       */
      bool is_white_noise_kernel() const override
      {
        return true;
      }

      /**
       * @brief カーネルのクローンを作成
       *
       * @return カーネルのコピーへのshared_ptr
       */
      std::shared_ptr<KernelBase<T>> clone() const override
      {
        return std::make_shared<WhiteNoiseKernel<T>>(*this);
      }

    private:
      /**
       * @brief ノイズ分散（noise variance）σ²
       */
      T noise_variance_;
  };

}  // namespace ml
