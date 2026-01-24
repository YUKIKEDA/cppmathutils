#pragma once

#include <concepts>
#include <memory>
#include <vector>

namespace ml
{

  /**
   * @brief カーネル関数の基底クラス
   *
   * 型情報を保持しながら多態性を実現するための基底クラス
   */
  template <std::floating_point T>
  class KernelBase
  {
    public:
      virtual ~KernelBase() = default;

      /**
       * @brief カーネル関数の評価
       *
       * @param x_i 第1の入力ベクトル
       * @param x_j 第2の入力ベクトル
       * @return カーネル関数の値 k(x_i, x_j)
       */
      virtual T operator()(const std::vector<T>& x_i, const std::vector<T>& x_j) const = 0;

      /**
       * @brief カーネルのクローンを作成
       *
       * @return カーネルのコピーへのshared_ptr
       */
      virtual std::shared_ptr<KernelBase<T>> clone() const = 0;

      /**
       * @brief WhiteNoiseKernelの場合、ノイズ分散を返す
       *
       * @return WhiteNoiseKernelの場合はノイズ分散、それ以外は0
       */
      virtual T get_noise_variance_if_white_noise() const
      {
        return T(0);
      }

      /**
       * @brief WhiteNoiseKernelかどうかを判定
       *
       * @return WhiteNoiseKernelの場合はtrue、それ以外はfalse
       */
      virtual bool is_white_noise_kernel() const
      {
        return false;
      }
  };

}  // namespace ml
