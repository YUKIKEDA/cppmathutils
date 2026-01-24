#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <ml/GaussianProcessRegression/ard_kernel.hpp>
#include <ml/GaussianProcessRegression/kernel_base.hpp>
#include <ml/GaussianProcessRegression/rbf_kernel.hpp>
#include <ml/GaussianProcessRegression/white_noise_kernel.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ml
{

  // 前方宣言
  template <std::floating_point T>
  class SumKernel;

  template <std::floating_point T>
  class ProductKernel;

  /**
   * @brief SumKernelに含まれるWhiteNoiseKernelのノイズ分散の合計を取得
   *
   * 型情報を利用して効率的にWhiteNoiseKernelを検出します。
   *
   * @param sum_kernel SumKernel
   * @return WhiteNoiseKernelが含まれている場合はそのノイズ分散の合計、含まれていない場合は0
   */
  template <std::floating_point T>
  T extract_white_noise_variance_from_sum_kernel(const SumKernel<T>& sum_kernel)
  {
    T total_noise = T(0);
    const auto& kernels = sum_kernel.get_kernels();

    for (const auto& kernel : kernels)
    {
      if (kernel && kernel->is_white_noise_kernel())
      {
        total_noise += kernel->get_noise_variance_if_white_noise();
      }
    }
    return total_noise;
  }

  /**
   * @brief カーネルの加算（Sum Kernel）
   *
   * 複数のカーネルを加算することで、異なる成分をモデル化できます。
   * カーネル関数: k_total(x_i, x_j) = k_1(x_i, x_j) + k_2(x_i, x_j) + ... + k_m(x_i, x_j)
   *
   * @see <a
   * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#composite-kernels">Composite
   * Kernels Documentation</a>
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class SumKernel: public KernelBase<T>
  {
    public:
      static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (float, double, or long double)");

      /**
       * @brief コンストラクタ
       *
       * @param kernel1 第1のカーネル関数
       * @param kernel2 第2のカーネル関数
       */
      SumKernel(std::shared_ptr<KernelBase<T>> kernel1, std::shared_ptr<KernelBase<T>> kernel2):
          kernels_({kernel1, kernel2})
      {
        if (!kernel1 || !kernel2)
        {
          throw std::invalid_argument("Kernels must not be null");
        }
      }

      /**
       * @brief コンストラクタ（複数のカーネル）
       *
       * @param kernels カーネル関数のベクトル（少なくとも2つ必要）
       */
      explicit SumKernel(const std::vector<std::shared_ptr<KernelBase<T>>>& kernels):
          kernels_(kernels)
      {
        if (kernels.size() < 2)
        {
          throw std::invalid_argument("At least two kernels are required");
        }
        for (const auto& kernel : kernels_)
        {
          if (!kernel)
          {
            throw std::invalid_argument("All kernels must not be null");
          }
        }
      }

      /**
       * @brief カーネル関数の評価
       *
       * @param x_i 第1の入力ベクトル
       * @param x_j 第2の入力ベクトル
       * @return カーネル関数の値 k_total(x_i, x_j) = Σ_k k_k(x_i, x_j)
       */
      T operator()(const std::vector<T>& x_i, const std::vector<T>& x_j) const override
      {
        T sum = T(0);
        for (const auto& kernel : kernels_)
        {
          sum += (*kernel)(x_i, x_j);
        }
        return sum;
      }

      /**
       * @brief 含まれるカーネルの数を取得
       *
       * @return カーネルの数
       */
      size_t get_n_kernels() const
      {
        return kernels_.size();
      }

      /**
       * @brief カーネルを追加
       *
       * @param kernel 追加するカーネル関数
       */
      void add_kernel(std::shared_ptr<KernelBase<T>> kernel)
      {
        if (!kernel)
        {
          throw std::invalid_argument("Kernel must not be null");
        }
        kernels_.push_back(kernel);
      }

      /**
       * @brief 内部のカーネルベクトルを取得（SumKernel同士の合成用）
       *
       * @return カーネル関数のベクトルへの参照
       */
      const std::vector<std::shared_ptr<KernelBase<T>>>& get_kernels() const
      {
        return kernels_;
      }

      /**
       * @brief カーネルのクローンを作成
       *
       * @return カーネルのコピーへのshared_ptr
       */
      std::shared_ptr<KernelBase<T>> clone() const override
      {
        std::vector<std::shared_ptr<KernelBase<T>>> cloned_kernels;
        cloned_kernels.reserve(kernels_.size());
        for (const auto& kernel : kernels_)
        {
          cloned_kernels.push_back(kernel->clone());
        }
        return std::make_shared<SumKernel<T>>(cloned_kernels);
      }

    private:
      /**
       * @brief カーネル関数のベクトル
       */
      std::vector<std::shared_ptr<KernelBase<T>>> kernels_;
  };

  /**
   * @brief カーネルの乗算（Product Kernel）
   *
   * 複数のカーネルを乗算することで、相互作用をモデル化できます。
   * カーネル関数: k_total(x_i, x_j) = k_1(x_i, x_j) * k_2(x_i, x_j) * ... * k_m(x_i, x_j)
   *
   * @see <a
   * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#composite-kernels">Composite
   * Kernels Documentation</a>
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class ProductKernel: public KernelBase<T>
  {
    public:
      static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (float, double, or long double)");

      /**
       * @brief コンストラクタ
       *
       * @param kernel1 第1のカーネル関数
       * @param kernel2 第2のカーネル関数
       */
      ProductKernel(std::shared_ptr<KernelBase<T>> kernel1, std::shared_ptr<KernelBase<T>> kernel2):
          kernels_({kernel1, kernel2})
      {
        if (!kernel1 || !kernel2)
        {
          throw std::invalid_argument("Kernels must not be null");
        }
      }

      /**
       * @brief コンストラクタ（複数のカーネル）
       *
       * @param kernels カーネル関数のベクトル（少なくとも2つ必要）
       */
      explicit ProductKernel(const std::vector<std::shared_ptr<KernelBase<T>>>& kernels):
          kernels_(kernels)
      {
        if (kernels.size() < 2)
        {
          throw std::invalid_argument("At least two kernels are required");
        }
        for (const auto& kernel : kernels_)
        {
          if (!kernel)
          {
            throw std::invalid_argument("All kernels must not be null");
          }
        }
      }

      /**
       * @brief カーネル関数の評価
       *
       * @param x_i 第1の入力ベクトル
       * @param x_j 第2の入力ベクトル
       * @return カーネル関数の値 k_total(x_i, x_j) = Π_k k_k(x_i, x_j)
       */
      T operator()(const std::vector<T>& x_i, const std::vector<T>& x_j) const override
      {
        T product = T(1);
        for (const auto& kernel : kernels_)
        {
          product *= (*kernel)(x_i, x_j);
        }
        return product;
      }

      /**
       * @brief 含まれるカーネルの数を取得
       *
       * @return カーネルの数
       */
      size_t get_n_kernels() const
      {
        return kernels_.size();
      }

      /**
       * @brief カーネルを追加
       *
       * @param kernel 追加するカーネル関数
       */
      void add_kernel(std::shared_ptr<KernelBase<T>> kernel)
      {
        if (!kernel)
        {
          throw std::invalid_argument("Kernel must not be null");
        }
        kernels_.push_back(kernel);
      }

      /**
       * @brief 内部のカーネルベクトルを取得（ProductKernel同士の合成用）
       *
       * @return カーネル関数のベクトルへの参照
       */
      const std::vector<std::shared_ptr<KernelBase<T>>>& get_kernels() const
      {
        return kernels_;
      }

      /**
       * @brief カーネルのクローンを作成
       *
       * @return カーネルのコピーへのshared_ptr
       */
      std::shared_ptr<KernelBase<T>> clone() const override
      {
        std::vector<std::shared_ptr<KernelBase<T>>> cloned_kernels;
        cloned_kernels.reserve(kernels_.size());
        for (const auto& kernel : kernels_)
        {
          cloned_kernels.push_back(kernel->clone());
        }
        return std::make_shared<ProductKernel<T>>(cloned_kernels);
      }

    private:
      /**
       * @brief カーネル関数のベクトル
       */
      std::vector<std::shared_ptr<KernelBase<T>>> kernels_;
  };

  // ============================================================================
  // ヘルパー関数: カーネルをshared_ptrに変換
  // ============================================================================

  /**
   * @brief カーネルをshared_ptrに変換するヘルパー関数
   */
  template <std::floating_point T>
  std::shared_ptr<KernelBase<T>> to_kernel_ptr(std::shared_ptr<KernelBase<T>> kernel)
  {
    return kernel;
  }

  template <std::floating_point T>
  std::shared_ptr<KernelBase<T>> to_kernel_ptr(const RBFKernel<T>& kernel)
  {
    return std::make_shared<RBFKernel<T>>(kernel);
  }

  template <std::floating_point T>
  std::shared_ptr<KernelBase<T>> to_kernel_ptr(const ARDKernel<T>& kernel)
  {
    return std::make_shared<ARDKernel<T>>(kernel);
  }

  template <std::floating_point T>
  std::shared_ptr<KernelBase<T>> to_kernel_ptr(const WhiteNoiseKernel<T>& kernel)
  {
    return std::make_shared<WhiteNoiseKernel<T>>(kernel);
  }

  template <std::floating_point T>
  std::shared_ptr<KernelBase<T>> to_kernel_ptr(const SumKernel<T>& kernel)
  {
    return std::make_shared<SumKernel<T>>(kernel);
  }

  template <std::floating_point T>
  std::shared_ptr<KernelBase<T>> to_kernel_ptr(const ProductKernel<T>& kernel)
  {
    return std::make_shared<ProductKernel<T>>(kernel);
  }

  // ============================================================================
  // 加算演算子（テンプレート化して簡素化）
  // ============================================================================

  /**
   * @brief 2つのカーネルの加算演算子（汎用版）
   *
   * @param lhs 左辺のカーネル
   * @param rhs 右辺のカーネル
   * @return SumKernel（2つのカーネルの和）
   */
  template <std::floating_point T, typename Kernel1, typename Kernel2>
  SumKernel<T> operator+(const Kernel1& lhs, const Kernel2& rhs)
  {
    auto lhs_ptr = to_kernel_ptr<T>(lhs);
    auto rhs_ptr = to_kernel_ptr<T>(rhs);
    return SumKernel<T>(lhs_ptr, rhs_ptr);
  }

  /**
   * @brief SumKernelと他のカーネルの加算演算子
   *
   * @param sum_kernel SumKernel
   * @param other 他のカーネル
   * @return SumKernel（既存のSumKernelに新しいカーネルを追加）
   */
  template <std::floating_point T, typename Kernel>
  SumKernel<T> operator+(const SumKernel<T>& sum_kernel, const Kernel& other)
  {
    auto other_ptr = to_kernel_ptr<T>(other);
    std::vector<std::shared_ptr<KernelBase<T>>> kernels = sum_kernel.get_kernels();
    kernels.push_back(other_ptr);
    return SumKernel<T>(kernels);
  }

  /**
   * @brief 他のカーネルとSumKernelの加算演算子
   *
   * @param other 他のカーネル
   * @param sum_kernel SumKernel
   * @return SumKernel（既存のSumKernelに新しいカーネルを追加）
   */
  template <std::floating_point T, typename Kernel>
  SumKernel<T> operator+(const Kernel& other, const SumKernel<T>& sum_kernel)
  {
    auto other_ptr = to_kernel_ptr<T>(other);
    std::vector<std::shared_ptr<KernelBase<T>>> kernels;
    kernels.push_back(other_ptr);
    const auto& existing_kernels = sum_kernel.get_kernels();
    kernels.insert(kernels.end(), existing_kernels.begin(), existing_kernels.end());
    return SumKernel<T>(kernels);
  }

  /**
   * @brief SumKernel同士の加算演算子
   *
   * @param sum_kernel1 第1のSumKernel
   * @param sum_kernel2 第2のSumKernel
   * @return SumKernel（2つのSumKernelを結合）
   */
  template <std::floating_point T>
  SumKernel<T> operator+(const SumKernel<T>& sum_kernel1, const SumKernel<T>& sum_kernel2)
  {
    std::vector<std::shared_ptr<KernelBase<T>>> kernels;
    const auto& kernels1 = sum_kernel1.get_kernels();
    const auto& kernels2 = sum_kernel2.get_kernels();
    kernels.reserve(kernels1.size() + kernels2.size());
    kernels.insert(kernels.end(), kernels1.begin(), kernels1.end());
    kernels.insert(kernels.end(), kernels2.begin(), kernels2.end());
    return SumKernel<T>(kernels);
  }

  /**
   * @brief shared_ptr同士の加算演算子
   *
   * @param kernel_ptr1 第1のカーネルへのshared_ptr
   * @param kernel_ptr2 第2のカーネルへのshared_ptr
   * @return SumKernel
   */
  template <std::floating_point T>
  SumKernel<T> operator+(
    std::shared_ptr<KernelBase<T>> kernel_ptr1, std::shared_ptr<KernelBase<T>> kernel_ptr2)
  {
    if (!kernel_ptr1 || !kernel_ptr2)
    {
      throw std::invalid_argument("Kernel pointers must not be null");
    }
    return SumKernel<T>(kernel_ptr1, kernel_ptr2);
  }

  /**
   * @brief shared_ptrとカーネルの加算演算子
   *
   * @param kernel_ptr カーネルへのshared_ptr
   * @param other 他のカーネル
   * @return SumKernel
   */
  template <std::floating_point T, typename Kernel>
  SumKernel<T> operator+(std::shared_ptr<KernelBase<T>> kernel_ptr, const Kernel& other)
  {
    if (!kernel_ptr)
    {
      throw std::invalid_argument("Kernel pointer must not be null");
    }
    auto other_ptr = to_kernel_ptr<T>(other);
    return SumKernel<T>(kernel_ptr, other_ptr);
  }

  /**
   * @brief カーネルとshared_ptrの加算演算子
   *
   * @param other 他のカーネル
   * @param kernel_ptr カーネルへのshared_ptr
   * @return SumKernel
   */
  template <std::floating_point T, typename Kernel>
  SumKernel<T> operator+(const Kernel& other, std::shared_ptr<KernelBase<T>> kernel_ptr)
  {
    if (!kernel_ptr)
    {
      throw std::invalid_argument("Kernel pointer must not be null");
    }
    auto other_ptr = to_kernel_ptr<T>(other);
    return SumKernel<T>(other_ptr, kernel_ptr);
  }

  // ============================================================================
  // 乗算演算子（テンプレート化して簡素化）
  // ============================================================================

  /**
   * @brief 2つのカーネルの乗算演算子（汎用版）
   *
   * @param lhs 左辺のカーネル
   * @param rhs 右辺のカーネル
   * @return ProductKernel（2つのカーネルの積）
   */
  template <std::floating_point T, typename Kernel1, typename Kernel2>
  ProductKernel<T> operator*(const Kernel1& lhs, const Kernel2& rhs)
  {
    auto lhs_ptr = to_kernel_ptr<T>(lhs);
    auto rhs_ptr = to_kernel_ptr<T>(rhs);
    return ProductKernel<T>(lhs_ptr, rhs_ptr);
  }

  /**
   * @brief ProductKernelと他のカーネルの乗算演算子
   *
   * @param product_kernel ProductKernel
   * @param other 他のカーネル
   * @return ProductKernel（既存のProductKernelに新しいカーネルを追加）
   */
  template <std::floating_point T, typename Kernel>
  ProductKernel<T> operator*(const ProductKernel<T>& product_kernel, const Kernel& other)
  {
    auto other_ptr = to_kernel_ptr<T>(other);
    std::vector<std::shared_ptr<KernelBase<T>>> kernels = product_kernel.get_kernels();
    kernels.push_back(other_ptr);
    return ProductKernel<T>(kernels);
  }

  /**
   * @brief 他のカーネルとProductKernelの乗算演算子
   *
   * @param other 他のカーネル
   * @param product_kernel ProductKernel
   * @return ProductKernel（既存のProductKernelに新しいカーネルを追加）
   */
  template <std::floating_point T, typename Kernel>
  ProductKernel<T> operator*(const Kernel& other, const ProductKernel<T>& product_kernel)
  {
    auto other_ptr = to_kernel_ptr<T>(other);
    std::vector<std::shared_ptr<KernelBase<T>>> kernels;
    kernels.push_back(other_ptr);
    const auto& existing_kernels = product_kernel.get_kernels();
    kernels.insert(kernels.end(), existing_kernels.begin(), existing_kernels.end());
    return ProductKernel<T>(kernels);
  }

  /**
   * @brief ProductKernel同士の乗算演算子
   *
   * @param product_kernel1 第1のProductKernel
   * @param product_kernel2 第2のProductKernel
   * @return ProductKernel（2つのProductKernelを結合）
   */
  template <std::floating_point T>
  ProductKernel<T> operator*(
    const ProductKernel<T>& product_kernel1, const ProductKernel<T>& product_kernel2)
  {
    std::vector<std::shared_ptr<KernelBase<T>>> kernels;
    const auto& kernels1 = product_kernel1.get_kernels();
    const auto& kernels2 = product_kernel2.get_kernels();
    kernels.reserve(kernels1.size() + kernels2.size());
    kernels.insert(kernels.end(), kernels1.begin(), kernels1.end());
    kernels.insert(kernels.end(), kernels2.begin(), kernels2.end());
    return ProductKernel<T>(kernels);
  }

  /**
   * @brief shared_ptrとカーネルの乗算演算子
   *
   * @param kernel_ptr カーネルへのshared_ptr
   * @param other 他のカーネル
   * @return ProductKernel
   */
  template <std::floating_point T, typename Kernel>
  ProductKernel<T> operator*(std::shared_ptr<KernelBase<T>> kernel_ptr, const Kernel& other)
  {
    if (!kernel_ptr)
    {
      throw std::invalid_argument("Kernel pointer must not be null");
    }
    auto other_ptr = to_kernel_ptr<T>(other);
    return ProductKernel<T>(kernel_ptr, other_ptr);
  }

  /**
   * @brief カーネルとshared_ptrの乗算演算子
   *
   * @param other 他のカーネル
   * @param kernel_ptr カーネルへのshared_ptr
   * @return ProductKernel
   */
  template <std::floating_point T, typename Kernel>
  ProductKernel<T> operator*(const Kernel& other, std::shared_ptr<KernelBase<T>> kernel_ptr)
  {
    if (!kernel_ptr)
    {
      throw std::invalid_argument("Kernel pointer must not be null");
    }
    auto other_ptr = to_kernel_ptr<T>(other);
    return ProductKernel<T>(other_ptr, kernel_ptr);
  }

}  // namespace ml
