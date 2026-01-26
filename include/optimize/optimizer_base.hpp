#pragma once

#include <concepts>
#include <functional>
#include <vector>

namespace optimize
{

  /**
   * @brief 最適化結果
   *
   * @tparam T 浮動小数点型
   */
  template <std::floating_point T>
  struct OptimizationResult
  {
    public:
      /**
       * @brief 最適化されたパラメータ
       */
      std::vector<T> parameters;

      /**
       * @brief 目的関数の値
       */
      T objective_value;

      /**
       * @brief 反復回数
       */
      size_t iterations;

      /**
       * @brief 収束したかどうか
       */
      bool converged;
  };

  /**
   * @brief 最適化アルゴリズムの基底クラス
   *
   * 目的関数を最小化または最大化するための汎用的なインターフェースを提供します。
   *
   * @tparam T 浮動小数点型
   */
  template <std::floating_point T>
  class OptimizerBase
  {
    public:
      virtual ~OptimizerBase() = default;

      /**
       * @brief 目的関数を最小化
       *
       * @param objective 目的関数 f(x) -> T
       * @param initial_params 初期パラメータ
       * @param bounds パラメータの境界（オプション、空の場合は制約なし）
       * @return 最適化結果
       */
      virtual OptimizationResult<T> minimize(
        const std::function<T(const std::vector<T>&)>& objective,
        const std::vector<T>& initial_params,
        const std::vector<std::pair<T, T>>& bounds = {}) = 0;
  };

}  // namespace optimize
