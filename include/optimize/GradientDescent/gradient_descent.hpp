#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <optimize/optimizer_base.hpp>
#include <stdexcept>
#include <vector>

namespace optimize
{

  /**
   * @brief 勾配降下法（Gradient Descent）最適化アルゴリズム
   *
   * 最も基本的な最適化アルゴリズムで、目的関数の勾配の反対方向にパラメータを更新します。
   * 数値微分を使用して勾配を計算します。
   *
   * @tparam T 浮動小数点型
   */
  template <std::floating_point T>
  class GradientDescentOptimizer: public OptimizerBase<T>
  {
    public:
      /**
       * @brief コンストラクタ
       *
       * @param learning_rate 学習率（デフォルト: 0.01）
       * @param max_iterations 最大反復回数（デフォルト: 1000）
       * @param tolerance 収束判定の許容誤差（デフォルト: 1e-6）
       * @param epsilon 数値微分の微小値（デフォルト: 1e-8）
       */
      explicit GradientDescentOptimizer(T learning_rate = T(0.01),
        size_t max_iterations = 1'000,
        T tolerance = T(1e-6),
        T epsilon = T(1e-8)):
          learning_rate_(learning_rate),
          max_iterations_(max_iterations),
          tolerance_(tolerance),
          epsilon_(epsilon)
      {
        if (learning_rate_ <= T(0))
        {
          throw std::invalid_argument("learning_rate must be positive");
        }
        if (tolerance_ < T(0))
        {
          throw std::invalid_argument("tolerance must be non-negative");
        }
        if (epsilon_ <= T(0))
        {
          throw std::invalid_argument("epsilon must be positive");
        }
      }

      /**
       * @brief 目的関数を最小化
       *
       * @param objective 目的関数 f(x) -> T
       * @param initial_params 初期パラメータ
       * @param bounds パラメータの境界（オプション、空の場合は制約なし）
       * @param gradient 勾配関数 ∇f(x) ->
       * std::vector<T>（オプション、nullptrの場合は数値微分を使用）
       * @return 最適化結果
       */
      OptimizationResult<T> minimize(const std::function<T(const std::vector<T>&)>& objective,
        const std::vector<T>& initial_params,
        const std::vector<std::pair<T, T>>& bounds = {},
        const std::function<std::vector<T>(const std::vector<T>&)>* gradient = nullptr) override
      {
        if (initial_params.empty())
        {
          throw std::invalid_argument("initial_params must not be empty");
        }

        OptimizationResult<T> result;
        result.parameters = initial_params;
        result.iterations = 0;
        result.converged = false;

        T prev_objective = objective(result.parameters);

        for (size_t iter = 0; iter < max_iterations_; ++iter)
        {
          // 勾配を計算（提供されていればそれを使用、なければ数値微分）
          std::vector<T> grad(result.parameters.size());
          if (gradient != nullptr)
          {
            grad = (*gradient)(result.parameters);
          }
          else
          {
            // 数値微分で勾配を計算（中央差分）
            for (size_t i = 0; i < result.parameters.size(); ++i)
            {
              std::vector<T> params_plus = result.parameters;
              params_plus[i] += epsilon_;
              T objective_plus = objective(params_plus);

              std::vector<T> params_minus = result.parameters;
              params_minus[i] -= epsilon_;
              T objective_minus = objective(params_minus);

              // 中央差分: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
              grad[i] = (objective_plus - objective_minus) / (T(2) * epsilon_);
            }
          }

          // パラメータを更新: x = x - learning_rate * gradient
          for (size_t i = 0; i < result.parameters.size(); ++i)
          {
            result.parameters[i] -= learning_rate_ * grad[i];
          }

          // 境界制約を適用
          if (!bounds.empty())
          {
            for (size_t i = 0; i < result.parameters.size() && i < bounds.size(); ++i)
            {
              result.parameters[i] =
                std::clamp(result.parameters[i], bounds[i].first, bounds[i].second);
            }
          }

          // 目的関数を再計算
          T current_objective = objective(result.parameters);

          // 収束判定
          result.iterations = iter + 1;
          if (std::abs(current_objective - prev_objective) < tolerance_)
          {
            result.converged = true;
            result.objective_value = current_objective;
            break;
          }

          prev_objective = current_objective;
        }

        if (!result.converged)
        {
          result.objective_value = objective(result.parameters);
        }

        return result;
      }

      /**
       * @brief 目的関数を最大化
       *
       * 最小化の符号を反転して実装します。
       *
       * @param objective 目的関数 f(x) -> T
       * @param initial_params 初期パラメータ
       * @param bounds パラメータの境界（オプション、空の場合は制約なし）
       * @param gradient 勾配関数 ∇f(x) ->
       * std::vector<T>（オプション、nullptrの場合は数値微分を使用）
       * @return 最適化結果
       */
      OptimizationResult<T> maximize(const std::function<T(const std::vector<T>&)>& objective,
        const std::vector<T>& initial_params,
        const std::vector<std::pair<T, T>>& bounds = {},
        const std::function<std::vector<T>(const std::vector<T>&)>* gradient = nullptr) override
      {
        // 最大化は負の最小化として実装
        auto neg_objective = [&objective](const std::vector<T>& params) -> T
        {
          return -objective(params);
        };

        // 勾配が提供されている場合、負の勾配を計算する関数を作成
        std::function<std::vector<T>(const std::vector<T>&)> neg_grad_func;
        const std::function<std::vector<T>(const std::vector<T>&)>* neg_gradient = nullptr;
        if (gradient != nullptr)
        {
          neg_grad_func = [gradient](const std::vector<T>& params) -> std::vector<T>
          {
            std::vector<T> grad = (*gradient)(params);
            for (auto& g : grad)
            {
              g = -g;
            }
            return grad;
          };
          neg_gradient = &neg_grad_func;
        }

        OptimizationResult<T> result =
          minimize(neg_objective, initial_params, bounds, neg_gradient);
        result.objective_value = -result.objective_value;
        return result;
      }

    private:
      /**
       * @brief 学習率
       */
      T learning_rate_;

      /**
       * @brief 最大反復回数
       */
      size_t max_iterations_;

      /**
       * @brief 収束判定の許容誤差
       */
      T tolerance_;

      /**
       * @brief 数値微分の微小値
       */
      T epsilon_;
  };

}  // namespace optimize
