#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <optimize/optimizer_base.hpp>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace optimize
{

  /**
   * @brief 焼きなまし法 (Simulated Annealing) アルゴリズム
   *
   * 金属の焼きなまし過程にヒントを得た確率的な最適化アルゴリズム。
   * 初期には高い温度で広く探索し、徐々に温度を下げることで最適解に収束させます。
   * 局所最適解から脱出して大域最適解に近づくことを目的としています。
   *
   * @tparam T 浮動小数点型
   */
  template <std::floating_point T>
  class SimulatedAnnealing: public OptimizerBase<T>
  {
    private:
      // パラメータ
      T initial_temperature_;       // 初期温度 T_0
      T cooling_rate_;              // 冷却率 α (0 < α < 1)
      T min_temperature_;           // 最小温度 T_min
      size_t max_iterations_;       // 最大反復回数
      size_t markov_chain_length_;  // マルコフ連鎖の長さ L（各温度での反復回数）
      T step_size_;                 // ステップサイズ（近傍生成用）
      bool adaptive_step_size_;     // ステップサイズを温度に応じて適応的に変更するか
      std::mt19937 rng_;            // 乱数生成器

      /**
       * @brief 境界制約を適用
       *
       * @param x パラメータベクトル
       * @param bounds 境界制約
       */
      void apply_bounds(std::vector<T>& x, const std::vector<std::pair<T, T>>& bounds)
      {
        if (bounds.empty())
        {
          return;
        }

        if (bounds.size() != x.size())
        {
          throw std::invalid_argument("Bounds size mismatch");
        }

        for (size_t i = 0; i < x.size(); ++i)
        {
          x[i] = std::clamp(x[i], bounds[i].first, bounds[i].second);
        }
      }

      /**
       * @brief 近傍解を生成（ガウス分布による提案）
       *
       * @param current 現在の解
       * @param temperature 現在の温度
       * @return 候補解
       */
      std::vector<T> generate_neighbor(const std::vector<T>& current, T temperature)
      {
        const size_t dim = current.size();
        std::vector<T> neighbor(dim);

        // ステップサイズを計算（温度に応じて調整する場合）
        T sigma = step_size_;
        if (adaptive_step_size_)
        {
          // σ_t = σ_0 * sqrt(T_t / T_0)
          sigma = step_size_ * std::sqrt(temperature / initial_temperature_);
        }

        // ガウス分布からサンプル生成
        std::normal_distribution<T> normal_dist(0.0, sigma);
        for (size_t i = 0; i < dim; ++i)
        {
          neighbor[i] = current[i] + normal_dist(rng_);
        }

        return neighbor;
      }

      /**
       * @brief 受理確率を計算（最小化の場合）
       *
       * @param current_value 現在の目的関数値
       * @param proposed_value 提案された目的関数値
       * @param temperature 現在の温度
       * @return 受理確率
       */
      T compute_acceptance_probability_minimize(T current_value, T proposed_value, T temperature)
      {
        T delta_f = proposed_value - current_value;

        // 改善する場合（Δf < 0）は常に受理
        if (delta_f < 0)
        {
          return 1.0;
        }

        // 悪化する場合（Δf > 0）は確率的に受理
        // P = exp(-Δf / T)
        T exponent = -delta_f / temperature;

        // 数値安定性のため、指数が小さすぎる場合は0を返す
        if (exponent < -50.0)
        {
          return 0.0;
        }

        return std::exp(exponent);
      }

      /**
       * @brief 受理確率を計算（最大化の場合）
       *
       * @param current_value 現在の目的関数値
       * @param proposed_value 提案された目的関数値
       * @param temperature 現在の温度
       * @return 受理確率
       */
      T compute_acceptance_probability_maximize(T current_value, T proposed_value, T temperature)
      {
        T delta_f = proposed_value - current_value;

        // 改善する場合（Δf > 0）は常に受理
        if (delta_f > 0)
        {
          return 1.0;
        }

        // 悪化する場合（Δf < 0）は確率的に受理
        // P = exp(Δf / T)
        T exponent = delta_f / temperature;

        // 数値安定性のため、指数が小さすぎる場合は0を返す
        if (exponent < -50.0)
        {
          return 0.0;
        }

        return std::exp(exponent);
      }

      /**
       * @brief 温度を更新（指数冷却）
       *
       * @param current_temperature 現在の温度
       * @return 更新された温度
       */
      T update_temperature(T current_temperature)
      {
        // 指数冷却: T_{t+1} = α * T_t
        return cooling_rate_ * current_temperature;
      }

    public:
      /**
       * @brief コンストラクタ
       *
       * @param initial_temperature 初期温度 T_0（デフォルト: 100.0）
       * @param cooling_rate 冷却率 α（デフォルト: 0.95、0 < α < 1）
       * @param min_temperature 最小温度 T_min（デフォルト: 1e-6）
       * @param max_iterations 最大反復回数（デフォルト: 10000）
       * @param markov_chain_length マルコフ連鎖の長さ L（デフォルト: 100）
       * @param step_size ステップサイズ（デフォルト: 1.0）
       * @param adaptive_step_size ステップサイズを温度に応じて適応的に変更するか（デフォルト:
       * true）
       * @param seed 乱数生成器のシード（デフォルト: ランダム）
       */
      explicit SimulatedAnnealing(T initial_temperature = T(100.0),
        T cooling_rate = T(0.95),
        T min_temperature = T(1e-6),
        size_t max_iterations = 10'000,
        size_t markov_chain_length = 100,
        T step_size = T(1.0),
        bool adaptive_step_size = true,
        std::optional<unsigned int> seed = std::nullopt):
          initial_temperature_(initial_temperature),
          cooling_rate_(cooling_rate),
          min_temperature_(min_temperature),
          max_iterations_(max_iterations),
          markov_chain_length_(markov_chain_length),
          step_size_(step_size),
          adaptive_step_size_(adaptive_step_size),
          rng_(seed.has_value() ? *seed : std::random_device{}())
      {
        if (initial_temperature <= 0)
        {
          throw std::invalid_argument("initial_temperature must be greater than 0");
        }
        if (cooling_rate <= 0 || cooling_rate >= 1)
        {
          throw std::invalid_argument("cooling_rate must be in (0, 1)");
        }
        if (min_temperature <= 0)
        {
          throw std::invalid_argument("min_temperature must be greater than 0");
        }
        if (min_temperature >= initial_temperature)
        {
          throw std::invalid_argument("min_temperature must be less than initial_temperature");
        }
        if (max_iterations == 0)
        {
          throw std::invalid_argument("max_iterations must be greater than 0");
        }
        if (markov_chain_length == 0)
        {
          throw std::invalid_argument("markov_chain_length must be greater than 0");
        }
        if (step_size <= 0)
        {
          throw std::invalid_argument("step_size must be greater than 0");
        }
      }

      /**
       * @brief 目的関数を最小化
       *
       * @param objective 目的関数 f(x) -> T
       * @param initial_params 初期パラメータ
       * @param bounds パラメータの境界（オプション、空の場合は制約なし）
       * @param gradient 勾配関数（未使用、焼きなまし法では勾配は不要）
       * @return 最適化結果
       */
      OptimizationResult<T> minimize(const std::function<T(const std::vector<T>&)>& objective,
        const std::vector<T>& initial_params,
        const std::vector<std::pair<T, T>>& bounds = {},
        const std::function<std::vector<T>(const std::vector<T>&)>* gradient = nullptr) override
      {
        (void)gradient;  // 未使用

        if (initial_params.empty())
        {
          throw std::invalid_argument("initial_params must not be empty");
        }

        // 初期状態
        std::vector<T> current_params = initial_params;
        apply_bounds(current_params, bounds);
        T current_value = objective(current_params);

        // 最良の状態を追跡
        std::vector<T> best_params = current_params;
        T best_value = current_value;

        // 温度の初期化
        T temperature = initial_temperature_;

        // 一様乱数生成器（受理判定用）
        std::uniform_real_distribution<T> uniform_dist(0.0, 1.0);

        // 焼きなまし法のメインループ
        size_t total_iterations = 0;
        size_t accepted = 0;

        while (temperature > min_temperature_ && total_iterations < max_iterations_)
        {
          // 各温度でのマルコフ連鎖
          for (size_t t = 0; t < markov_chain_length_ && total_iterations < max_iterations_; ++t)
          {
            // 近傍解を生成
            std::vector<T> proposed_params = generate_neighbor(current_params, temperature);
            apply_bounds(proposed_params, bounds);

            // 提案された状態の目的関数値を計算
            T proposed_value = objective(proposed_params);

            // 受理確率を計算
            T acceptance_prob =
              compute_acceptance_probability_minimize(current_value, proposed_value, temperature);

            // 受理/棄却判定
            if (uniform_dist(rng_) < acceptance_prob)
            {
              current_params = proposed_params;
              current_value = proposed_value;
              ++accepted;

              // 最良の状態を更新
              if (proposed_value < best_value)
              {
                best_params = proposed_params;
                best_value = proposed_value;
              }
            }

            ++total_iterations;
          }

          // 温度を更新
          temperature = update_temperature(temperature);
        }

        // 収束判定（簡易版：温度が最小温度以下になったか、最大反復回数に達したか）
        bool converged = (temperature <= min_temperature_);

        OptimizationResult<T> result;
        result.parameters = best_params;
        result.objective_value = best_value;
        result.iterations = total_iterations;
        result.converged = converged;

        return result;
      }

      /**
       * @brief 目的関数を最大化
       *
       * @param objective 目的関数 f(x) -> T
       * @param initial_params 初期パラメータ
       * @param bounds パラメータの境界（オプション、空の場合は制約なし）
       * @param gradient 勾配関数（未使用、焼きなまし法では勾配は不要）
       * @return 最適化結果
       */
      OptimizationResult<T> maximize(const std::function<T(const std::vector<T>&)>& objective,
        const std::vector<T>& initial_params,
        const std::vector<std::pair<T, T>>& bounds = {},
        const std::function<std::vector<T>(const std::vector<T>&)>* gradient = nullptr) override
      {
        (void)gradient;  // 未使用

        if (initial_params.empty())
        {
          throw std::invalid_argument("initial_params must not be empty");
        }

        // 初期状態
        std::vector<T> current_params = initial_params;
        apply_bounds(current_params, bounds);
        T current_value = objective(current_params);

        // 最良の状態を追跡
        std::vector<T> best_params = current_params;
        T best_value = current_value;

        // 温度の初期化
        T temperature = initial_temperature_;

        // 一様乱数生成器（受理判定用）
        std::uniform_real_distribution<T> uniform_dist(0.0, 1.0);

        // 焼きなまし法のメインループ
        size_t total_iterations = 0;
        size_t accepted = 0;

        while (temperature > min_temperature_ && total_iterations < max_iterations_)
        {
          // 各温度でのマルコフ連鎖
          for (size_t t = 0; t < markov_chain_length_ && total_iterations < max_iterations_; ++t)
          {
            // 近傍解を生成
            std::vector<T> proposed_params = generate_neighbor(current_params, temperature);
            apply_bounds(proposed_params, bounds);

            // 提案された状態の目的関数値を計算
            T proposed_value = objective(proposed_params);

            // 受理確率を計算
            T acceptance_prob =
              compute_acceptance_probability_maximize(current_value, proposed_value, temperature);

            // 受理/棄却判定
            if (uniform_dist(rng_) < acceptance_prob)
            {
              current_params = proposed_params;
              current_value = proposed_value;
              ++accepted;

              // 最良の状態を更新
              if (proposed_value > best_value)
              {
                best_params = proposed_params;
                best_value = proposed_value;
              }
            }

            ++total_iterations;
          }

          // 温度を更新
          temperature = update_temperature(temperature);
        }

        // 収束判定（簡易版：温度が最小温度以下になったか、最大反復回数に達したか）
        bool converged = (temperature <= min_temperature_);

        OptimizationResult<T> result;
        result.parameters = best_params;
        result.objective_value = best_value;
        result.iterations = total_iterations;
        result.converged = converged;

        return result;
      }
  };

}  // namespace optimize
