#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <linalg/cholesky.hpp>
#include <optimize/optimizer_base.hpp>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace optimize
{

  /**
   * @brief Adaptive Metropolis (AM) アルゴリズム
   *
   * Metropolis-Hastingsアルゴリズムの改良版で、提案分布の共分散行列を適応的に更新することで、
   * サンプリング効率を大幅に向上させます。
   *
   * @tparam T 浮動小数点型
   */
  template <std::floating_point T>
  class AdaptiveMetropolis: public OptimizerBase<T>
  {
    private:
      // パラメータ
      size_t max_iterations_;     // 最大反復回数
      size_t adaptation_period_;  // 適応期間（共分散行列を更新する期間）
      T scaling_factor_;          // スケーリングパラメータ s_d
      T regularization_;          // 正則化パラメータ ε
      T convergence_tolerance_;   // 収束判定の許容誤差
      std::mt19937 rng_;          // 乱数生成器

      /**
       * @brief 多変量正規分布からサンプルを生成
       *
       * @param mean 平均ベクトル
       * @param covariance 共分散行列
       * @return サンプルベクトル
       */
      std::vector<T> sample_multivariate_normal(
        const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance)
      {
        const Eigen::Index dim = mean.size();
        if (covariance.rows() != dim || covariance.cols() != dim)
        {
          throw std::invalid_argument("Covariance matrix size mismatch");
        }

        // Cholesky分解: C = L * L^T
        Eigen::MatrixXd L;
        try
        {
          L = linalg::cholesky_decompose<T>(covariance);
        }
        catch (const std::runtime_error&)
        {
          // Cholesky分解が失敗した場合、正則化を追加して再試行
          Eigen::MatrixXd regularized = covariance;
          regularized.diagonal().array() += regularization_ * 10.0;
          try
          {
            L = linalg::cholesky_decompose<T>(regularized);
          }
          catch (const std::runtime_error&)
          {
            throw std::runtime_error("Cholesky decomposition failed even after regularization");
          }
        }

        // 標準正規分布からサンプル生成
        std::normal_distribution<T> normal_dist(0.0, 1.0);
        Eigen::VectorXd z(dim);
        for (Eigen::Index i = 0; i < dim; ++i)
        {
          z(i) = normal_dist(rng_);
        }

        // x = mean + L * z
        Eigen::VectorXd sample = mean + L * z;

        // std::vectorに変換
        std::vector<T> result(static_cast<size_t>(dim));
        for (Eigen::Index i = 0; i < dim; ++i)
        {
          result[static_cast<size_t>(i)] = sample(i);
        }

        return result;
      }

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
       * @brief 共分散行列をオンライン更新
       *
       * @param covariance 現在の共分散行列（更新される）
       * @param mean 現在の平均ベクトル（更新される）
       * @param new_sample 新しいサンプル
       * @param iteration 現在の反復回数（1から始まる）
       * @param s_d スケーリングパラメータ
       */
      void update_covariance_online(Eigen::MatrixXd& covariance,
        Eigen::VectorXd& mean,
        const Eigen::VectorXd& new_sample,
        size_t iteration,
        T s_d)
      {
        const size_t dim = mean.size();
        const T t = static_cast<T>(iteration);
        const T t_plus_1 = t + 1.0;

        // 平均の更新: x̄_{t+1} = x̄_t + (1/(t+1)) * (x_{t+1} - x̄_t)
        Eigen::VectorXd mean_new = mean + (new_sample - mean) / t_plus_1;

        // 共分散行列の更新
        // C_{t+1} = (t-1)/t * C_t + s_d/t * (t * x̄_t * x̄_t^T - (t+1) * x̄_{t+1} * x̄_{t+1}^T +
        // x_{t+1} * x_{t+1}^T + ε * I)
        if (iteration > 1)
        {
          Eigen::MatrixXd outer_old = mean * mean.transpose();
          Eigen::MatrixXd outer_new = mean_new * mean_new.transpose();
          Eigen::MatrixXd outer_sample = new_sample * new_sample.transpose();
          Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dim, dim);

          covariance = ((t - 1.0) / t) * covariance
            + (s_d / t)
              * (t * outer_old - t_plus_1 * outer_new + outer_sample + regularization_ * identity);
        }
        else
        {
          // 最初の反復: 初期共分散行列を設定
          Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dim, dim);
          covariance = s_d * (regularization_ * identity);
        }

        mean = mean_new;
      }

      /**
       * @brief Metropolis-Hastingsの受理確率を計算
       *
       * @param current_log_prob 現在の状態の対数確率
       * @param proposed_log_prob 提案された状態の対数確率
       * @return 受理確率
       */
      T compute_acceptance_probability(T current_log_prob, T proposed_log_prob)
      {
        // 対称な提案分布（多変量正規分布）の場合、提案確率の比は1
        // α = min(1, exp(log_prob_proposed - log_prob_current))
        T log_ratio = proposed_log_prob - current_log_prob;
        return std::min(1.0, std::exp(log_ratio));
      }

    public:
      /**
       * @brief コンストラクタ
       *
       * @param max_iterations 最大反復回数（デフォルト: 10000）
       * @param adaptation_period 適応期間（デフォルト: 最大反復回数の50%）
       * @param scaling_factor スケーリングパラメータ（デフォルト: 自動計算）
       * @param regularization 正則化パラメータ ε（デフォルト: 1e-6）
       * @param convergence_tolerance 収束判定の許容誤差（デフォルト: 1e-6）
       * @param seed 乱数生成器のシード（デフォルト: ランダム）
       */
      explicit AdaptiveMetropolis(size_t max_iterations = 10'000,
        size_t adaptation_period = 0,
        T scaling_factor = T(0),
        T regularization = T(1e-6),
        T convergence_tolerance = T(1e-6),
        std::optional<unsigned int> seed = std::nullopt):
          max_iterations_(max_iterations),
          adaptation_period_(adaptation_period == 0 ? max_iterations / 2 : adaptation_period),
          scaling_factor_(scaling_factor),
          regularization_(regularization),
          convergence_tolerance_(convergence_tolerance),
          rng_(seed.has_value() ? *seed : std::random_device{}())
      {
        if (max_iterations == 0)
        {
          throw std::invalid_argument("max_iterations must be greater than 0");
        }
        if (adaptation_period > max_iterations)
        {
          throw std::invalid_argument("adaptation_period must be <= max_iterations");
        }
      }

      /**
       * @brief 目的関数を最小化
       *
       * 目的関数の負の対数を目標分布として扱い、MCMCサンプリングを実行します。
       * 最終的に、サンプルの中で目的関数値が最小のものを返します。
       *
       * @param objective 目的関数 f(x) -> T
       * @param initial_params 初期パラメータ
       * @param bounds パラメータの境界（オプション、空の場合は制約なし）
       * @return 最適化結果
       */
      OptimizationResult<T> minimize(const std::function<T(const std::vector<T>&)>& objective,
        const std::vector<T>& initial_params,
        const std::vector<std::pair<T, T>>& bounds = {}) override
      {
        if (initial_params.empty())
        {
          throw std::invalid_argument("initial_params must not be empty");
        }

        const size_t dim = initial_params.size();

        // スケーリングパラメータが指定されていない場合、自動計算
        T s_d = scaling_factor_;
        if (s_d == T(0))
        {
          // s_d = 2.38^2 / d (Roberts et al., 1997)
          s_d = T(2.38 * 2.38) / static_cast<T>(dim);
        }

        // 初期状態
        std::vector<T> current_params = initial_params;
        apply_bounds(current_params, bounds);
        T current_value = objective(current_params);
        T current_log_prob = -current_value;  // 負の対数（最小化の場合）

        // 最良の状態を追跡
        std::vector<T> best_params = current_params;
        T best_value = current_value;

        // 共分散行列と平均の初期化
        Eigen::VectorXd mean = Eigen::Map<const Eigen::VectorXd>(current_params.data(), dim);
        Eigen::MatrixXd covariance(dim, dim);
        {
          Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dim, dim);
          covariance = s_d * (regularization_ * identity);
        }

        // burnin期間後のサンプル平均を計算するための変数
        Eigen::VectorXd sample_sum = Eigen::VectorXd::Zero(dim);
        size_t sample_count = 0;

        // 一様乱数生成器（受理判定用）
        std::uniform_real_distribution<T> uniform_dist(0.0, 1.0);

        // MCMCサンプリング
        size_t accepted = 0;
        for (size_t iter = 1; iter <= max_iterations_; ++iter)
        {
          // 提案分布から候補を生成
          Eigen::VectorXd current_eigen =
            Eigen::Map<const Eigen::VectorXd>(current_params.data(), dim);
          std::vector<T> proposed_params = sample_multivariate_normal(current_eigen, covariance);
          apply_bounds(proposed_params, bounds);

          // 提案された状態の目的関数値を計算
          T proposed_value = objective(proposed_params);
          T proposed_log_prob = -proposed_value;  // 負の対数

          // 受理確率を計算
          T acceptance_prob = compute_acceptance_probability(current_log_prob, proposed_log_prob);

          // 受理/棄却判定
          if (uniform_dist(rng_) < acceptance_prob)
          {
            current_params = proposed_params;
            current_value = proposed_value;
            current_log_prob = proposed_log_prob;
            ++accepted;

            // 最良の状態を更新
            if (proposed_value < best_value)
            {
              best_params = proposed_params;
              best_value = proposed_value;
            }
          }

          // 適応期間中は共分散行列を更新
          if (iter <= adaptation_period_)
          {
            Eigen::VectorXd sample_eigen =
              Eigen::Map<const Eigen::VectorXd>(current_params.data(), dim);
            update_covariance_online(covariance, mean, sample_eigen, iter, s_d);
          }

          // burnin期間後のサンプルを収集して平均を計算
          if (iter > adaptation_period_)
          {
            Eigen::VectorXd sample_eigen =
              Eigen::Map<const Eigen::VectorXd>(current_params.data(), dim);
            sample_sum += sample_eigen;
            ++sample_count;
          }
        }

        // 収束判定（簡易版：受理率を確認）
        T acceptance_rate = static_cast<T>(accepted) / static_cast<T>(max_iterations_);
        bool converged = (acceptance_rate > 0.1) && (acceptance_rate < 0.9);

        OptimizationResult<T> result;

        // burnin期間後のサンプル平均を返す（サンプルがある場合）
        if (sample_count > 0)
        {
          Eigen::VectorXd sample_mean = sample_sum / static_cast<T>(sample_count);
          result.parameters.resize(dim);
          for (size_t i = 0; i < dim; ++i)
          {
            result.parameters[i] = sample_mean(static_cast<Eigen::Index>(i));
          }
          // 平均パラメータでの目的関数値を計算
          result.objective_value = objective(result.parameters);
        }
        else
        {
          // サンプルがない場合は最良の状態を返す
          result.parameters = best_params;
          result.objective_value = best_value;
        }

        result.iterations = max_iterations_;
        result.converged = converged;

        return result;
      }
  };

}  // namespace optimize
