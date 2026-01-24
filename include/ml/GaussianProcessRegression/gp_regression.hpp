#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <linalg/cholesky.hpp>
#include <memory>
#include <ml/GaussianProcessRegression/ard_kernel.hpp>
#include <ml/GaussianProcessRegression/composite_kernel.hpp>
#include <ml/GaussianProcessRegression/kernel_base.hpp>
#include <ml/GaussianProcessRegression/rbf_kernel.hpp>
#include <ml/GaussianProcessRegression/white_noise_kernel.hpp>
#include <optimize/optimizer_base.hpp>
#include <stats/distance.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ml
{

  /**
   * @brief ガウス過程回帰（Gaussian Process Regression, GPR）モデル
   *
   * ガウス過程回帰は、非線形回帰問題に対して柔軟なモデルを提供するベイジアン機械学習手法です。
   * ガウス過程は、任意の有限個の入力点に対する出力の同時分布が多変量正規分布に従う確率過程として定義されます。
   * この性質により、ガウス過程回帰は不確実性を自然に表現でき、予測の信頼区間を提供することができます。
   *
   * モデル: y = f(x) + ε, f(x) ~ GP(μ(x), k(x_i, x_j)), ε ~ N(0, σ²)
   *
   * 予測分布: p(y_* | x_*, y, X) = N(μ_*, σ_*²)
   * - μ_* = k_*^T (K + σ²I)^(-1) y
   * - σ_*² = k(x_*, x_*) - k_*^T (K + σ²I)^(-1) k_* + σ²
   *
   * @see <a
   * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md">Gaussian
   * Process Regression Documentation</a>
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class GaussianProcessRegression
  {
    public:
      static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (float, double, or long double)");

      /**
       * @brief デフォルトコンストラクタ
       *
       * RBFカーネル（σ_f = 1.0, ℓ = 1.0）で初期化します。
       * 注意: ノイズ分散はカーネルに含める必要があります（例: RBFKernel() + WhiteNoiseKernel()）。
       */
      GaussianProcessRegression():
          kernel_(std::make_shared<RBFKernel<T>>(T(1.0), T(1.0))),
          jitter_(T(1e-6)),
          fitted_(false),
          n_samples_(0),
          input_dim_(0),
          K_(),
          L_(),
          alpha_()
      {
      }

      /**
       * @brief カーネル関数を指定してコンストラクタ
       *
       * 注意: ノイズ分散はカーネルに含める必要があります（例: RBFKernel() + WhiteNoiseKernel()）。
       *
       * @param kernel カーネル関数（shared_ptr）
       * @param jitter 数値的安定性のためのジッター（デフォルト: 1e-6）
       */
      explicit GaussianProcessRegression(std::shared_ptr<KernelBase<T>> kernel, T jitter = T(1e-6)):
          kernel_(kernel),
          jitter_(jitter),
          fitted_(false),
          n_samples_(0),
          input_dim_(0),
          K_(),
          L_(),
          alpha_()
      {
        if (!kernel_)
        {
          throw std::invalid_argument("kernel must not be null");
        }
        if (jitter_ < T(0))
        {
          throw std::invalid_argument("jitter must be non-negative");
        }
      }

      /**
       * @brief RBFカーネルを指定してコンストラクタ
       *
       * 注意: ノイズ分散はカーネルに含める必要があります（例: RBFKernel() + WhiteNoiseKernel()）。
       *
       * @param rbf_kernel RBFカーネル
       * @param jitter 数値的安定性のためのジッター（デフォルト: 1e-6）
       */
      explicit GaussianProcessRegression(const RBFKernel<T>& rbf_kernel, T jitter = T(1e-6)):
          kernel_(std::make_shared<RBFKernel<T>>(rbf_kernel)),
          jitter_(jitter),
          fitted_(false),
          n_samples_(0),
          input_dim_(0),
          K_(),
          L_(),
          alpha_()
      {
        if (jitter_ < T(0))
        {
          throw std::invalid_argument("jitter must be non-negative");
        }
      }

      /**
       * @brief SumKernelを指定してコンストラクタ
       *
       * 注意: ノイズ分散はカーネルに含める必要があります（例: RBFKernel() + WhiteNoiseKernel()）。
       *
       * @param sum_kernel SumKernel（合成カーネル）
       * @param jitter 数値的安定性のためのジッター（デフォルト: 1e-6）
       */
      explicit GaussianProcessRegression(const SumKernel<T>& sum_kernel, T jitter = T(1e-6)):
          kernel_(std::make_shared<SumKernel<T>>(sum_kernel)),
          jitter_(jitter),
          fitted_(false),
          n_samples_(0),
          input_dim_(0),
          K_(),
          L_(),
          alpha_()
      {
        if (jitter_ < T(0))
        {
          throw std::invalid_argument("jitter must be non-negative");
        }
      }

      /**
       * @brief ProductKernelを指定してコンストラクタ
       *
       * 注意: ノイズ分散はカーネルに含める必要があります（例: RBFKernel() + WhiteNoiseKernel()）。
       *
       * @param product_kernel ProductKernel（合成カーネル）
       * @param jitter 数値的安定性のためのジッター（デフォルト: 1e-6）
       */
      explicit GaussianProcessRegression(
        const ProductKernel<T>& product_kernel, T jitter = T(1e-6)):
          kernel_(std::make_shared<ProductKernel<T>>(product_kernel)),
          jitter_(jitter),
          fitted_(false),
          n_samples_(0),
          input_dim_(0),
          K_(),
          L_(),
          alpha_()
      {
        if (jitter_ < T(0))
        {
          throw std::invalid_argument("jitter must be non-negative");
        }
      }

      /**
       * @brief データからモデルを学習（フィッティング）
       *
       * 共分散行列 K を構築し、Cholesky分解を用いて (K + σ²I)^(-1) y を計算します。
       * 最適化アルゴリズムが指定されている場合、対数周辺尤度を最大化するように
       * ハイパーパラメータを最適化します（scikit-learnの設計に準拠）。
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
       * of Predictive Distribution</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（サイズ: n）
       * @param optimizer 最適化アルゴリズム（オプション、nullptrの場合は最適化しない）
       * @throws std::invalid_argument X の行数と y のサイズが異なる場合、またはデータが空の場合
       * @throws std::runtime_error 最適化可能なハイパーパラメータが見つからない場合
       */
      void fit(const std::vector<std::vector<T>>& X,
        const std::vector<T>& y,
        std::shared_ptr<optimize::OptimizerBase<T>> optimizer = nullptr)
      {
        if (X.empty())
        {
          throw std::invalid_argument("X must not be empty");
        }
        if (X.size() != y.size())
        {
          throw std::invalid_argument("Number of rows in X must match size of y");
        }

        // 最適化アルゴリズムが指定されている場合、ハイパーパラメータを最適化
        if (optimizer)
        {
          optimize_hyperparameters(X, y, optimizer);
        }
        else
        {
          // 最適化なしでフィッティング
          fit_internal(X, y);
        }
      }

      /**
       * @brief 単一の観測値に対する予測（平均のみ）
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
       * of Predictive Distribution</a>
       *
       * @param x 説明変数のベクトル（サイズ: d）
       * @return 予測平均 μ_*
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       * @throws std::invalid_argument x のサイズが入力次元と一致しない場合
       */
      T predict(const std::vector<T>& x) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (x.size() != input_dim_)
        {
          throw std::invalid_argument("Size of x must match input dimension");
        }

        // k_* = [k(x_1, x_*), k(x_2, x_*), ..., k(x_n, x_*)]^T を計算
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
        // of Predictive Distribution</a>
        Eigen::Matrix<T, Eigen::Dynamic, 1> k_star(n_samples_);
        for (size_t i = 0; i < n_samples_; ++i)
        {
          k_star(i) = (*kernel_)(X_train_[i], x);
        }

        // μ_* = k_*^T α
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
        // of Predictive Distribution</a>
        return k_star.dot(alpha_);
      }

      /**
       * @brief 単一の観測値に対する予測（平均と分散）
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
       * of Predictive Distribution</a>
       *
       * @param x 説明変数のベクトル（サイズ: d）
       * @param mean 予測平均 μ_* を格納する変数への参照
       * @param variance 予測分散 σ_*² を格納する変数への参照
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       * @throws std::invalid_argument x のサイズが入力次元と一致しない場合
       */
      void predict(const std::vector<T>& x, T& mean, T& variance) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (x.size() != input_dim_)
        {
          throw std::invalid_argument("Size of x must match input dimension");
        }

        // k_* = [k(x_1, x_*), k(x_2, x_*), ..., k(x_n, x_*)]^T を計算
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
        // of Predictive Distribution</a>
        Eigen::Matrix<T, Eigen::Dynamic, 1> k_star(n_samples_);
        for (size_t i = 0; i < n_samples_; ++i)
        {
          k_star(i) = (*kernel_)(X_train_[i], x);
        }

        // μ_* = k_*^T α
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
        // of Predictive Distribution</a>
        mean = k_star.dot(alpha_);

        // v = L^(-1) k_* を計算（前進代入）
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#cholesky-decomposition">Cholesky
        // Decomposition</a>
        Eigen::Matrix<T, Eigen::Dynamic, 1> v = L_.triangularView<Eigen::Lower>().solve(k_star);

        // σ_*² = k(x_*, x_*) - v^T v
        // 注意: ノイズ分散はカーネルに含まれています（例: RBFKernel() + WhiteNoiseKernel()）。
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
        // of Predictive Distribution</a>
        const T k_star_star = (*kernel_)(x, x);
        // WhiteNoiseKernelのノイズ分散を取得
        T white_noise_value = T(0);
        if (auto sum_kernel = std::dynamic_pointer_cast<SumKernel<T>>(kernel_))
        {
          white_noise_value = extract_white_noise_variance_from_sum_kernel(*sum_kernel);
        }
        else if (kernel_->is_white_noise_kernel())
        {
          white_noise_value = kernel_->get_noise_variance_if_white_noise();
        }
        variance = k_star_star - v.squaredNorm() + white_noise_value;

        // 数値誤差により分散が負になる場合があるため、0以下にならないようにする
        if (variance < T(0))
        {
          variance = T(0);
        }
      }

      /**
       * @brief 複数の観測値に対する予測（平均のみ）
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
       * of Predictive Distribution</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: m × d）
       * @return 予測平均のベクトル（サイズ: m）
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       * @throws std::invalid_argument X の列数が入力次元と一致しない場合
       */
      std::vector<T> predict(const std::vector<std::vector<T>>& X) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (X.empty())
        {
          return std::vector<T>();
        }
        if (X[0].size() != input_dim_)
        {
          throw std::invalid_argument("Number of columns in X must match input dimension");
        }

        std::vector<T> y_pred(X.size());
        for (size_t i = 0; i < X.size(); ++i)
        {
          if (X[i].size() != input_dim_)
          {
            throw std::invalid_argument("All rows in X must have the same size");
          }
          y_pred[i] = predict(X[i]);
        }
        return y_pred;
      }

      /**
       * @brief 複数の観測値に対する予測（平均と分散）
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#derivation-of-predictive-distribution">Derivation
       * of Predictive Distribution</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: m × d）
       * @param means 予測平均のベクトル（サイズ: m）を格納する変数への参照
       * @param variances 予測分散のベクトル（サイズ: m）を格納する変数への参照
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       * @throws std::invalid_argument X の列数が入力次元と一致しない場合
       */
      void predict(const std::vector<std::vector<T>>& X,
        std::vector<T>& means,
        std::vector<T>& variances) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (X.empty())
        {
          means.clear();
          variances.clear();
          return;
        }
        if (X[0].size() != input_dim_)
        {
          throw std::invalid_argument("Number of columns in X must match input dimension");
        }

        means.resize(X.size());
        variances.resize(X.size());
        for (size_t i = 0; i < X.size(); ++i)
        {
          if (X[i].size() != input_dim_)
          {
            throw std::invalid_argument("All rows in X must have the same size");
          }
          predict(X[i], means[i], variances[i]);
        }
      }

      /**
       * @brief 対数周辺尤度（Log Marginal Likelihood）を計算
       *
       * 対数周辺尤度は、モデルの証拠（evidence）を表し、データとモデルの適合度を測ります。
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#log-marginal-likelihood">Log
       * Marginal Likelihood</a>
       *
       * log p(y | X, θ) = -0.5 * y^T (K + σ²I)^(-1) y - 0.5 * log|K + σ²I| - 0.5 * n * log(2π)
       *
       * @return 対数周辺尤度
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T log_marginal_likelihood() const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }

        // データ適合項: -0.5 * y^T (K + σ²I)^(-1) y = -0.5 * y^T α
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#log-marginal-likelihood">Log
        // Marginal Likelihood</a>
        Eigen::Matrix<T, Eigen::Dynamic, 1> y_vector(n_samples_);
        for (size_t i = 0; i < n_samples_; ++i)
        {
          y_vector(i) = y_train_[i];
        }
        const T data_fit = T(-0.5) * y_vector.dot(alpha_);

        // 複雑度ペナルティ項: -0.5 * log|K + σ²I| = -sum(log(L_ii))
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#cholesky-decomposition">Cholesky
        // Decomposition</a>
        T log_det = T(0);
        for (size_t i = 0; i < n_samples_; ++i)
        {
          log_det += std::log(L_(i, i));
        }
        const T complexity_penalty = -log_det;

        // 正規化項: -0.5 * n * log(2π)
        const T normalization =
          T(-0.5) * static_cast<T>(n_samples_) * std::log(T(2) * std::acos(T(-1)));

        return data_fit + complexity_penalty + normalization;
      }

      /**
       * @brief 決定係数（R²）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#coefficient-of-determination-r">Coefficient
       * of Determination (R²)</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（実測値、サイズ: n）
       * @return 決定係数 R²
       * @throws std::invalid_argument X の行数と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T score(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (X.size() != y.size())
        {
          throw std::invalid_argument("Number of rows in X must match size of y");
        }

        const size_t n = X.size();
        if (n == 0)
        {
          return T(0);
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#coefficient-of-determination-r">Coefficient
        // of Determination (R²)</a>
        // y の標本平均を計算
        T mean_y = T(0);
        for (size_t i = 0; i < n; ++i)
        {
          mean_y += y[i];
        }
        mean_y /= static_cast<T>(n);

        // 残差平方和（SSR）と総平方和（SST）を計算
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#coefficient-of-determination-r">Coefficient
        // of Determination (R²)</a>
        T ssr = T(0);  // Sum of Squared Residuals
        T sst = T(0);  // Total Sum of Squares

        for (size_t i = 0; i < n; ++i)
        {
          const T y_pred = predict(X[i]);
          const T residual = y[i] - y_pred;  // 残差
          const T deviation = y[i] - mean_y;
          ssr += residual * residual;        // SSR の累積
          sst += deviation * deviation;      // SST の累積
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#coefficient-of-determination-r">Coefficient
        // of Determination (R²)</a>
        if (std::abs(sst) < std::numeric_limits<T>::epsilon())
        {
          // y の分散が0の場合（すべての y が同じ値）
          return T(1);
        }

        return T(1) - ssr / sst;
      }

      /**
       * @brief 平均二乗誤差（MSE）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#mean-squared-error-mse">Mean
       * Squared Error (MSE)</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（実測値、サイズ: n）
       * @return 平均二乗誤差
       * @throws std::invalid_argument X の行数と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T mse(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (X.size() != y.size())
        {
          throw std::invalid_argument("Number of rows in X must match size of y");
        }

        const size_t n = X.size();
        if (n == 0)
        {
          return T(0);
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#mean-squared-error-mse">Mean
        // Squared Error (MSE)</a>
        T sum_squared_error = T(0);
        for (size_t i = 0; i < n; ++i)
        {
          const T error = y[i] - predict(X[i]);  // 残差
          sum_squared_error += error * error;    // 二乗誤差の累積
        }

        return sum_squared_error / static_cast<T>(n);
      }

      /**
       * @brief 平均二乗平方根誤差（RMSE）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#root-mean-squared-error-rmse">Root
       * Mean Squared Error (RMSE)</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（実測値、サイズ: n）
       * @return 平均二乗平方根誤差
       * @throws std::invalid_argument X の行数と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T rmse(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const
      {
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#root-mean-squared-error-rmse">Root
        // Mean Squared Error (RMSE)</a>
        return std::sqrt(mse(X, y));
      }

      /**
       * @brief 平均絶対誤差（MAE）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#mean-absolute-error-mae">Mean
       * Absolute Error (MAE)</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（実測値、サイズ: n）
       * @return 平均絶対誤差
       * @throws std::invalid_argument X の行数と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T mae(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (X.size() != y.size())
        {
          throw std::invalid_argument("Number of rows in X must match size of y");
        }

        const size_t n = X.size();
        if (n == 0)
        {
          return T(0);
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/GaussianProcessRegression/README.md#mean-absolute-error-mae">Mean
        // Absolute Error (MAE)</a>
        T sum_absolute_error = T(0);
        for (size_t i = 0; i < n; ++i)
        {
          sum_absolute_error += std::abs(y[i] - predict(X[i]));  // 絶対誤差の累積
        }

        return sum_absolute_error / static_cast<T>(n);
      }

      /**
       * @brief モデルがフィッティング済みかどうかを確認
       *
       * @return フィッティング済みの場合 true
       */
      bool is_fitted() const
      {
        return fitted_;
      }

      /**
       * @brief モデルをリセット（パラメータを初期化）
       */
      void reset()
      {
        K_.resize(0, 0);
        L_.resize(0, 0);
        alpha_.resize(0);
        X_train_.clear();
        y_train_.clear();
        fitted_ = false;
        n_samples_ = 0;
        input_dim_ = 0;
      }

      /**
       * @brief サンプル数を取得
       *
       * @return フィッティングに使用したサンプル数
       */
      size_t get_n_samples() const
      {
        return n_samples_;
      }

      /**
       * @brief 入力次元数を取得
       *
       * @return 入力次元数
       */
      size_t get_input_dim() const
      {
        return input_dim_;
      }

      /**
       * @brief ジッターを取得
       *
       * @return ジッター ε
       */
      T get_jitter() const
      {
        return jitter_;
      }

      /**
       * @brief ジッターを設定
       *
       * @param jitter ジッター（jitter ≥ 0）
       */
      void set_jitter(T jitter)
      {
        if (jitter < T(0))
        {
          throw std::invalid_argument("jitter must be non-negative");
        }
        jitter_ = jitter;
      }

    private:
      /**
       * @brief カーネル関数
       */
      std::shared_ptr<KernelBase<T>> kernel_;

      /**
       * @brief 数値的安定性のためのジッター ε
       */
      T jitter_;

      /**
       * @brief フィッティング済みかどうか
       */
      bool fitted_;

      /**
       * @brief サンプル数
       */
      size_t n_samples_;

      /**
       * @brief 入力次元数
       */
      size_t input_dim_;

      /**
       * @brief 共分散行列 K（n × n）
       */
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K_;

      /**
       * @brief Cholesky分解の下三角行列 L（C = LL^T）
       */
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> L_;

      /**
       * @brief α = (K + σ²I)^(-1) y
       */
      Eigen::Matrix<T, Eigen::Dynamic, 1> alpha_;

      /**
       * @brief 訓練データ（説明変数）
       */
      std::vector<std::vector<T>> X_train_;

      /**
       * @brief 訓練データ（目的変数）
       */
      std::vector<T> y_train_;

      /**
       * @brief ハイパーパラメータ情報
       */
      struct HyperparameterInfo
      {
          /**
           * @brief 対数空間でのパラメータ値
           */
          std::vector<T> log_params;

          /**
           * @brief パラメータ設定関数
           */
          std::vector<std::function<void(T)>> param_setters;

          /**
           * @brief パラメータの境界（対数空間）
           */
          std::vector<std::pair<T, T>> bounds;
      };

      /**
       * @brief ハイパーパラメータを収集
       *
       * @return ハイパーパラメータ情報
       */
      HyperparameterInfo collect_hyperparameters() const
      {
        HyperparameterInfo info;

        // SumKernel内のカーネルからハイパーパラメータを収集
        auto sum_kernel = std::dynamic_pointer_cast<SumKernel<T>>(kernel_);
        if (sum_kernel)
        {
          const auto& kernels = sum_kernel->get_kernels();
          for (const auto& kernel : kernels)
          {
            collect_kernel_hyperparameters(kernel, info);
          }
        }
        // 単一のカーネルの場合
        else
        {
          collect_kernel_hyperparameters(kernel_, info);
        }

        return info;
      }

      /**
       * @brief カーネルからハイパーパラメータを収集（ヘルパー関数）
       *
       * @param kernel カーネル
       * @param info ハイパーパラメータ情報（出力）
       */
      void collect_kernel_hyperparameters(
        const std::shared_ptr<KernelBase<T>>& kernel, HyperparameterInfo& info) const
      {
        // RBFKernelのハイパーパラメータ
        if (auto rbf = std::dynamic_pointer_cast<RBFKernel<T>>(kernel))
        {
          T sigma_f = rbf->get_sigma_f();
          T length_scale = rbf->get_length_scale();
          info.log_params.push_back(std::log(sigma_f));
          info.log_params.push_back(std::log(length_scale));
          info.param_setters.push_back(
            [rbf](T log_val)
            {
              rbf->set_sigma_f(std::exp(log_val));
            });
          info.param_setters.push_back(
            [rbf](T log_val)
            {
              rbf->set_length_scale(std::exp(log_val));
            });
          // 境界: 正の値のみ（対数空間では -∞ から +∞）
          info.bounds.push_back({std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()});
          info.bounds.push_back({std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()});
        }
        // ARDKernelのハイパーパラメータ
        else if (auto ard = std::dynamic_pointer_cast<ARDKernel<T>>(kernel))
        {
          T sigma_f = ard->get_sigma_f();
          info.log_params.push_back(std::log(sigma_f));
          info.param_setters.push_back(
            [ard](T log_val)
            {
              ard->set_sigma_f(std::exp(log_val));
            });
          info.bounds.push_back({std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()});
          const auto& length_scales = ard->get_length_scales();
          for (size_t d = 0; d < length_scales.size(); ++d)
          {
            info.log_params.push_back(std::log(length_scales[d]));
            info.param_setters.push_back(
              [ard, d](T log_val)
              {
                ard->set_length_scale(d, std::exp(log_val));
              });
            info.bounds.push_back(
              {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()});
          }
        }
        // WhiteNoiseKernelのハイパーパラメータ
        if (auto wn = std::dynamic_pointer_cast<WhiteNoiseKernel<T>>(kernel))
        {
          T noise_variance = wn->get_noise_variance();
          info.log_params.push_back(std::log(noise_variance));
          info.param_setters.push_back(
            [wn](T log_val)
            {
              wn->set_noise_variance(std::exp(log_val));
            });
          info.bounds.push_back({std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max()});
        }
      }

      /**
       * @brief 対数空間のパラメータからハイパーパラメータを設定
       *
       * @param log_params 対数空間でのパラメータ値
       * @param info ハイパーパラメータ情報
       */
      void set_hyperparameters_from_log_space(
        const std::vector<T>& log_params, const HyperparameterInfo& info)
      {
        if (log_params.size() != info.param_setters.size())
        {
          throw std::invalid_argument("Size of log_params must match number of parameters");
        }
        for (size_t i = 0; i < log_params.size(); ++i)
        {
          info.param_setters[i](log_params[i]);
        }
      }

      /**
       * @brief ハイパーパラメータを最適化
       *
       * @param X 説明変数の行列
       * @param y 目的変数のベクトル
       * @param optimizer 最適化アルゴリズム
       */
      void optimize_hyperparameters(const std::vector<std::vector<T>>& X,
        const std::vector<T>& y,
        std::shared_ptr<optimize::OptimizerBase<T>> optimizer)
      {
        // ハイパーパラメータを収集
        HyperparameterInfo param_info = collect_hyperparameters();
        if (param_info.log_params.empty())
        {
          throw std::runtime_error("No optimizable hyperparameters found");
        }

        // 目的関数: 対数周辺尤度を最大化（負の対数周辺尤度を最小化）
        auto objective = [this, &X, &y, &param_info](const std::vector<T>& log_params) -> T
        {
          // パラメータを設定
          set_hyperparameters_from_log_space(log_params, param_info);
          // フィッティング（最適化フラグなしで再帰を防ぐ）
          fit_internal(X, y);
          // 負の対数周辺尤度を返す（最小化のため）
          return -log_marginal_likelihood();
        };

        // 最適化実行（最小化）
        optimize::OptimizationResult<T> result =
          optimizer->minimize(objective, param_info.log_params, param_info.bounds);

        // 最適化されたパラメータを設定
        set_hyperparameters_from_log_space(result.parameters, param_info);
        fit_internal(X, y);
      }

      /**
       * @brief 内部フィッティング（最適化ループ内で使用、再帰を防ぐ）
       *
       * @param X 説明変数の行列
       * @param y 目的変数のベクトル
       */
      void fit_internal(const std::vector<std::vector<T>>& X, const std::vector<T>& y)
      {
        n_samples_ = X.size();
        input_dim_ = X[0].size();

        // すべての行が同じサイズか確認
        for (size_t i = 1; i < n_samples_; ++i)
        {
          if (X[i].size() != input_dim_)
          {
            throw std::invalid_argument("All rows in X must have the same size");
          }
        }

        // 共分散行列 K を構築
        K_ = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_samples_, n_samples_);

        // WhiteNoiseKernelのノイズ分散を事前に取得（効率化）
        T white_noise_variance = T(0);
        if (auto sum_kernel = std::dynamic_pointer_cast<SumKernel<T>>(kernel_))
        {
          white_noise_variance = extract_white_noise_variance_from_sum_kernel(*sum_kernel);
        }
        else if (kernel_->is_white_noise_kernel())
        {
          white_noise_variance = kernel_->get_noise_variance_if_white_noise();
        }

        for (size_t i = 0; i < n_samples_; ++i)
        {
          for (size_t j = 0; j < n_samples_; ++j)
          {
            K_(i, j) = (*kernel_)(X[i], X[j]);
            // WhiteNoiseKernelは対角要素（i == j）でのみ値を持つ
            if (i == j)
            {
              K_(i, j) += white_noise_variance;
            }
          }
        }

        // 共分散行列 C = K + εI を構築（ε はジッター）
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> C = K_;
        C.diagonal().array() += jitter_;

        // Cholesky分解: C = LL^T
        L_ = linalg::cholesky_decompose(C);

        // α = (K + σ²I)^(-1) y = (LL^T)^(-1) y = (L^T)^(-1) L^(-1) y を計算
        Eigen::Matrix<T, Eigen::Dynamic, 1> y_vector(n_samples_);
        for (size_t i = 0; i < n_samples_; ++i)
        {
          y_vector(i) = y[i];
        }
        // 前進代入: L v = y を解く
        Eigen::Matrix<T, Eigen::Dynamic, 1> v = L_.triangularView<Eigen::Lower>().solve(y_vector);
        // 後退代入: L^T α = v を解く
        alpha_ = L_.transpose().triangularView<Eigen::Upper>().solve(v);

        // 訓練データを保存（予測時に使用）
        X_train_ = X;
        y_train_ = y;

        fitted_ = true;
      }
  };

}  // namespace ml
