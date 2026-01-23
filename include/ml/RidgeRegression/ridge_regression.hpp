#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <linalg/cholesky.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ml
{

  /**
   * @brief リッジ回帰（Ridge Regression）モデル
   *
   * 線形回帰にL2正則化（L2 regularization）を追加した手法。
   * 最小二乗法の目的関数に、パラメータのL2ノルム（二乗和）をペナルティ項として加えることで、
   * パラメータの大きさを制約し、過学習や多重共線性の問題を緩和します。
   *
   * モデル: y = w_0 + w_1 * φ_1(x) + w_2 * φ_2(x) + ... + w_p * φ_p(x) + ε
   * 目的関数: J(w) = ||y - Φw||² + λ||w||²
   *
   * ここで φ_i(x) は任意の基底関数（x, x², sin(x), exp(x) など）です。
   * λ は正則化パラメータ（regularization parameter）です。
   *
   * @see <a
   * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md">Ridge
   * Regression Documentation</a>
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class RidgeRegression
  {
    public:
      static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (float, double, or long double)");

      /**
       * @brief 基底関数の型
       *
       * 入力ベクトル x を受け取り、スカラー値を返す関数
       */
      using BasisFunction = std::function<T(const std::vector<T>&)>;

      /**
       * @brief デフォルトコンストラクタ
       *
       * 正則化パラメータ λ = 1.0、切片項を正則化から除外する設定で初期化します。
       */
      RidgeRegression():
          basis_functions_(),
          coefficients_(),
          fitted_(false),
          n_samples_(0),
          n_basis_functions_(0),
          alpha_(T(1.0)),
          fit_intercept_(true),
          normalize_intercept_(false)
      {
      }

      /**
       * @brief 正則化パラメータを指定してコンストラクタ
       *
       * @param alpha 正則化パラメータ λ（alpha ≥ 0）
       * @param fit_intercept 切片項をフィッティングするかどうか（デフォルト: true）
       * @param normalize_intercept 切片項を正則化に含めるかどうか（デフォルト: false）
       *                            注意: fit_intercept が false
       * の場合、このパラメータは無視されます
       */
      explicit RidgeRegression(
        T alpha, bool fit_intercept = true, bool normalize_intercept = false):
          basis_functions_(),
          coefficients_(),
          fitted_(false),
          n_samples_(0),
          n_basis_functions_(0),
          alpha_(alpha),
          fit_intercept_(fit_intercept),
          normalize_intercept_(normalize_intercept)
      {
        if (alpha < T(0))
        {
          throw std::invalid_argument("alpha must be non-negative");
        }
      }

      /**
       * @brief 基底関数と正則化パラメータを指定してコンストラクタ
       *
       * @param basis_functions 基底関数のベクトル（φ_1, φ_2, ..., φ_p）
       *                        注意: φ_0 = 1（切片項）は自動的に追加されます
       * @param alpha 正則化パラメータ λ（alpha ≥ 0、デフォルト: 1.0）
       * @param fit_intercept 切片項をフィッティングするかどうか（デフォルト: true）
       * @param normalize_intercept 切片項を正則化に含めるかどうか（デフォルト: false）
       */
      explicit RidgeRegression(const std::vector<BasisFunction>& basis_functions,
        T alpha = T(1.0),
        bool fit_intercept = true,
        bool normalize_intercept = false):
          basis_functions_(basis_functions),
          coefficients_(),
          fitted_(false),
          n_samples_(0),
          n_basis_functions_(basis_functions.size()),
          alpha_(alpha),
          fit_intercept_(fit_intercept),
          normalize_intercept_(normalize_intercept)
      {
        if (alpha < T(0))
        {
          throw std::invalid_argument("alpha must be non-negative");
        }
      }

      /**
       * @brief 正則化パラメータを取得
       *
       * @return 正則化パラメータ λ
       */
      T get_alpha() const
      {
        return alpha_;
      }

      /**
       * @brief 切片項をフィッティングするかどうかを取得
       *
       * @return 切片項をフィッティングする場合 true
       */
      bool get_fit_intercept() const
      {
        return fit_intercept_;
      }

      /**
       * @brief 切片項を正則化に含めるかどうかを取得
       *
       * @return 切片項を正則化に含める場合 true
       */
      bool get_normalize_intercept() const
      {
        return normalize_intercept_;
      }

      /**
       * @brief データからモデルを学習（フィッティング）
       *
       * リッジ回帰を用いてパラメータ w_0, w_1, ..., w_p を推定します。
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#parameter-estimation">Parameter
       * Estimation</a>
       *
       * 正規方程式: (Φ^T Φ + λI)w = Φ^T y
       * 推定式: w = (Φ^T Φ + λI)^(-1) Φ^T y
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（サイズ: n）
       * @throws std::invalid_argument X の行数と y のサイズが異なる場合、またはデータが空の場合
       * @throws std::runtime_error 基底関数が設定されていない場合
       */
      void fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y)
      {
        if (basis_functions_.empty())
        {
          throw std::runtime_error("Basis functions must be set via constructor before fitting.");
        }
        if (X.empty())
        {
          throw std::invalid_argument("X must not be empty");
        }
        if (X.size() != y.size())
        {
          throw std::invalid_argument("Number of rows in X must match size of y");
        }

        n_samples_ = X.size();
        const size_t input_dim = X[0].size();

        // すべての行が同じサイズか確認
        for (size_t i = 1; i < n_samples_; ++i)
        {
          if (X[i].size() != input_dim)
          {
            throw std::invalid_argument("All rows in X must have the same size");
          }
        }

        // リッジ回帰でパラメータを推定
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#parameter-estimation">Parameter
        // Estimation</a>
        compute_ridge_parameters(X, y);

        fitted_ = true;
      }

      /**
       * @brief 単一の観測値に対する予測
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#evaluation-metrics">Evaluation
       * Metrics</a>
       *
       * @param x 説明変数のベクトル（サイズ: d）
       * @return 予測値 y = w_0 + w_1 * φ_1(x) + w_2 * φ_2(x) + ... + w_p * φ_p(x)
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       * @throws std::invalid_argument x のサイズが入力次元と一致しない場合
       */
      T predict(const std::vector<T>& x) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#evaluation-metrics">Evaluation
        // Metrics</a>
        T y_pred = T(0);
        size_t offset = 0;

        if (fit_intercept_)
        {
          y_pred = coefficients_[0];  // w_0 (切片、φ_0 = 1)
          offset = 1;
        }

        for (size_t j = 0; j < n_basis_functions_; ++j)
        {
          y_pred += coefficients_[j + offset] * basis_functions_[j](x);  // w_j * φ_j(x)
        }
        return y_pred;
      }

      /**
       * @brief 複数の観測値に対する予測
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#evaluation-metrics">Evaluation
       * Metrics</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: m × d）
       * @return 予測値のベクトル（サイズ: m）
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

        std::vector<T> y_pred(X.size());
        for (size_t i = 0; i < X.size(); ++i)
        {
          // @see <a
          // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#evaluation-metrics">Evaluation
          // Metrics</a>
          y_pred[i] = predict(X[i]);
        }
        return y_pred;
      }

      /**
       * @brief 回帰係数を取得
       *
       * @return 回帰係数のベクトル
       *         - fit_intercept = true の場合: サイズ p+1、最初の要素が切片 w_0、以降が w_1, w_2,
       * ..., w_p
       *         - fit_intercept = false の場合: サイズ p、w_1, w_2, ..., w_p
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      std::vector<T> get_coefficients() const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        return coefficients_;
      }

      /**
       * @brief 切片 w_0 を取得
       *
       * @return 切片の値（fit_intercept = false の場合、0を返す）
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T get_intercept() const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (!fit_intercept_)
        {
          return T(0);
        }
        return coefficients_[0];
      }

      /**
       * @brief 決定係数（R²）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#coefficient-of-determination-r">Coefficient
       * of Determination (R²)</a>
       *
       * 注意: リッジ回帰では R² が負の値になる場合があります（特に λ が大きい場合）。
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#coefficient-of-determination-r">Coefficient
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#coefficient-of-determination-r">Coefficient
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#coefficient-of-determination-r">Coefficient
        // of Determination (R²)</a>
        if (std::abs(sst) < std::numeric_limits<T>::epsilon())
        {
          // y の分散が0の場合（すべての y が同じ値）
          return T(1);
        }

        return T(1) - ssr / sst;
      }

      /**
       * @brief 調整済み決定係数（Adjusted R²）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#evaluation-metrics">Evaluation
       * Metrics</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（実測値、サイズ: n）
       * @return 調整済み決定係数 R²_adj
       * @throws std::invalid_argument X の行数と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T adjusted_score(const std::vector<std::vector<T>>& X, const std::vector<T>& y) const
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
        const size_t p = n_basis_functions_;
        const size_t n_params = fit_intercept_ ? (p + 1) : p;

        if (n <= n_params)
        {
          // サンプル数がパラメータ数以下では調整済みR²は定義できない
          return T(0);
        }

        const T r2 = score(X, y);
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#evaluation-metrics">Evaluation
        // Metrics</a>
        return T(1) - (T(1) - r2) * static_cast<T>(n - 1) / static_cast<T>(n - n_params);
      }

      /**
       * @brief 平均二乗誤差（MSE）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#mean-squared-error-mse">Mean
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#mean-squared-error-mse">Mean
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
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#root-mean-squared-error-rmse">Root
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#root-mean-squared-error-rmse">Root
        // Mean Squared Error (RMSE)</a>
        return std::sqrt(mse(X, y));
      }

      /**
       * @brief 平均絶対誤差（MAE）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#mean-absolute-error-mae">Mean
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#mean-absolute-error-mae">Mean
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
        coefficients_.clear();
        fitted_ = false;
        n_samples_ = 0;
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
       * @brief 基底関数の数を取得
       *
       * @return 基底関数の数（φ_0 = 1 は含まない）
       */
      size_t get_n_basis_functions() const
      {
        return n_basis_functions_;
      }

    private:
      /**
       * @brief リッジ回帰によるパラメータ推定
       *
       * 正規方程式 (Φ^T Φ + λI)w = Φ^T y を解きます。
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#parameter-estimation">Parameter
       * Estimation</a>
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#computational-efficiency">Computational
       * Efficiency</a>
       *
       * λ > 0 の場合、行列 (Φ^T Φ + λI) は常に正定値であり、逆行列が一意に存在します。
       * コレスキー分解を用いて効率的に解きます。
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（サイズ: n）
       */
      void compute_ridge_parameters(const std::vector<std::vector<T>>& X, const std::vector<T>& y)
      {
        const size_t n = X.size();
        const size_t p = n_basis_functions_;
        const size_t n_params = fit_intercept_ ? (p + 1) : p;

        // Eigen行列に変換（計画行列: n × n_params）
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Phi(n, n_params);
        Eigen::Matrix<T, Eigen::Dynamic, 1> y_vector(n);

        // 計画行列を構築
        for (size_t i = 0; i < n; ++i)
        {
          size_t col = 0;
          if (fit_intercept_)
          {
            Phi(i, col) = T(1);  // φ_0(x) = 1（切片項）
            ++col;
          }
          for (size_t j = 0; j < p; ++j)
          {
            // @see <a
            // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#matrix-notation">Matrix
            // Notation</a>
            Phi(i, col) = basis_functions_[j](X[i]);  // φ_j(x_i)
            ++col;
          }
          y_vector(i) = y[i];
        }

        // 正規方程式: (Φ^T Φ + λI)w = Φ^T y
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#parameter-estimation">Parameter
        // Estimation</a>
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> PhiTPhi = Phi.transpose() * Phi;
        Eigen::Matrix<T, Eigen::Dynamic, 1> PhiTy = Phi.transpose() * y_vector;

        // 正則化項を追加: Φ^T Φ + λI
        // 切片項を正則化から除外する場合、単位行列の最初の対角要素を0にする
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> I =
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(n_params, n_params);
        if (fit_intercept_ && !normalize_intercept_)
        {
          I(0, 0) = T(0);  // 切片項を正則化から除外
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = PhiTPhi + alpha_ * I;

        // コレスキー分解を用いて解く（数値的に安定）
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#computational-efficiency">Computational
        // Efficiency</a>
        Eigen::Matrix<T, Eigen::Dynamic, 1> w_vector = linalg::cholesky_solve(A, PhiTy);

        // 係数を保存
        coefficients_.resize(n_params);
        for (size_t j = 0; j < n_params; ++j)
        {
          coefficients_[j] = w_vector(j);
        }
      }

      /**
       * @brief 基底関数（φ_1, φ_2, ..., φ_p）
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md">Ridge
       * Regression Documentation</a>
       *
       * 注意: φ_0 = 1（切片項）は自動的に追加されるため、ここには含まれません
       */
      std::vector<BasisFunction> basis_functions_;

      /**
       * @brief 回帰係数（w_0, w_1, ..., w_p または w_1, w_2, ..., w_p）
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#parameter-estimation">Parameter
       * Estimation</a>
       */
      std::vector<T> coefficients_;

      /**
       * @brief フィッティング済みかどうか
       */
      bool fitted_;

      /**
       * @brief サンプル数
       */
      size_t n_samples_;

      /**
       * @brief 基底関数の数（φ_0 = 1 は含まない）
       */
      size_t n_basis_functions_;

      /**
       * @brief 正則化パラメータ λ（alpha）
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#regularization-parameter">Regularization
       * Parameter</a>
       */
      T alpha_;

      /**
       * @brief 切片項をフィッティングするかどうか
       */
      bool fit_intercept_;

      /**
       * @brief 切片項を正則化に含めるかどうか
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/RidgeRegression/README.md#intercept-handling">Intercept
       * Handling</a>
       */
      bool normalize_intercept_;
  };

}  // namespace ml
