#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ml
{

  /**
   * @brief 線形回帰（Linear Regression）モデル
   *
   * パラメータ（重み）に対して線形なモデルを用いて目的変数を予測する統計的手法。
   * 基底関数（basis functions）を任意に指定できる一般化された線形回帰モデルです。
   *
   * モデル: y = w_0 + w_1 * φ_1(x) + w_2 * φ_2(x) + ... + w_p * φ_p(x) + ε
   *
   * ここで φ_i(x) は任意の基底関数（x, x², sin(x), exp(x) など）です。
   * 重要なのは、重み w_i に対して線形であることです。
   *
   * @see <a
   * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md">Linear
   * Regression Documentation</a>
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class LinearRegression
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
       */
      LinearRegression():
          basis_functions_(),
          coefficients_(),
          fitted_(false),
          n_samples_(0),
          n_basis_functions_(0)
      {
      }

      /**
       * @brief 基底関数を指定してコンストラクタ
       *
       * @param basis_functions 基底関数のベクトル（φ_1, φ_2, ..., φ_p）
       *                        注意: φ_0 = 1（切片項）は自動的に追加されます
       */
      explicit LinearRegression(const std::vector<BasisFunction>& basis_functions):
          basis_functions_(basis_functions),
          coefficients_(),
          fitted_(false),
          n_samples_(0),
          n_basis_functions_(basis_functions.size())
      {
      }

      /**
       * @brief データからモデルを学習（フィッティング）
       *
       * 最小二乗法（OLS）を用いてパラメータ w_0, w_1, ..., w_p を推定します。
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#parameter-estimation-by-least-squares">Parameter
       * Estimation by Least Squares</a>
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

        // 最小二乗法でパラメータを推定
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#parameter-estimation-formulas">Parameter
        // Estimation Formulas</a>
        compute_ols_parameters(X, y);

        fitted_ = true;
      }

      /**
       * @brief 単一の観測値に対する予測
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#estimated-values">Estimated
       * Values</a>
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#estimated-values">Estimated
        // Values</a>
        T y_pred = coefficients_[0];                                // w_0 (切片、φ_0 = 1)
        for (size_t j = 0; j < n_basis_functions_; ++j)
        {
          y_pred += coefficients_[j + 1] * basis_functions_[j](x);  // w_j * φ_j(x)
        }
        return y_pred;
      }

      /**
       * @brief 複数の観測値に対する予測
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#estimated-values">Estimated
       * Values</a>
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
          // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#estimated-values">Estimated
          // Values</a>
          y_pred[i] = predict(X[i]);
        }
        return y_pred;
      }

      /**
       * @brief 回帰係数を取得
       *
       * @return 回帰係数のベクトル（サイズ: p+1、最初の要素が切片 w_0、以降が w_1, w_2, ..., w_p）
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
       * @return 切片の値
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T get_intercept() const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        return coefficients_[0];
      }

      /**
       * @brief 決定係数（R²）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#coefficient-of-determination-r">Coefficient
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#coefficient-of-determination-r">Coefficient
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#coefficient-of-determination-r">Coefficient
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#coefficient-of-determination-r">Coefficient
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
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#adjusted-r">Adjusted
       * R²</a>
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
        if (n <= n_basis_functions_ + 1)
        {
          // サンプル数がパラメータ数以下では調整済みR²は定義できない
          return T(0);
        }

        const T r2 = score(X, y);
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#adjusted-r">Adjusted
        // R²</a>
        return T(1)
          - (T(1) - r2) * static_cast<T>(n - 1) / static_cast<T>(n - n_basis_functions_ - 1);
      }

      /**
       * @brief 平均二乗誤差（MSE）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#mean-squared-error-mse">Mean
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#mean-squared-error-mse">Mean
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
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#root-mean-squared-error-rmse">Root
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#root-mean-squared-error-rmse">Root
        // Mean Squared Error (RMSE)</a>
        return std::sqrt(mse(X, y));
      }

      /**
       * @brief 平均絶対誤差（MAE）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#mean-absolute-error-mae">Mean
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
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#mean-absolute-error-mae">Mean
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
       * @brief 最小二乗法（OLS）によるパラメータ推定
       *
       * QR分解を用いて数値的に安定な方法で正規方程式を解きます。
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#parameter-estimation-by-least-squares">Parameter
       * Estimation by Least Squares</a>
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#computational-efficiency">Computational
       * Efficiency</a>
       *
       * @param X 説明変数の行列（各行が1つの観測値、各列が説明変数、サイズ: n × d）
       * @param y 目的変数のベクトル（サイズ: n）
       */
      void compute_ols_parameters(const std::vector<std::vector<T>>& X, const std::vector<T>& y)
      {
        const size_t n = X.size();
        const size_t p = n_basis_functions_;

        // Eigen行列に変換（計画行列: n × (p+1)、最初の列はすべて1（φ_0 = 1））
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Phi(n, p + 1);
        Eigen::Matrix<T, Eigen::Dynamic, 1> y_vector(n);

        for (size_t i = 0; i < n; ++i)
        {
          Phi(i, 0) = T(1);  // φ_0(x) = 1（切片項）
          for (size_t j = 0; j < p; ++j)
          {
            // @see <a
            // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#matrix-notation">Matrix
            // Notation</a>
            Phi(i, j + 1) = basis_functions_[j](X[i]);  // φ_j(x_i)
          }
          y_vector(i) = y[i];
        }

        // QR分解を用いて正規方程式を解く
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#computational-efficiency">Computational
        // Efficiency</a>
        Eigen::HouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> qr(Phi);
        Eigen::Matrix<T, Eigen::Dynamic, 1> w_vector = qr.solve(y_vector);

        // 係数を保存（w_0, w_1, ..., w_p）
        coefficients_.resize(p + 1);
        for (size_t j = 0; j <= p; ++j)
        {
          coefficients_[j] = w_vector(j);
        }
      }

      /**
       * @brief 基底関数（φ_1, φ_2, ..., φ_p）
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#basis-functions">Basis
       * Functions</a>
       *
       * 注意: φ_0 = 1（切片項）は自動的に追加されるため、ここには含まれません
       */
      std::vector<BasisFunction> basis_functions_;

      /**
       * @brief 回帰係数（w_0, w_1, ..., w_p）
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/LinearRegression/README.md#parameter-estimation-formulas">Parameter
       * Estimation Formulas</a>
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
  };

}  // namespace ml
