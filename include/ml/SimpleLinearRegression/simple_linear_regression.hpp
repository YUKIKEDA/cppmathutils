#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ml
{

  /**
   * @brief 単回帰（Simple Linear Regression）モデル
   *
   * 1つの説明変数 x を用いて目的変数 y を予測する線形回帰モデル。
   * 最小二乗法（OLS）を用いてパラメータ w_0（切片）と w_1（傾き）を推定します。
   *
   * モデル: y = w_0 + w_1 * x + ε
   *
   * @tparam T 浮動小数点型（float, double, long double など）
   */
  template <std::floating_point T = double>
  class SimpleLinearRegression
  {
    public:
      static_assert(std::is_floating_point_v<T>,
        "T must be a floating-point type (float, double, or long double)");

      /**
       * @brief デフォルトコンストラクタ
       */
      SimpleLinearRegression():
          w0_(T(0)),
          w1_(T(0)),
          fitted_(false)
      {
      }

      /**
       * @brief データからモデルを学習（フィッティング）
       *
       * 最小二乗法（OLS）を用いてパラメータ w_0 と w_1 を推定します。
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-by-least-squares">Parameter
       * Estimation by Least Squares</a>
       *
       * @param x 説明変数のベクトル
       * @param y 目的変数のベクトル
       * @throws std::invalid_argument x と y のサイズが異なる場合、またはデータが空の場合
       */
      void fit(const std::vector<T>& x, const std::vector<T>& y)
      {
        if (x.size() != y.size())
        {
          throw std::invalid_argument("x and y must have the same size");
        }
        if (x.empty())
        {
          throw std::invalid_argument("x and y must not be empty");
        }

        // 最小二乗法でパラメータを推定
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-formulas">Parameter
        // Estimation Formulas</a>
        compute_ols_parameters(x, y);

        fitted_ = true;
      }

      /**
       * @brief 単一の値に対する予測
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#estimated-values">Estimated
       * Values</a>
       *
       * @param x 説明変数の値
       * @return 予測値 y = w_0 + w_1 * x
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T predict(T x) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#estimated-values">Estimated
        // Values</a>
        return w0_ + w1_ * x;
      }

      /**
       * @brief 複数の値に対する予測
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#estimated-values">Estimated
       * Values</a>
       *
       * @param x 説明変数のベクトル
       * @return 予測値のベクトル
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      std::vector<T> predict(const std::vector<T>& x) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }

        std::vector<T> y_pred(x.size());
        for (size_t i = 0; i < x.size(); ++i)
        {
          // @see <a
          // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#estimated-values">Estimated
          // Values</a>
          y_pred[i] = w0_ + w1_ * x[i];
        }
        return y_pred;
      }

      /**
       * @brief 切片 w_0 を取得
       *
       * @return 切片の値
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T get_w0() const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        return w0_;
      }

      /**
       * @brief 傾き w_1 を取得
       *
       * @return 傾きの値
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T get_w1() const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        return w1_;
      }

      /**
       * @brief 決定係数（R²）を計算
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#coefficient-of-determination-r">Coefficient
       * of Determination (R²)</a>
       *
       * @param x 説明変数のベクトル
       * @param y 目的変数のベクトル（実測値）
       * @return 決定係数 R²
       * @throws std::invalid_argument x と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T score(const std::vector<T>& x, const std::vector<T>& y) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (x.size() != y.size())
        {
          throw std::invalid_argument("x and y must have the same size");
        }

        const size_t n = x.size();
        if (n == 0)
        {
          return T(0);
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#coefficient-of-determination-r">Coefficient
        // of Determination (R²)</a> y の標本平均を計算
        T mean_y = T(0);
        for (size_t i = 0; i < n; ++i)
        {
          mean_y += y[i];
        }
        mean_y /= static_cast<T>(n);

        // 残差平方和（SSR）と総平方和（SST）を計算
        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#coefficient-of-determination-r">Coefficient
        // of Determination (R²)</a>
        T ssr = T(0);  // Sum of Squared Residuals
        T sst = T(0);  // Total Sum of Squares

        for (size_t i = 0; i < n; ++i)
        {
          const T y_pred = predict(x[i]);
          const T residual = y[i] - y_pred;  // 残差
          const T deviation = y[i] - mean_y;
          ssr += residual * residual;        // SSR の累積
          sst += deviation * deviation;      // SST の累積
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#coefficient-of-determination-r">Coefficient
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
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#mean-squared-error-mse">Mean
       * Squared Error (MSE)</a>
       *
       * @param x 説明変数のベクトル
       * @param y 目的変数のベクトル（実測値）
       * @return 平均二乗誤差
       * @throws std::invalid_argument x と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T mse(const std::vector<T>& x, const std::vector<T>& y) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (x.size() != y.size())
        {
          throw std::invalid_argument("x and y must have the same size");
        }

        const size_t n = x.size();
        if (n == 0)
        {
          return T(0);
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#mean-squared-error-mse">Mean
        // Squared Error (MSE)</a>
        T sum_squared_error = T(0);
        for (size_t i = 0; i < n; ++i)
        {
          const T error = y[i] - predict(x[i]);  // 残差
          sum_squared_error += error * error;    // 二乗誤差の累積
        }

        return sum_squared_error / static_cast<T>(n);
      }

      /**
       * @brief 平均二乗平方根誤差（RMSE）を計算
       *
       * @see <a href="README.md#平均二乗平方根誤差rmse">平均二乗平方根誤差（RMSE）</a>
       *
       * @param x 説明変数のベクトル
       * @param y 目的変数のベクトル（実測値）
       * @return 平均二乗平方根誤差
       * @throws std::invalid_argument x と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T rmse(const std::vector<T>& x, const std::vector<T>& y) const
      {
        // @see <a href="README.md#平均二乗平方根誤差rmse">平均二乗平方根誤差（RMSE）</a>
        return std::sqrt(mse(x, y));
      }

      /**
       * @brief 平均絶対誤差（MAE）を計算
       *
       * @see <a href="README.md#平均絶対誤差mae">平均絶対誤差（MAE）</a>
       *
       * @param x 説明変数のベクトル
       * @param y 目的変数のベクトル（実測値）
       * @return 平均絶対誤差
       * @throws std::invalid_argument x と y のサイズが異なる場合
       * @throws std::runtime_error モデルがまだフィッティングされていない場合
       */
      T mae(const std::vector<T>& x, const std::vector<T>& y) const
      {
        if (!fitted_)
        {
          throw std::runtime_error("Model has not been fitted yet. Call fit() first.");
        }
        if (x.size() != y.size())
        {
          throw std::invalid_argument("x and y must have the same size");
        }

        const size_t n = x.size();
        if (n == 0)
        {
          return T(0);
        }

        // @see <a href="README.md#平均絶対誤差mae">平均絶対誤差（MAE）</a>
        T sum_absolute_error = T(0);
        for (size_t i = 0; i < n; ++i)
        {
          sum_absolute_error += std::abs(y[i] - predict(x[i]));  // 絶対誤差の累積
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
        w0_ = T(0);
        w1_ = T(0);
        fitted_ = false;
      }

    private:
      /**
       * @brief 最小二乗法（OLS）によるパラメータ推定
       *
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-by-least-squares">Parameter
       * Estimation by Least Squares</a>
       * @see <a
       * href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-formulas">Parameter
       * Estimation Formulas</a>
       *
       * @param x 説明変数のベクトル
       * @param y 目的変数のベクトル
       */
      void compute_ols_parameters(const std::vector<T>& x, const std::vector<T>& y)
      {
        const size_t n = x.size();

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-formulas">Parameter
        // Estimation Formulas</a> x と y の標本平均を計算
        T mean_x = T(0);
        T mean_y = T(0);
        for (size_t i = 0; i < n; ++i)
        {
          mean_x += x[i];
          mean_y += y[i];
        }
        mean_x /= static_cast<T>(n);
        mean_y /= static_cast<T>(n);

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-formulas">Parameter
        // Estimation Formulas</a> x の偏差平方和（S_xx）と共分散の分子（S_xy）を計算
        T s_xx = T(0);
        T s_xy = T(0);

        for (size_t i = 0; i < n; ++i)
        {
          const T dx = x[i] - mean_x;  // 偏差
          const T dy = y[i] - mean_y;  // 偏差
          s_xx += dx * dx;             // S_xx の累積
          s_xy += dx * dy;             // S_xy の累積
        }

        // @see <a
        // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-formulas">Parameter
        // Estimation Formulas</a> 傾き（w_1）と切片（w_0）を計算
        if (std::abs(s_xx) < std::numeric_limits<T>::epsilon())
        {
          // x の分散が0の場合（すべての x が同じ値）
          // この場合、傾きは定義できないため w1 = 0 とし、w0 = y の平均とする
          w1_ = T(0);
          w0_ = mean_y;
        }
        else
        {
          // @see <a
          // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#slope-estimation">Slope
          // Estimation</a>
          w1_ = s_xy / s_xx;
          // @see <a
          // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#intercept-estimation">Intercept
          // Estimation</a>
          w0_ = mean_y - w1_ * mean_x;
        }
      }

      T w0_;  // 切片（intercept）、@see <a
              // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-formulas">Parameter
              // Estimation Formulas</a>
      T w1_;  // 傾き（slope）、@see <a
              // href="https://github.com/YUKIKEDA/cppmathutils/blob/master/include/ml/SimpleLinearRegression/README.md#parameter-estimation-formulas">Parameter
              // Estimation Formulas</a>
      bool fitted_;  // フィッティング済みかどうか
  };

}  // namespace ml
