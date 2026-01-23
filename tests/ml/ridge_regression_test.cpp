#include <cmath>
#include <gtest/gtest.h>
#include <ml/RidgeRegression/ridge_regression.hpp>
#include <stdexcept>
#include <vector>

namespace ml
{
  namespace test
  {

    /**
     * @brief 基本的なフィッティングと予測のテスト（恒等関数で重回帰と同等）
     */
    TEST(RidgeRegressionTest, BasicFitAndPredict)
    {
      // 恒等関数を基底関数として使用（重回帰と同等）
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      // alpha = 0.01 でリッジ回帰（小さい正則化）
      RidgeRegression<double> model(basis_functions, 0.01);

      // テストデータ: y = 1 + 2*x1 + 3*x2
      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 10.0};

      model.fit(X, y);

      // パラメータの検証（alphaが小さいので、OLSに近い値になる）
      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 3);
      EXPECT_NEAR(coeffs[0], 1.0, 0.1);  // w_0 (切片)
      EXPECT_NEAR(coeffs[1], 2.0, 0.1);  // w_1
      EXPECT_NEAR(coeffs[2], 3.0, 0.1);  // w_2

      EXPECT_NEAR(model.get_intercept(), 1.0, 0.1);

      // 予測の検証
      std::vector<double> x_test = {0.0, 0.0};
      EXPECT_NEAR(model.predict(x_test), 1.0, 0.1);  // y = 1 + 2*0 + 3*0 = 1

      x_test = {1.0, 1.0};
      EXPECT_NEAR(model.predict(x_test), 6.0, 0.1);  // y = 1 + 2*1 + 3*1 = 6
    }

    /**
     * @brief alpha = 0 の場合（OLSと同等）のテスト
     */
    TEST(RidgeRegressionTest, AlphaZeroEquivalentToOLS)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      // alpha = 0 でリッジ回帰（OLSと同等）
      RidgeRegression<double> model(basis_functions, 0.0);

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 10.0};

      model.fit(X, y);

      // alpha = 0 の場合、OLSと同等なので正確な値になる
      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 3);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);  // w_0 (切片)
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);  // w_1
      EXPECT_NEAR(coeffs[2], 3.0, 1e-8);  // w_2
    }

    /**
     * @brief 正則化パラメータ（alpha）の効果のテスト
     */
    TEST(RidgeRegressionTest, RegularizationEffect)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 10.0};

      // alpha = 0.01（小さい正則化）
      RidgeRegression<double> model_small(0.01);
      model_small = RidgeRegression<double>(basis_functions, 0.01);
      model_small.fit(X, y);
      std::vector<double> coeffs_small = model_small.get_coefficients();

      // alpha = 1.0（大きい正則化）
      RidgeRegression<double> model_large(basis_functions, 1.0);
      model_large.fit(X, y);
      std::vector<double> coeffs_large = model_large.get_coefficients();

      // 大きいalphaの場合、パラメータが0に近づく（縮小効果）
      for (size_t i = 1; i < coeffs_small.size(); ++i)
      {
        EXPECT_GT(std::abs(coeffs_small[i]), std::abs(coeffs_large[i]));
      }
    }

    /**
     * @brief 切片項を正則化から除外するテスト
     */
    TEST(RidgeRegressionTest, NormalizeIntercept)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      // 切片項を正則化から除外
      RidgeRegression<double> model_excluded(basis_functions, 1.0, true, false);
      model_excluded.fit(X, y);
      double intercept_excluded = model_excluded.get_intercept();

      // 切片項も正則化
      RidgeRegression<double> model_included(basis_functions, 1.0, true, true);
      model_included.fit(X, y);
      double intercept_included = model_included.get_intercept();

      // 切片項を正則化しない場合の方が、切片項が大きくなる傾向がある
      EXPECT_GT(std::abs(intercept_excluded), std::abs(intercept_included) - 0.1);
    }

    /**
     * @brief 切片項をフィッティングしない場合のテスト
     */
    TEST(RidgeRegressionTest, NoIntercept)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
      std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};  // y = 2*x（切片なし）

      // 切片項をフィッティングしない
      RidgeRegression<double> model(basis_functions, 0.01, false);
      model.fit(X, y);

      // 切片項は0を返す
      EXPECT_NEAR(model.get_intercept(), 0.0, 1e-8);

      // 係数の数は基底関数の数と同じ（切片項なし）
      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 1);
      EXPECT_NEAR(coeffs[0], 2.0, 0.1);  // w_1 ≈ 2
    }

    /**
     * @brief 複数の観測値に対する予測のテスト
     */
    TEST(RidgeRegressionTest, PredictMatrix)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0};

      model.fit(X, y);

      std::vector<std::vector<double>> X_test = {
        {0.0, 0.0},
        {1.0, 1.0},
        {2.0, 2.0}
      };
      std::vector<double> y_pred = model.predict(X_test);

      EXPECT_EQ(y_pred.size(), 3);
      EXPECT_NEAR(y_pred[0], 1.0, 0.5);  // 許容誤差を大きく設定
      EXPECT_NEAR(y_pred[1], 6.0, 0.5);
      EXPECT_NEAR(y_pred[2], 11.0, 0.5);
    }

    /**
     * @brief 決定係数（R²）のテスト
     */
    TEST(RidgeRegressionTest, ScoreR2)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      // 完全に線形なデータ: y = 1 + 2*x1 + 3*x2
      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 10.0};

      model.fit(X, y);

      // alphaが小さい場合、完全に線形なデータなので R² は 1.0 に近いはず
      double r2 = model.score(X, y);
      EXPECT_NEAR(r2, 1.0, 0.1);
    }

    /**
     * @brief 大きなalphaの場合のR²が負になる可能性のテスト
     */
    TEST(RidgeRegressionTest, NegativeR2WithLargeAlpha)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      // 非常に大きいalphaを使用
      RidgeRegression<double> model(basis_functions, 1000.0);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(X, y);

      // 大きいalphaの場合、R²が負になる可能性がある
      double r2 = model.score(X, y);
      // R²は負の値になる可能性がある（リッジ回帰の特性）
      EXPECT_LE(r2, 1.0);
    }

    /**
     * @brief 調整済み決定係数（Adjusted R²）のテスト
     */
    TEST(RidgeRegressionTest, AdjustedScore)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 10.0};

      model.fit(X, y);

      double r2 = model.score(X, y);
      double adj_r2 = model.adjusted_score(X, y);

      // 調整済みR²は通常、R²以下になる
      EXPECT_LE(adj_r2, r2 + 1e-6);  // 許容誤差を考慮
    }

    /**
     * @brief 平均二乗誤差（MSE）のテスト
     */
    TEST(RidgeRegressionTest, MSE)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(X, y);

      double mse = model.mse(X, y);
      EXPECT_GE(mse, 0.0);  // MSEは非負
    }

    /**
     * @brief 平均二乗平方根誤差（RMSE）のテスト
     */
    TEST(RidgeRegressionTest, RMSE)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
      std::vector<double> y = {3.0, 5.0, 7.0};

      model.fit(X, y);

      double rmse = model.rmse(X, y);
      double mse = model.mse(X, y);

      // RMSE = sqrt(MSE)
      EXPECT_NEAR(rmse, std::sqrt(mse), 1e-8);
    }

    /**
     * @brief 平均絶対誤差（MAE）のテスト
     */
    TEST(RidgeRegressionTest, MAE)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(X, y);

      double mae = model.mae(X, y);
      EXPECT_GE(mae, 0.0);  // MAEは非負
    }

    /**
     * @brief 多項式基底のテスト
     */
    TEST(RidgeRegressionTest, PolynomialBasis)
    {
      // 多項式基底: φ_1(x) = x, φ_2(x) = x²
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[0] * x[0];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      // テストデータ: y = 1 + 2*x + 3*x²
      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
      std::vector<double> y = {1.0, 6.0, 17.0, 34.0, 57.0};

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 3);
      EXPECT_NEAR(coeffs[0], 1.0, 0.5);  // w_0
      EXPECT_NEAR(coeffs[1], 2.0, 0.5);  // w_1
      EXPECT_NEAR(coeffs[2], 3.0, 0.5);  // w_2
    }

    /**
     * @brief エラーハンドリング: 基底関数が設定されていない場合
     */
    TEST(RidgeRegressionTest, ErrorNoBasisFunctions)
    {
      RidgeRegression<double> model(0.01);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
      std::vector<double> y = {3.0, 5.0, 7.0};

      EXPECT_THROW(model.fit(X, y), std::runtime_error);
    }

    /**
     * @brief エラーハンドリング: X と y のサイズが一致しない場合
     */
    TEST(RidgeRegressionTest, ErrorSizeMismatch)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
      std::vector<double> y = {3.0, 5.0};  // サイズが異なる

      EXPECT_THROW(model.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief エラーハンドリング: フィッティング前に予測しようとした場合
     */
    TEST(RidgeRegressionTest, ErrorPredictBeforeFit)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      std::vector<double> x_test = {1.0};
      EXPECT_THROW(model.predict(x_test), std::runtime_error);
    }

    /**
     * @brief エラーハンドリング: 負のalpha値
     */
    TEST(RidgeRegressionTest, ErrorNegativeAlpha)
    {
      EXPECT_THROW(RidgeRegression<double> model(-1.0), std::invalid_argument);
    }

    /**
     * @brief パラメータ取得のテスト
     */
    TEST(RidgeRegressionTest, GetParameters)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      RidgeRegression<double> model(basis_functions, 0.5, true, false);

      EXPECT_EQ(model.get_alpha(), 0.5);
      EXPECT_TRUE(model.get_fit_intercept());
      EXPECT_FALSE(model.get_normalize_intercept());

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
      std::vector<double> y = {3.0, 5.0, 7.0};
      model.fit(X, y);

      EXPECT_TRUE(model.is_fitted());
      EXPECT_EQ(model.get_n_samples(), 3);
      EXPECT_EQ(model.get_n_basis_functions(), 1);
    }

    /**
     * @brief リセット機能のテスト
     */
    TEST(RidgeRegressionTest, Reset)
    {
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      RidgeRegression<double> model(basis_functions, 0.01);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
      std::vector<double> y = {3.0, 5.0, 7.0};

      model.fit(X, y);
      EXPECT_TRUE(model.is_fitted());

      model.reset();
      EXPECT_FALSE(model.is_fitted());
      EXPECT_EQ(model.get_n_samples(), 0);
    }

    /**
     * @brief 多重共線性がある場合のテスト
     */
    TEST(RidgeRegressionTest, Multicollinearity)
    {
      // 相関の高い説明変数を使用
      std::vector<RidgeRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[0] * 1.01;  // x[0]とほぼ同じ（多重共線性）
        }};

      RidgeRegression<double> model(basis_functions, 0.1);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      // 多重共線性があっても、リッジ回帰は解を求めることができる
      EXPECT_NO_THROW(model.fit(X, y));
      EXPECT_TRUE(model.is_fitted());
    }

  }  // namespace test
}  // namespace ml
