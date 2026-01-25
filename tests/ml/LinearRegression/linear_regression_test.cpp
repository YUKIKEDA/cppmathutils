#include <cmath>
#include <corecrt_math_defines.h>
#include <gtest/gtest.h>
#include <ml/LinearRegression/linear_regression.hpp>
#include <stdexcept>
#include <vector>

namespace ml
{
  namespace test
  {

    /**
     * @brief 基本的なフィッティングと予測のテスト（恒等関数で重回帰と同等）
     */
    TEST(LinearRegressionTest, BasicFitAndPredict)
    {
      // 恒等関数を基底関数として使用（重回帰と同等）
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      LinearRegression<double> model(basis_functions);

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

      // パラメータの検証（w_0 = 1, w_1 = 2, w_2 = 3）
      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 3);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);  // w_0 (切片)
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);  // w_1
      EXPECT_NEAR(coeffs[2], 3.0, 1e-8);  // w_2

      EXPECT_NEAR(model.get_intercept(), 1.0, 1e-8);

      // 予測の検証
      std::vector<double> x_test = {0.0, 0.0};
      EXPECT_NEAR(model.predict(x_test), 1.0, 1e-8);  // y = 1 + 2*0 + 3*0 = 1

      x_test = {1.0, 1.0};
      EXPECT_NEAR(model.predict(x_test), 6.0, 1e-8);  // y = 1 + 2*1 + 3*1 = 6
    }

    /**
     * @brief 多項式基底のテスト
     */
    TEST(LinearRegressionTest, PolynomialBasis)
    {
      // 多項式基底: φ_1(x) = x, φ_2(x) = x²
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[0] * x[0];
        }};

      LinearRegression<double> model(basis_functions);

      // テストデータ: y = 1 + 2*x + 3*x²
      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
      std::vector<double> y = {1.0, 6.0, 17.0, 34.0, 57.0};

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 3);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);  // w_0
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);  // w_1
      EXPECT_NEAR(coeffs[2], 3.0, 1e-8);  // w_2

      // 予測の検証
      std::vector<double> x_test = {5.0};
      EXPECT_NEAR(model.predict(x_test), 86.0, 1e-8);  // y = 1 + 2*5 + 3*25 = 86
    }

    /**
     * @brief 三角関数基底のテスト
     */
    TEST(LinearRegressionTest, TrigonometricBasis)
    {
      // 三角関数基底: φ_1(x) = sin(x), φ_2(x) = cos(x)
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return std::sin(x[0]);
        },
        [](const std::vector<double>& x)
        {
          return std::cos(x[0]);
        }};

      LinearRegression<double> model(basis_functions);

      // テストデータ: y = 1 + 2*sin(x) + 3*cos(x)
      std::vector<std::vector<double>> X = {
        {0.0}, {M_PI / 2.0}, {M_PI}, {3.0 * M_PI / 2.0}, {2.0 * M_PI}};
      std::vector<double> y;
      for (const auto& x : X)
      {
        y.push_back(1.0 + 2.0 * std::sin(x[0]) + 3.0 * std::cos(x[0]));
      }

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 3);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-6);  // w_0
      EXPECT_NEAR(coeffs[1], 2.0, 1e-6);  // w_1
      EXPECT_NEAR(coeffs[2], 3.0, 1e-6);  // w_2
    }

    /**
     * @brief 複数の観測値に対する予測のテスト
     */
    TEST(LinearRegressionTest, PredictMatrix)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      LinearRegression<double> model(basis_functions);

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
      EXPECT_NEAR(y_pred[0], 1.0, 1e-8);
      EXPECT_NEAR(y_pred[1], 6.0, 1e-8);
      EXPECT_NEAR(y_pred[2], 11.0, 1e-8);
    }

    /**
     * @brief 決定係数（R²）のテスト
     */
    TEST(LinearRegressionTest, ScoreR2)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      LinearRegression<double> model(basis_functions);

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

      // 完全に線形なデータなので R² は 1.0 に近いはず
      double r2 = model.score(X, y);
      EXPECT_NEAR(r2, 1.0, 1e-8);
    }

    /**
     * @brief 調整済み決定係数（Adjusted R²）のテスト
     */
    TEST(LinearRegressionTest, AdjustedScore)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      LinearRegression<double> model(basis_functions);

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
      EXPECT_LE(adj_r2, r2);
      // 完全に線形なデータなので、調整済みR²も1.0に近いはず
      EXPECT_NEAR(adj_r2, 1.0, 1e-6);
    }

    /**
     * @brief 平均二乗誤差（MSE）のテスト
     */
    TEST(LinearRegressionTest, MSE)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      LinearRegression<double> model(basis_functions);

      // 完全に線形なデータ
      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 10.0};

      model.fit(X, y);

      // 完全に線形なデータなので MSE は 0 に近いはず
      double mse = model.mse(X, y);
      EXPECT_NEAR(mse, 0.0, 1e-8);
    }

    /**
     * @brief 平均二乗平方根誤差（RMSE）のテスト
     */
    TEST(LinearRegressionTest, RMSE)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}};
      std::vector<double> y = {3.0, 5.0, 7.0};

      model.fit(X, y);

      double rmse = model.rmse(X, y);
      double mse = model.mse(X, y);

      // RMSE = sqrt(MSE)
      EXPECT_NEAR(rmse, std::sqrt(mse), 1e-10);
    }

    /**
     * @brief 平均絶対誤差（MAE）のテスト
     */
    TEST(LinearRegressionTest, MAE)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      LinearRegression<double> model(basis_functions);

      // 完全に線形なデータ
      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0, 11.0, 10.0};

      model.fit(X, y);

      // 完全に線形なデータなので MAE は 0 に近いはず
      double mae = model.mae(X, y);
      EXPECT_NEAR(mae, 0.0, 1e-8);
    }

    /**
     * @brief ノイズを含むデータでのテスト
     */
    TEST(LinearRegressionTest, NoisyData)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      LinearRegression<double> model(basis_functions);

      // y = 1 + 2*x1 + 3*x2 + ノイズ
      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 1.0},
        {1.0, 3.0},
        {3.0, 2.0},
        {2.0, 3.0},
        {4.0, 1.0},
        {1.0, 4.0}
      };
      std::vector<double> y = {6.1, 8.2, 9.1, 11.3, 10.2, 10.1, 13.1, 12.2, 12.1, 15.2};

      model.fit(X, y);

      // パラメータは真の値（w_0=1, w_1=2, w_2=3）に近いはず
      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_NEAR(coeffs[0], 1.0, 1.0);  // 切片（ノイズがあるので許容誤差を緩める）
      EXPECT_NEAR(coeffs[1], 2.0, 0.5);  // w_1
      EXPECT_NEAR(coeffs[2], 3.0, 0.5);  // w_2

      // R² は高い値になるはず
      double r2 = model.score(X, y);
      EXPECT_GT(r2, 0.9);
    }

    /**
     * @brief 空の基底関数でのエラーテスト
     */
    TEST(LinearRegressionTest, EmptyBasisFunctionsError)
    {
      LinearRegression<double> model;

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0};

      EXPECT_THROW(model.fit(X, y), std::runtime_error);
    }

    /**
     * @brief 空のデータでのエラーテスト
     */
    TEST(LinearRegressionTest, EmptyDataError)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      std::vector<std::vector<double>> X;
      std::vector<double> y;

      EXPECT_THROW(model.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief 行数不一致のエラーテスト
     */
    TEST(LinearRegressionTest, RowSizeMismatchError)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}};
      std::vector<double> y = {3.0};

      EXPECT_THROW(model.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief 列数不一致のエラーテスト
     */
    TEST(LinearRegressionTest, ColumnSizeMismatchError)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      std::vector<std::vector<double>> X = {
        {1.0},
        {2.0, 1.0},
        {3.0}
      };
      std::vector<double> y = {3.0, 5.0, 7.0};

      EXPECT_THROW(model.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief 未フィッティングでの予測エラーテスト
     */
    TEST(LinearRegressionTest, PredictWithoutFitError)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      EXPECT_THROW(model.predict(std::vector<double>{1.0}), std::runtime_error);

      std::vector<std::vector<double>> X = {{1.0}};
      EXPECT_THROW(model.predict(X), std::runtime_error);
    }

    /**
     * @brief 未フィッティングでのパラメータ取得エラーテスト
     */
    TEST(LinearRegressionTest, GetParametersWithoutFitError)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      EXPECT_THROW(model.get_coefficients(), std::runtime_error);
      EXPECT_THROW(model.get_intercept(), std::runtime_error);
    }

    /**
     * @brief 未フィッティングでの評価指標エラーテスト
     */
    TEST(LinearRegressionTest, ScoreWithoutFitError)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}};
      std::vector<double> y = {3.0, 5.0};

      EXPECT_THROW(model.score(X, y), std::runtime_error);
      EXPECT_THROW(model.adjusted_score(X, y), std::runtime_error);
      EXPECT_THROW(model.mse(X, y), std::runtime_error);
      EXPECT_THROW(model.rmse(X, y), std::runtime_error);
      EXPECT_THROW(model.mae(X, y), std::runtime_error);
    }

    /**
     * @brief 評価指標でのサイズ不一致エラーテスト
     */
    TEST(LinearRegressionTest, ScoreSizeMismatchError)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      std::vector<std::vector<double>> X_train = {{1.0}, {2.0}, {3.0}};
      std::vector<double> y_train = {3.0, 5.0, 7.0};
      model.fit(X_train, y_train);

      std::vector<std::vector<double>> X_test = {{1.0}, {2.0}};
      std::vector<double> y_test = {3.0, 5.0, 7.0};

      EXPECT_THROW(model.score(X_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.adjusted_score(X_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.mse(X_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.rmse(X_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.mae(X_test, y_test), std::invalid_argument);
    }

    /**
     * @brief is_fitted() のテスト
     */
    TEST(LinearRegressionTest, IsFitted)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      EXPECT_FALSE(model.is_fitted());

      std::vector<std::vector<double>> X = {{1.0}, {2.0}};
      std::vector<double> y = {3.0, 5.0};
      model.fit(X, y);

      EXPECT_TRUE(model.is_fitted());
    }

    /**
     * @brief reset() のテスト
     */
    TEST(LinearRegressionTest, Reset)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      std::vector<std::vector<double>> X = {{1.0}, {2.0}};
      std::vector<double> y = {3.0, 5.0};
      model.fit(X, y);

      EXPECT_TRUE(model.is_fitted());
      EXPECT_EQ(model.get_n_samples(), 2);

      model.reset();

      EXPECT_FALSE(model.is_fitted());
      EXPECT_EQ(model.get_n_samples(), 0);
      EXPECT_THROW(model.predict(std::vector<double>{1.0}), std::runtime_error);
    }

    /**
     * @brief get_n_basis_functions() のテスト
     */
    TEST(LinearRegressionTest, GetNBasisFunctions)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        },
        [](const std::vector<double>& x)
        {
          return x[0] * x[1];
        }};

      LinearRegression<double> model(basis_functions);

      EXPECT_EQ(model.get_n_basis_functions(), 3);
    }

    /**
     * @brief float型でのテスト
     */
    TEST(LinearRegressionTest, FloatType)
    {
      std::vector<LinearRegression<float>::BasisFunction> basis_functions = {
        [](const std::vector<float>& x)
        {
          return x[0];
        },
        [](const std::vector<float>& x)
        {
          return x[1];
        }};

      LinearRegression<float> model(basis_functions);

      std::vector<std::vector<float>> X = {
        {1.0f, 1.0f},
        {2.0f, 1.0f},
        {1.0f, 2.0f}
      };
      std::vector<float> y = {6.0f, 8.0f, 9.0f};

      model.fit(X, y);

      std::vector<float> coeffs = model.get_coefficients();
      EXPECT_NEAR(coeffs[0], 1.0f, 1e-5f);
      EXPECT_NEAR(coeffs[1], 2.0f, 1e-5f);
      EXPECT_NEAR(coeffs[2], 3.0f, 1e-5f);
    }

    /**
     * @brief 大きなデータセットでのテスト
     */
    TEST(LinearRegressionTest, LargeDataset)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        },
        [](const std::vector<double>& x)
        {
          return x[2];
        }};

      LinearRegression<double> model(basis_functions);

      const size_t n = 100;
      std::vector<std::vector<double>> X(n);
      std::vector<double> y(n);

      // y = 1 + 2*x1 + 3*x2 + 4*x3
      for (size_t i = 0; i < n; ++i)
      {
        X[i] = {
          static_cast<double>(i % 10),
          static_cast<double>((i / 10) % 10),
          static_cast<double>((i % 7) + 1)  // 1-7の範囲で変化
        };
        y[i] = 1.0 + 2.0 * X[i][0] + 3.0 * X[i][1] + 4.0 * X[i][2];
      }

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);
      EXPECT_NEAR(coeffs[2], 3.0, 1e-8);
      EXPECT_NEAR(coeffs[3], 4.0, 1e-8);

      double r2 = model.score(X, y);
      EXPECT_NEAR(r2, 1.0, 1e-8);
    }

    /**
     * @brief 混合基底関数のテスト（多項式と三角関数の組み合わせ）
     */
    TEST(LinearRegressionTest, MixedBasisFunctions)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[0] * x[0];
        },
        [](const std::vector<double>& x)
        {
          return std::sin(x[0]);
        }};

      LinearRegression<double> model(basis_functions);

      // テストデータ: y = 1 + 2*x + 3*x² + 4*sin(x)
      std::vector<std::vector<double>> X = {
        {0.0}, {M_PI / 2.0}, {M_PI}, {3.0 * M_PI / 2.0}, {2.0 * M_PI}};
      std::vector<double> y;
      for (const auto& x : X)
      {
        y.push_back(1.0 + 2.0 * x[0] + 3.0 * x[0] * x[0] + 4.0 * std::sin(x[0]));
      }

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 4);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-6);  // w_0
      EXPECT_NEAR(coeffs[1], 2.0, 1e-6);  // w_1
      EXPECT_NEAR(coeffs[2], 3.0, 1e-6);  // w_2
      EXPECT_NEAR(coeffs[3], 4.0, 1e-6);  // w_3
    }

    /**
     * @brief サンプル数が少ない場合の調整済みR²のテスト
     */
    TEST(LinearRegressionTest, AdjustedScoreWithFewSamples)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        },
        [](const std::vector<double>& x)
        {
          return x[1];
        }};

      LinearRegression<double> model(basis_functions);

      // サンプル数が基底関数数+1以下の場合
      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0};
      model.fit(X, y);

      // サンプル数が少なすぎる場合、調整済みR²は0を返す
      double adj_r2 = model.adjusted_score(X, y);
      EXPECT_EQ(adj_r2, 0.0);
    }

    /**
     * @brief 単回帰と同等のテスト（恒等関数1つ）
     */
    TEST(LinearRegressionTest, SingleBasisFunction)
    {
      std::vector<LinearRegression<double>::BasisFunction> basis_functions = {
        [](const std::vector<double>& x)
        {
          return x[0];
        }};

      LinearRegression<double> model(basis_functions);

      // y = 1 + 2*x (単回帰と同等)
      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 2);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);  // w_0
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);  // w_1
    }

  }  // namespace test
}  // namespace ml
