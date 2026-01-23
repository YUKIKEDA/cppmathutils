#include <cmath>
#include <gtest/gtest.h>
#include <ml/MultipleLinearRegression/multiple_linear_regression.hpp>
#include <stdexcept>
#include <vector>

namespace ml
{
  namespace test
  {

    /**
     * @brief 基本的なフィッティングと予測のテスト
     */
    TEST(MultipleLinearRegressionTest, BasicFitAndPredict)
    {
      MultipleLinearRegression<double> model;

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

      // パラメータの検証（β_0 = 1, β_1 = 2, β_2 = 3）
      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 3);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);  // β_0 (切片)
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);  // β_1
      EXPECT_NEAR(coeffs[2], 3.0, 1e-8);  // β_2

      EXPECT_NEAR(model.get_intercept(), 1.0, 1e-8);

      // 予測の検証
      std::vector<double> x_test = {0.0, 0.0};
      EXPECT_NEAR(model.predict(x_test), 1.0, 1e-8);  // y = 1 + 2*0 + 3*0 = 1

      x_test = {1.0, 1.0};
      EXPECT_NEAR(model.predict(x_test), 6.0, 1e-8);  // y = 1 + 2*1 + 3*1 = 6
    }

    /**
     * @brief 複数の観測値に対する予測のテスト
     */
    TEST(MultipleLinearRegressionTest, PredictMatrix)
    {
      MultipleLinearRegression<double> model;

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
    TEST(MultipleLinearRegressionTest, ScoreR2)
    {
      MultipleLinearRegression<double> model;

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
    TEST(MultipleLinearRegressionTest, AdjustedScore)
    {
      MultipleLinearRegression<double> model;

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
    TEST(MultipleLinearRegressionTest, MSE)
    {
      MultipleLinearRegression<double> model;

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
    TEST(MultipleLinearRegressionTest, RMSE)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0};

      model.fit(X, y);

      double rmse = model.rmse(X, y);
      double mse = model.mse(X, y);

      // RMSE = sqrt(MSE)
      EXPECT_NEAR(rmse, std::sqrt(mse), 1e-10);
    }

    /**
     * @brief 平均絶対誤差（MAE）のテスト
     */
    TEST(MultipleLinearRegressionTest, MAE)
    {
      MultipleLinearRegression<double> model;

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
    TEST(MultipleLinearRegressionTest, NoisyData)
    {
      MultipleLinearRegression<double> model;

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

      // パラメータは真の値（β_0=1, β_1=2, β_2=3）に近いはず
      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_NEAR(coeffs[0], 1.0, 1.0);  // 切片（ノイズがあるので許容誤差を緩める）
      EXPECT_NEAR(coeffs[1], 2.0, 0.5);  // β_1
      EXPECT_NEAR(coeffs[2], 3.0, 0.5);  // β_2

      // R² は高い値になるはず
      double r2 = model.score(X, y);
      EXPECT_GT(r2, 0.9);
    }

    /**
     * @brief 負の値を含むデータのテスト
     */
    TEST(MultipleLinearRegressionTest, NegativeValues)
    {
      MultipleLinearRegression<double> model;

      // x1 と x2 を独立にする（多重共線性を避ける）
      // より多くの点を追加して安定性を向上
      std::vector<std::vector<double>> X = {
        {-2.0, -1.0},
        {-1.0, -2.0},
        {0.0,  -1.0},
        {1.0,  0.0 },
        {2.0,  1.0 },
        {-1.0, 1.0 },
        {1.0,  -1.0}
      };
      // y = 1 + 2*x1 + 3*x2
      std::vector<double> y = {
        1.0 + 2.0 * (-2.0) + 3.0 * (-1.0),  // -6.0
        1.0 + 2.0 * (-1.0) + 3.0 * (-2.0),  // -7.0
        1.0 + 2.0 * (0.0) + 3.0 * (-1.0),   // -2.0
        1.0 + 2.0 * (1.0) + 3.0 * (0.0),    // 3.0
        1.0 + 2.0 * (2.0) + 3.0 * (1.0),    // 8.0
        1.0 + 2.0 * (-1.0) + 3.0 * (1.0),   // 2.0
        1.0 + 2.0 * (1.0) + 3.0 * (-1.0)    // 0.0
      };

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);
      EXPECT_NEAR(coeffs[2], 3.0, 1e-8);
    }

    /**
     * @brief 単一説明変数のテスト（単回帰と同等）
     */
    TEST(MultipleLinearRegressionTest, SingleFeature)
    {
      MultipleLinearRegression<double> model;

      // y = 1 + 2*x1 (単回帰と同等)
      std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 2);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);  // β_0
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);  // β_1
    }

    /**
     * @brief 3変数のテスト
     */
    TEST(MultipleLinearRegressionTest, ThreeFeatures)
    {
      MultipleLinearRegression<double> model;

      // y = 1 + 2*x1 + 3*x2 + 4*x3
      std::vector<std::vector<double>> X = {
        {1.0, 1.0, 1.0},
        {2.0, 1.0, 1.0},
        {1.0, 2.0, 1.0},
        {1.0, 1.0, 2.0},
        {2.0, 2.0, 2.0}
      };
      std::vector<double> y = {10.0, 12.0, 13.0, 14.0, 19.0};

      model.fit(X, y);

      std::vector<double> coeffs = model.get_coefficients();
      EXPECT_EQ(coeffs.size(), 4);
      EXPECT_NEAR(coeffs[0], 1.0, 1e-8);
      EXPECT_NEAR(coeffs[1], 2.0, 1e-8);
      EXPECT_NEAR(coeffs[2], 3.0, 1e-8);
      EXPECT_NEAR(coeffs[3], 4.0, 1e-8);
    }

    /**
     * @brief 空のデータでのエラーテスト
     */
    TEST(MultipleLinearRegressionTest, EmptyDataError)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X;
      std::vector<double> y;

      EXPECT_THROW(model.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief 行数不一致のエラーテスト
     */
    TEST(MultipleLinearRegressionTest, RowSizeMismatchError)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0}
      };
      std::vector<double> y = {6.0};

      EXPECT_THROW(model.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief 列数不一致のエラーテスト
     */
    TEST(MultipleLinearRegressionTest, ColumnSizeMismatchError)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0},
        {1.0, 2.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0};

      EXPECT_THROW(model.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief 説明変数が0個のエラーテスト
     */
    TEST(MultipleLinearRegressionTest, ZeroFeaturesError)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X = {{}, {}, {}};
      std::vector<double> y = {6.0, 8.0, 9.0};

      EXPECT_THROW(model.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief 未フィッティングでの予測エラーテスト
     */
    TEST(MultipleLinearRegressionTest, PredictWithoutFitError)
    {
      MultipleLinearRegression<double> model;

      EXPECT_THROW(model.predict(std::vector<double>{1.0, 1.0}), std::runtime_error);

      std::vector<std::vector<double>> X = {
        {1.0, 1.0}
      };
      EXPECT_THROW(model.predict(X), std::runtime_error);
    }

    /**
     * @brief 未フィッティングでのパラメータ取得エラーテスト
     */
    TEST(MultipleLinearRegressionTest, GetParametersWithoutFitError)
    {
      MultipleLinearRegression<double> model;

      EXPECT_THROW(model.get_coefficients(), std::runtime_error);
      EXPECT_THROW(model.get_intercept(), std::runtime_error);
    }

    /**
     * @brief 未フィッティングでの評価指標エラーテスト
     */
    TEST(MultipleLinearRegressionTest, ScoreWithoutFitError)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0};

      EXPECT_THROW(model.score(X, y), std::runtime_error);
      EXPECT_THROW(model.adjusted_score(X, y), std::runtime_error);
      EXPECT_THROW(model.mse(X, y), std::runtime_error);
      EXPECT_THROW(model.rmse(X, y), std::runtime_error);
      EXPECT_THROW(model.mae(X, y), std::runtime_error);
    }

    /**
     * @brief 評価指標でのサイズ不一致エラーテスト
     */
    TEST(MultipleLinearRegressionTest, ScoreSizeMismatchError)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X_train = {
        {1.0, 1.0},
        {2.0, 1.0},
        {1.0, 2.0}
      };
      std::vector<double> y_train = {6.0, 8.0, 9.0};
      model.fit(X_train, y_train);

      std::vector<std::vector<double>> X_test = {
        {1.0, 1.0},
        {2.0, 1.0}
      };
      std::vector<double> y_test = {6.0, 8.0, 9.0};

      EXPECT_THROW(model.score(X_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.adjusted_score(X_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.mse(X_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.rmse(X_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.mae(X_test, y_test), std::invalid_argument);
    }

    /**
     * @brief 予測時の特徴数不一致エラーテスト
     */
    TEST(MultipleLinearRegressionTest, PredictFeatureMismatchError)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X_train = {
        {1.0, 1.0},
        {2.0, 1.0}
      };
      std::vector<double> y_train = {6.0, 8.0};
      model.fit(X_train, y_train);

      // 特徴数が異なる
      std::vector<double> x_test = {1.0};
      EXPECT_THROW(model.predict(x_test), std::invalid_argument);

      std::vector<std::vector<double>> X_test = {
        {1.0, 1.0, 1.0}
      };
      EXPECT_THROW(model.predict(X_test), std::invalid_argument);
    }

    /**
     * @brief is_fitted() のテスト
     */
    TEST(MultipleLinearRegressionTest, IsFitted)
    {
      MultipleLinearRegression<double> model;

      EXPECT_FALSE(model.is_fitted());

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0};
      model.fit(X, y);

      EXPECT_TRUE(model.is_fitted());
    }

    /**
     * @brief reset() のテスト
     */
    TEST(MultipleLinearRegressionTest, Reset)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X = {
        {1.0, 1.0},
        {2.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0};
      model.fit(X, y);

      EXPECT_TRUE(model.is_fitted());
      EXPECT_EQ(model.get_n_samples(), 2);
      EXPECT_EQ(model.get_n_features(), 2);

      model.reset();

      EXPECT_FALSE(model.is_fitted());
      EXPECT_EQ(model.get_n_samples(), 0);
      EXPECT_EQ(model.get_n_features(), 0);
      EXPECT_THROW(model.predict(std::vector<double>{1.0, 1.0}), std::runtime_error);
    }

    /**
     * @brief get_n_samples() と get_n_features() のテスト
     */
    TEST(MultipleLinearRegressionTest, GetNSamplesAndFeatures)
    {
      MultipleLinearRegression<double> model;

      std::vector<std::vector<double>> X = {
        {1.0, 1.0, 1.0},
        {2.0, 1.0, 1.0},
        {1.0, 2.0, 1.0}
      };
      std::vector<double> y = {6.0, 8.0, 9.0};
      model.fit(X, y);

      EXPECT_EQ(model.get_n_samples(), 3);
      EXPECT_EQ(model.get_n_features(), 3);
    }

    /**
     * @brief float型でのテスト
     */
    TEST(MultipleLinearRegressionTest, FloatType)
    {
      MultipleLinearRegression<float> model;

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
    TEST(MultipleLinearRegressionTest, LargeDataset)
    {
      MultipleLinearRegression<double> model;

      const size_t n = 100;
      std::vector<std::vector<double>> X(n);
      std::vector<double> y(n);

      // y = 1 + 2*x1 + 3*x2 + 4*x3
      // 各特徴量を独立にする（多重共線性を避ける）
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
     * @brief 実用的なデータセットでのテスト（住宅価格の例）
     */
    TEST(MultipleLinearRegressionTest, RealisticDataset)
    {
      MultipleLinearRegression<double> model;

      // 住宅価格の予測: 価格 = β_0 + β_1*面積 + β_2*築年数
      std::vector<std::vector<double>> X = {
        {50.0, 5.0 }, // 50m², 築5年
        {60.0, 3.0 }, // 60m², 築3年
        {70.0, 10.0}, // 70m², 築10年
        {80.0, 2.0 }, // 80m², 築2年
        {90.0, 8.0 }  // 90m², 築8年
      };
      std::vector<double> y = {3000.0, 3500.0, 4000.0, 4500.0, 5000.0};

      model.fit(X, y);

      // モデルが学習できていることを確認
      EXPECT_TRUE(model.is_fitted());
      EXPECT_EQ(model.get_n_samples(), 5);
      EXPECT_EQ(model.get_n_features(), 2);

      // 予測が可能であることを確認
      std::vector<double> x_test = {75.0, 5.0};
      double predicted = model.predict(x_test);
      EXPECT_GT(predicted, 0.0);

      // R² が計算できることを確認
      double r2 = model.score(X, y);
      EXPECT_GE(r2, 0.0);
      EXPECT_LE(r2, 1.0);

      // 調整済みR²も計算できることを確認
      double adj_r2 = model.adjusted_score(X, y);
      EXPECT_GE(adj_r2, 0.0);
      EXPECT_LE(adj_r2, 1.0);
    }

    /**
     * @brief サンプル数が少ない場合の調整済みR²のテスト
     */
    TEST(MultipleLinearRegressionTest, AdjustedScoreWithFewSamples)
    {
      MultipleLinearRegression<double> model;

      // サンプル数が特徴数+1以下の場合
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

  }  // namespace test
}  // namespace ml
