#include <cmath>
#include <gtest/gtest.h>
#include <ml/SimpleLinearRegression/simple_linear_regression.hpp>
#include <stdexcept>
#include <vector>

namespace ml
{
  namespace test
  {

    /**
     * @brief 基本的なフィッティングと予測のテスト
     */
    TEST(SimpleLinearRegressionTest, BasicFitAndPredict)
    {
      SimpleLinearRegression<double> model;

      // テストデータ: y = 2x + 1
      std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(x, y);

      // パラメータの検証
      EXPECT_NEAR(model.get_w0(), 1.0, 1e-10);
      EXPECT_NEAR(model.get_w1(), 2.0, 1e-10);

      // 予測の検証
      EXPECT_NEAR(model.predict(0.0), 1.0, 1e-10);
      EXPECT_NEAR(model.predict(6.0), 13.0, 1e-10);
      EXPECT_NEAR(model.predict(10.0), 21.0, 1e-10);
    }

    /**
     * @brief 複数の値に対する予測のテスト
     */
    TEST(SimpleLinearRegressionTest, PredictVector)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {1.0, 2.0, 3.0};
      std::vector<double> y = {3.0, 5.0, 7.0};

      model.fit(x, y);

      std::vector<double> x_test = {0.0, 4.0, 5.0};
      std::vector<double> y_pred = model.predict(x_test);

      EXPECT_EQ(y_pred.size(), 3);
      EXPECT_NEAR(y_pred[0], 1.0, 1e-10);
      EXPECT_NEAR(y_pred[1], 9.0, 1e-10);
      EXPECT_NEAR(y_pred[2], 11.0, 1e-10);
    }

    /**
     * @brief 決定係数（R²）のテスト
     */
    TEST(SimpleLinearRegressionTest, ScoreR2)
    {
      SimpleLinearRegression<double> model;

      // 完全に線形なデータ: y = 2x + 1
      std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(x, y);

      // 完全に線形なデータなので R² は 1.0 に近いはず
      double r2 = model.score(x, y);
      EXPECT_NEAR(r2, 1.0, 1e-10);
    }

    /**
     * @brief 平均二乗誤差（MSE）のテスト
     */
    TEST(SimpleLinearRegressionTest, MSE)
    {
      SimpleLinearRegression<double> model;

      // 完全に線形なデータ
      std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(x, y);

      // 完全に線形なデータなので MSE は 0 に近いはず
      double mse = model.mse(x, y);
      EXPECT_NEAR(mse, 0.0, 1e-10);
    }

    /**
     * @brief 平均二乗平方根誤差（RMSE）のテスト
     */
    TEST(SimpleLinearRegressionTest, RMSE)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {1.0, 2.0, 3.0};
      std::vector<double> y = {3.0, 5.0, 7.0};

      model.fit(x, y);

      double rmse = model.rmse(x, y);
      double mse = model.mse(x, y);

      // RMSE = sqrt(MSE)
      EXPECT_NEAR(rmse, std::sqrt(mse), 1e-10);
    }

    /**
     * @brief 平均絶対誤差（MAE）のテスト
     */
    TEST(SimpleLinearRegressionTest, MAE)
    {
      SimpleLinearRegression<double> model;

      // 完全に線形なデータ
      std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
      std::vector<double> y = {3.0, 5.0, 7.0, 9.0, 11.0};

      model.fit(x, y);

      // 完全に線形なデータなので MAE は 0 に近いはず
      double mae = model.mae(x, y);
      EXPECT_NEAR(mae, 0.0, 1e-10);
    }

    /**
     * @brief ノイズを含むデータでのテスト
     */
    TEST(SimpleLinearRegressionTest, NoisyData)
    {
      SimpleLinearRegression<double> model;

      // y = 2x + 1 + ノイズ
      std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
      std::vector<double> y = {3.1, 4.9, 7.2, 8.8, 11.1, 12.9, 15.2, 16.8, 19.1, 20.9};

      model.fit(x, y);

      // パラメータは真の値（w0=1, w1=2）に近いはず
      EXPECT_NEAR(model.get_w0(), 1.0, 0.2);
      EXPECT_NEAR(model.get_w1(), 2.0, 0.1);

      // R² は高い値になるはず
      double r2 = model.score(x, y);
      EXPECT_GT(r2, 0.95);
    }

    /**
     * @brief 負の値を含むデータのテスト
     */
    TEST(SimpleLinearRegressionTest, NegativeValues)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {-2.0, -1.0, 0.0, 1.0, 2.0};
      std::vector<double> y = {-3.0, -1.0, 1.0, 3.0, 5.0};  // y = 2x + 1

      model.fit(x, y);

      EXPECT_NEAR(model.get_w0(), 1.0, 1e-10);
      EXPECT_NEAR(model.get_w1(), 2.0, 1e-10);
    }

    /**
     * @brief すべての x が同じ値の場合のテスト
     */
    TEST(SimpleLinearRegressionTest, ConstantX)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {5.0, 5.0, 5.0, 5.0, 5.0};
      std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

      model.fit(x, y);

      // x の分散が0なので、w1 は 0 になり、w0 は y の平均になる
      EXPECT_NEAR(model.get_w1(), 0.0, 1e-10);
      EXPECT_NEAR(model.get_w0(), 3.0, 1e-10);  // y の平均
    }

    /**
     * @brief すべての y が同じ値の場合のテスト
     */
    TEST(SimpleLinearRegressionTest, ConstantY)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
      std::vector<double> y = {5.0, 5.0, 5.0, 5.0, 5.0};

      model.fit(x, y);

      // y が一定なので、w1 = 0, w0 = y の値
      EXPECT_NEAR(model.get_w1(), 0.0, 1e-10);
      EXPECT_NEAR(model.get_w0(), 5.0, 1e-10);
    }

    /**
     * @brief 単一データポイントのテスト
     */
    TEST(SimpleLinearRegressionTest, SingleDataPoint)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {1.0};
      std::vector<double> y = {3.0};

      model.fit(x, y);

      // 1点だけでは一意の直線が決まる（その点を通る任意の直線）
      // 実装では w1 = 0, w0 = y になるはず（x の分散が0のため）
      EXPECT_NEAR(model.get_w1(), 0.0, 1e-10);
      EXPECT_NEAR(model.get_w0(), 3.0, 1e-10);
    }

    /**
     * @brief 空のデータでのエラーテスト
     */
    TEST(SimpleLinearRegressionTest, EmptyDataError)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x;
      std::vector<double> y;

      EXPECT_THROW(model.fit(x, y), std::invalid_argument);
    }

    /**
     * @brief サイズ不一致のエラーテスト
     */
    TEST(SimpleLinearRegressionTest, SizeMismatchError)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {1.0, 2.0, 3.0};
      std::vector<double> y = {1.0, 2.0};

      EXPECT_THROW(model.fit(x, y), std::invalid_argument);
    }

    /**
     * @brief 未フィッティングでの予測エラーテスト
     */
    TEST(SimpleLinearRegressionTest, PredictWithoutFitError)
    {
      SimpleLinearRegression<double> model;

      EXPECT_THROW(model.predict(1.0), std::runtime_error);
      EXPECT_THROW(model.predict(std::vector<double>{1.0, 2.0}), std::runtime_error);
    }

    /**
     * @brief 未フィッティングでのパラメータ取得エラーテスト
     */
    TEST(SimpleLinearRegressionTest, GetParametersWithoutFitError)
    {
      SimpleLinearRegression<double> model;

      EXPECT_THROW(model.get_w0(), std::runtime_error);
      EXPECT_THROW(model.get_w1(), std::runtime_error);
    }

    /**
     * @brief 未フィッティングでの評価指標エラーテスト
     */
    TEST(SimpleLinearRegressionTest, ScoreWithoutFitError)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {1.0, 2.0, 3.0};
      std::vector<double> y = {1.0, 2.0, 3.0};

      EXPECT_THROW(model.score(x, y), std::runtime_error);
      EXPECT_THROW(model.mse(x, y), std::runtime_error);
      EXPECT_THROW(model.rmse(x, y), std::runtime_error);
      EXPECT_THROW(model.mae(x, y), std::runtime_error);
    }

    /**
     * @brief 評価指標でのサイズ不一致エラーテスト
     */
    TEST(SimpleLinearRegressionTest, ScoreSizeMismatchError)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x_train = {1.0, 2.0, 3.0};
      std::vector<double> y_train = {3.0, 5.0, 7.0};
      model.fit(x_train, y_train);

      std::vector<double> x_test = {1.0, 2.0};
      std::vector<double> y_test = {3.0, 5.0, 7.0};

      EXPECT_THROW(model.score(x_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.mse(x_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.rmse(x_test, y_test), std::invalid_argument);
      EXPECT_THROW(model.mae(x_test, y_test), std::invalid_argument);
    }

    /**
     * @brief is_fitted() のテスト
     */
    TEST(SimpleLinearRegressionTest, IsFitted)
    {
      SimpleLinearRegression<double> model;

      EXPECT_FALSE(model.is_fitted());

      std::vector<double> x = {1.0, 2.0, 3.0};
      std::vector<double> y = {3.0, 5.0, 7.0};
      model.fit(x, y);

      EXPECT_TRUE(model.is_fitted());
    }

    /**
     * @brief reset() のテスト
     */
    TEST(SimpleLinearRegressionTest, Reset)
    {
      SimpleLinearRegression<double> model;

      std::vector<double> x = {1.0, 2.0, 3.0};
      std::vector<double> y = {3.0, 5.0, 7.0};
      model.fit(x, y);

      EXPECT_TRUE(model.is_fitted());

      model.reset();

      EXPECT_FALSE(model.is_fitted());
      EXPECT_THROW(model.predict(1.0), std::runtime_error);
    }

    /**
     * @brief float型でのテスト
     */
    TEST(SimpleLinearRegressionTest, FloatType)
    {
      SimpleLinearRegression<float> model;

      std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
      std::vector<float> y = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};

      model.fit(x, y);

      EXPECT_NEAR(model.get_w0(), 1.0f, 1e-5f);
      EXPECT_NEAR(model.get_w1(), 2.0f, 1e-5f);
    }

    /**
     * @brief 大きなデータセットでのテスト
     */
    TEST(SimpleLinearRegressionTest, LargeDataset)
    {
      SimpleLinearRegression<double> model;

      const size_t n = 1'000;
      std::vector<double> x(n);
      std::vector<double> y(n);

      // y = 2x + 1
      for (size_t i = 0; i < n; ++i)
      {
        x[i] = static_cast<double>(i);
        y[i] = 2.0 * x[i] + 1.0;
      }

      model.fit(x, y);

      EXPECT_NEAR(model.get_w0(), 1.0, 1e-10);
      EXPECT_NEAR(model.get_w1(), 2.0, 1e-10);

      double r2 = model.score(x, y);
      EXPECT_NEAR(r2, 1.0, 1e-10);
    }

    /**
     * @brief 実用的なデータセットでのテスト（身長と体重の例）
     */
    TEST(SimpleLinearRegressionTest, RealisticDataset)
    {
      SimpleLinearRegression<double> model;

      // 身長（cm）と体重（kg）のサンプルデータ
      std::vector<double> height = {150.0, 160.0, 170.0, 180.0, 190.0};
      std::vector<double> weight = {50.0, 60.0, 70.0, 80.0, 90.0};

      model.fit(height, weight);

      // モデルが学習できていることを確認
      EXPECT_TRUE(model.is_fitted());

      // 予測が可能であることを確認
      double predicted = model.predict(175.0);
      EXPECT_GT(predicted, 0.0);

      // R² が計算できることを確認
      double r2 = model.score(height, weight);
      EXPECT_GE(r2, 0.0);
      EXPECT_LE(r2, 1.0);
    }

  }  // namespace test
}  // namespace ml
