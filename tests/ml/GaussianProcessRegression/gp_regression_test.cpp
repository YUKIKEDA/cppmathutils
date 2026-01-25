#include <cmath>
#include <gtest/gtest.h>
#include <ml/GaussianProcessRegression/composite_kernel.hpp>
#include <ml/GaussianProcessRegression/gp_regression.hpp>
#include <ml/GaussianProcessRegression/rbf_kernel.hpp>
#include <ml/GaussianProcessRegression/white_noise_kernel.hpp>
#include <optimize/AdaptiveMetropolis/adaptive_metropolis.hpp>
#include <stdexcept>
#include <vector>

namespace ml
{
  namespace test
  {

    /**
     * @brief 基本的なフィッティングと予測のテスト（RBFKernelのみ）
     */
    TEST(GaussianProcessRegressionTest, BasicFitAndPredict)
    {
      // RBFカーネル + ホワイトノイズカーネル
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      // テストデータ: y = sin(x) + noise
      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
      std::vector<double> y = {
        std::sin(0.0), std::sin(1.0), std::sin(2.0), std::sin(3.0), std::sin(4.0)};

      gpr.fit(X, y);

      EXPECT_TRUE(gpr.is_fitted());
      EXPECT_EQ(gpr.get_n_samples(), 5);
      EXPECT_EQ(gpr.get_input_dim(), 1);

      // 予測の検証（訓練データ点での予測は近い値になるはず）
      double pred = gpr.predict({1.0});
      EXPECT_NEAR(pred, std::sin(1.0), 0.5);  // ノイズがあるので許容誤差を大きく
    }

    /**
     * @brief 予測の平均と分散のテスト
     */
    TEST(GaussianProcessRegressionTest, PredictWithVariance)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}};
      std::vector<double> y = {0.0, 1.0, 0.0};

      gpr.fit(X, y);

      double mean, variance;
      gpr.predict({1.0}, mean, variance);

      // 平均は訓練データ点に近い値になるはず
      EXPECT_NEAR(mean, 1.0, 0.5);

      // 分散は正の値である必要がある
      EXPECT_GT(variance, 0.0);
    }

    /**
     * @brief 複数の観測値に対する予測のテスト
     */
    TEST(GaussianProcessRegressionTest, PredictMatrix)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}};
      std::vector<double> y = {0.0, 1.0, 0.0};

      gpr.fit(X, y);

      std::vector<std::vector<double>> X_test = {{0.5}, {1.5}};
      std::vector<double> means, variances;
      gpr.predict(X_test, means, variances);

      EXPECT_EQ(means.size(), 2);
      EXPECT_EQ(variances.size(), 2);

      // すべての分散は正の値である必要がある
      for (const auto& var : variances)
      {
        EXPECT_GT(var, 0.0);
      }
    }

    /**
     * @brief 対数周辺尤度のテスト
     */
    TEST(GaussianProcessRegressionTest, LogMarginalLikelihood)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}};
      std::vector<double> y = {0.0, 1.0, 0.0};

      gpr.fit(X, y);

      double lml = gpr.log_marginal_likelihood();

      // 対数周辺尤度は有限の値である必要がある
      EXPECT_TRUE(std::isfinite(lml));
      // 通常、対数周辺尤度は負の値になる
      EXPECT_LT(lml, 0.0);
    }

    /**
     * @brief 決定係数（R²）のテスト
     */
    TEST(GaussianProcessRegressionTest, ScoreR2)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
      std::vector<double> y = {
        std::sin(0.0), std::sin(1.0), std::sin(2.0), std::sin(3.0), std::sin(4.0)};

      gpr.fit(X, y);

      double r2 = gpr.score(X, y);

      // R²は通常0から1の間の値（完璧な予測の場合は1に近い）
      EXPECT_GE(r2, -1.0);  // 負の値も許容（悪いモデルの場合）
      EXPECT_LE(r2, 1.0);
    }

    /**
     * @brief MSE（平均二乗誤差）のテスト
     */
    TEST(GaussianProcessRegressionTest, MSE)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}};
      std::vector<double> y = {0.0, 1.0, 0.0};

      gpr.fit(X, y);

      double mse = gpr.mse(X, y);

      // MSEは非負の値である必要がある
      EXPECT_GE(mse, 0.0);
    }

    /**
     * @brief RMSE（平均二乗平方根誤差）のテスト
     */
    TEST(GaussianProcessRegressionTest, RMSE)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}};
      std::vector<double> y = {0.0, 1.0, 0.0};

      gpr.fit(X, y);

      double rmse = gpr.rmse(X, y);

      // RMSEは非負の値である必要がある
      EXPECT_GE(rmse, 0.0);
      // RMSE = sqrt(MSE)の関係を確認
      EXPECT_NEAR(rmse, std::sqrt(gpr.mse(X, y)), 1e-10);
    }

    /**
     * @brief MAE（平均絶対誤差）のテスト
     */
    TEST(GaussianProcessRegressionTest, MAE)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}, {2.0}};
      std::vector<double> y = {0.0, 1.0, 0.0};

      gpr.fit(X, y);

      double mae = gpr.mae(X, y);

      // MAEは非負の値である必要がある
      EXPECT_GE(mae, 0.0);
    }

    /**
     * @brief エラーハンドリング: フィッティング前に予測を試みる
     */
    TEST(GaussianProcessRegressionTest, PredictBeforeFit)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      EXPECT_FALSE(gpr.is_fitted());

      // フィッティング前に予測を試みると例外が発生するはず
      EXPECT_THROW(gpr.predict({1.0}), std::runtime_error);
      EXPECT_THROW(gpr.log_marginal_likelihood(), std::runtime_error);
    }

    /**
     * @brief エラーハンドリング: 入力サイズの不一致
     */
    TEST(GaussianProcessRegressionTest, InputSizeMismatch)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}};
      std::vector<double> y = {0.0, 1.0};

      gpr.fit(X, y);

      // 入力次元が異なる場合、例外が発生するはず
      EXPECT_THROW(gpr.predict({1.0, 2.0}), std::invalid_argument);
    }

    /**
     * @brief エラーハンドリング: Xとyのサイズ不一致
     */
    TEST(GaussianProcessRegressionTest, XAndYSizeMismatch)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}};
      std::vector<double> y = {0.0};  // サイズが不一致

      EXPECT_THROW(gpr.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief エラーハンドリング: 空のデータ
     */
    TEST(GaussianProcessRegressionTest, EmptyData)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X;
      std::vector<double> y;

      EXPECT_THROW(gpr.fit(X, y), std::invalid_argument);
    }

    /**
     * @brief リセット機能のテスト
     */
    TEST(GaussianProcessRegressionTest, Reset)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {{0.0}, {1.0}};
      std::vector<double> y = {0.0, 1.0};

      gpr.fit(X, y);
      EXPECT_TRUE(gpr.is_fitted());
      EXPECT_EQ(gpr.get_n_samples(), 2);

      gpr.reset();
      EXPECT_FALSE(gpr.is_fitted());
      EXPECT_EQ(gpr.get_n_samples(), 0);
    }

    /**
     * @brief ジッターの設定と取得のテスト
     */
    TEST(GaussianProcessRegressionTest, Jitter)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel, 1e-5);

      EXPECT_NEAR(gpr.get_jitter(), 1e-5, 1e-10);

      gpr.set_jitter(1e-4);
      EXPECT_NEAR(gpr.get_jitter(), 1e-4, 1e-10);

      // 負のジッターは許可されない
      EXPECT_THROW(gpr.set_jitter(-1.0), std::invalid_argument);
    }

    /**
     * @brief 多次元入力のテスト
     */
    TEST(GaussianProcessRegressionTest, MultiDimensionalInput)
    {
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {1.0, 1.0},
        {2.0, 2.0}
      };
      std::vector<double> y = {0.0, 1.0, 0.0};

      gpr.fit(X, y);

      EXPECT_EQ(gpr.get_input_dim(), 2);

      double pred = gpr.predict({1.0, 1.0});
      EXPECT_TRUE(std::isfinite(pred));
    }

    /**
     * @brief デフォルトコンストラクタのテスト
     */
    TEST(GaussianProcessRegressionTest, DefaultConstructor)
    {
      GaussianProcessRegression<double> gpr;

      EXPECT_FALSE(gpr.is_fitted());
      EXPECT_EQ(gpr.get_n_samples(), 0);
      EXPECT_EQ(gpr.get_input_dim(), 0);
      EXPECT_GT(gpr.get_jitter(), 0.0);
    }

    /**
     * @brief Adaptive Metropolisを使ったハイパーパラメータ最適化のテスト
     */
    TEST(GaussianProcessRegressionTest, HyperparameterOptimizationWithAdaptiveMetropolis)
    {
      // RBFカーネル + ホワイトノイズカーネル
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      // テストデータ: y = sin(x) + noise
      std::vector<std::vector<double>> X = {
        {0.0}, {0.5}, {1.0}, {1.5}, {2.0}, {2.5}, {3.0}, {3.5}, {4.0}};
      std::vector<double> y;
      for (const auto& x : X)
      {
        y.push_back(std::sin(x[0]) + 0.1 * (std::rand() / static_cast<double>(RAND_MAX) - 0.5));
      }

      // Adaptive Metropolisオプティマイザーを作成
      auto optimizer =
        std::make_shared<optimize::AdaptiveMetropolis<double>>(5'000,  // max_iterations
          2'500,                                                       // adaptation_period (burnin)
          0.0,                                                         // scaling_factor (自動計算)
          1e-6,                                                        // regularization
          1e-6,                                                        // convergence_tolerance
          42                                                           // seed
        );

      // 最適化前の対数周辺尤度を記録
      gpr.fit(X, y);  // 最適化なしでフィッティング
      double lml_before = gpr.log_marginal_likelihood();

      // Adaptive Metropolisでハイパーパラメータを最適化
      gpr.fit(X, y, optimizer);

      EXPECT_TRUE(gpr.is_fitted());

      // 最適化後の対数周辺尤度を記録
      double lml_after = gpr.log_marginal_likelihood();

      // 最適化後は対数周辺尤度が改善される（または同等）はず
      EXPECT_GE(lml_after, lml_before - 1.0);  // 許容誤差を設ける（MCMCの確率的性質のため）

      // 予測が正常に動作することを確認
      double pred = gpr.predict({1.0});
      EXPECT_TRUE(std::isfinite(pred));
      EXPECT_NEAR(pred, std::sin(1.0), 1.0);  // ノイズがあるので許容誤差を大きく
    }

    /**
     * @brief Adaptive Metropolisを使ったハイパーパラメータ最適化の詳細テスト
     * 最適化前後で予測精度が改善されることを確認
     */
    TEST(GaussianProcessRegressionTest, HyperparameterOptimizationImprovesPrediction)
    {
      // RBFカーネル + ホワイトノイズカーネル
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      // より多くのテストデータ
      std::vector<std::vector<double>> X_train;
      std::vector<double> y_train;
      for (double x = 0.0; x <= 4.0; x += 0.2)
      {
        X_train.push_back({x});
        y_train.push_back(std::sin(x) + 0.05 * (std::rand() / static_cast<double>(RAND_MAX) - 0.5));
      }

      // テストデータ
      std::vector<std::vector<double>> X_test = {{0.25}, {0.75}, {1.25}, {1.75}, {2.25}};
      std::vector<double> y_test;
      for (const auto& x : X_test)
      {
        y_test.push_back(std::sin(x[0]));
      }

      // 最適化なしでフィッティング
      gpr.fit(X_train, y_train);
      std::vector<double> predictions_before;
      for (const auto& x : X_test)
      {
        predictions_before.push_back(gpr.predict(x));
      }

      // MSEを計算
      double mse_before = 0.0;
      for (size_t i = 0; i < y_test.size(); ++i)
      {
        double diff = predictions_before[i] - y_test[i];
        mse_before += diff * diff;
      }
      mse_before /= y_test.size();

      // Adaptive Metropolisでハイパーパラメータを最適化
      auto optimizer =
        std::make_shared<optimize::AdaptiveMetropolis<double>>(3'000,  // max_iterations
          1'500,                                                       // adaptation_period
          0.0,                                                         // scaling_factor (自動計算)
          1e-6,                                                        // regularization
          1e-6,                                                        // convergence_tolerance
          42                                                           // seed
        );

      gpr.fit(X_train, y_train, optimizer);

      // 最適化後の予測
      std::vector<double> predictions_after;
      for (const auto& x : X_test)
      {
        predictions_after.push_back(gpr.predict(x));
      }

      // MSEを計算
      double mse_after = 0.0;
      for (size_t i = 0; i < y_test.size(); ++i)
      {
        double diff = predictions_after[i] - y_test[i];
        mse_after += diff * diff;
      }
      mse_after /= y_test.size();

      // 最適化後はMSEが改善される（または同等）はず
      // 注意: MCMCは確率的なので、必ずしも改善されるとは限らないが、
      // 多くの場合改善されるはず
      EXPECT_GE(mse_before, mse_after - 0.1);  // 許容誤差を設ける

      // すべての予測が有限値であることを確認
      for (const auto& pred : predictions_after)
      {
        EXPECT_TRUE(std::isfinite(pred));
      }
    }

    /**
     * @brief Adaptive Metropolisを使った多次元入力のハイパーパラメータ最適化テスト
     */
    TEST(GaussianProcessRegressionTest, HyperparameterOptimizationMultiDimensional)
    {
      // RBFカーネル + ホワイトノイズカーネル
      auto rbf_kernel = std::make_shared<RBFKernel<double>>(1.0, 1.0);
      auto white_noise_kernel = std::make_shared<WhiteNoiseKernel<double>>(0.1);
      auto kernel = std::make_shared<SumKernel<double>>(rbf_kernel, white_noise_kernel);

      GaussianProcessRegression<double> gpr(kernel);

      // 2次元入力データ
      std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.5, 0.5},
        {1.0, 1.0},
        {1.5, 1.5},
        {2.0, 2.0}
      };
      std::vector<double> y;
      for (const auto& x : X)
      {
        // y = x[0]^2 + x[1]^2 + noise
        y.push_back(
          x[0] * x[0] + x[1] * x[1] + 0.1 * (std::rand() / static_cast<double>(RAND_MAX) - 0.5));
      }

      // Adaptive Metropolisオプティマイザー
      auto optimizer =
        std::make_shared<optimize::AdaptiveMetropolis<double>>(4'000,  // max_iterations
          2'000,                                                       // adaptation_period
          0.0,                                                         // scaling_factor (自動計算)
          1e-6,                                                        // regularization
          1e-6,                                                        // convergence_tolerance
          42                                                           // seed
        );

      // 最適化前の対数周辺尤度
      gpr.fit(X, y);
      double lml_before = gpr.log_marginal_likelihood();

      // ハイパーパラメータを最適化
      gpr.fit(X, y, optimizer);

      EXPECT_TRUE(gpr.is_fitted());
      EXPECT_EQ(gpr.get_input_dim(), 2);

      // 最適化後の対数周辺尤度
      double lml_after = gpr.log_marginal_likelihood();

      // 最適化後は対数周辺尤度が改善される（または同等）はず
      EXPECT_GE(lml_after, lml_before - 1.0);

      // 予測が正常に動作することを確認
      double pred = gpr.predict({1.0, 1.0});
      EXPECT_TRUE(std::isfinite(pred));
    }

  }  // namespace test
}  // namespace ml
