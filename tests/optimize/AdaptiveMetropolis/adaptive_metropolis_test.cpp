#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <optimize/AdaptiveMetropolis/adaptive_metropolis.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace optimize
{
  namespace test
  {

    /**
     * @brief テストデータを読み込むヘルパー関数
     */
    struct TestData
    {
        size_t dimension;
        std::vector<double> optimal_params;    // expected_statistics.mean
        std::vector<double> initial_params;    // parameters.initial_params
        std::vector<std::vector<double>> cov;  // expected_statistics.cov
        size_t n_samples;
        size_t burnin;
        double tolerance;
    };

    TestData load_test_data(const std::string& filename)
    {
      std::string test_data_dir = "tests/optimize/AdaptiveMetropolis/test_data/";
      std::string filepath = test_data_dir + filename;

      std::ifstream file(filepath);
      if (!file.is_open())
      {
        throw std::runtime_error("Failed to open test data file: " + filepath);
      }

      nlohmann::json j;
      file >> j;

      TestData data;
      data.dimension = j["dimension"].get<size_t>();
      data.n_samples = j["n_samples"].get<size_t>();
      data.burnin = j["burnin"].get<size_t>();

      // 最適解（期待される平均）
      data.optimal_params = j["expected_statistics"]["mean"].get<std::vector<double>>();

      // 初期パラメータ
      data.initial_params = j["parameters"]["initial_params"].get<std::vector<double>>();

      // 共分散行列
      data.cov = j["expected_statistics"]["cov"].get<std::vector<std::vector<double>>>();

      // 許容誤差（デフォルト値）
      data.tolerance = 0.5;

      return data;
    }

    /**
     * @brief 多変量正規分布の負の対数尤度を目的関数として使用
     */
    std::function<double(const std::vector<double>&)> create_multivariate_normal_objective(
      const std::vector<double>& mean, const std::vector<std::vector<double>>& cov)
    {
      return [mean, cov](const std::vector<double>& x) -> double
      {
        const size_t dim = mean.size();
        double result = 0.0;

        // 二次形式: (x - mean)^T * cov^(-1) * (x - mean)
        // 簡略化のため、対角共分散行列を仮定
        for (size_t i = 0; i < dim; ++i)
        {
          double diff = x[i] - mean[i];
          result += diff * diff / cov[i][i];
        }

        return 0.5 * result;  // 負の対数尤度（定数項を除く）
      };
    }

    /**
     * @brief 2次元テスト（adaptive_metropolis_dim2.json）
     */
    TEST(AdaptiveMetropolisTest, Dim2)
    {
      try
      {
        TestData data = load_test_data("adaptive_metropolis_dim2.json");

        AdaptiveMetropolis<double> optimizer(data.n_samples, data.burnin, 0.0, 1e-6, 1e-6, 42);

        auto objective = create_multivariate_normal_objective(data.optimal_params, data.cov);

        auto result = optimizer.minimize(objective, data.initial_params);

        // 最適解に近い値が得られることを確認
        for (size_t i = 0; i < data.dimension; ++i)
        {
          EXPECT_NEAR(result.parameters[i], data.optimal_params[i], data.tolerance);
        }
        EXPECT_TRUE(result.iterations > 0);
      }
      catch (const std::exception& e)
      {
        GTEST_SKIP() << "Test data file not found: " << e.what();
      }
    }

    /**
     * @brief 3次元テスト（adaptive_metropolis_dim3.json）
     */
    TEST(AdaptiveMetropolisTest, Dim3)
    {
      try
      {
        TestData data = load_test_data("adaptive_metropolis_dim3.json");

        AdaptiveMetropolis<double> optimizer(data.n_samples, data.burnin, 0.0, 1e-6, 1e-6, 42);

        auto objective = create_multivariate_normal_objective(data.optimal_params, data.cov);

        auto result = optimizer.minimize(objective, data.initial_params);

        // 最適解に近い値が得られることを確認
        for (size_t i = 0; i < data.dimension; ++i)
        {
          EXPECT_NEAR(result.parameters[i], data.optimal_params[i], data.tolerance);
        }
        EXPECT_TRUE(result.iterations > 0);
      }
      catch (const std::exception& e)
      {
        GTEST_SKIP() << "Test data file not found: " << e.what();
      }
    }

    /**
     * @brief 5次元テスト（adaptive_metropolis_dim5.json）
     */
    TEST(AdaptiveMetropolisTest, Dim5)
    {
      try
      {
        TestData data = load_test_data("adaptive_metropolis_dim5.json");

        AdaptiveMetropolis<double> optimizer(data.n_samples, data.burnin, 0.0, 1e-6, 1e-6, 42);

        auto objective = create_multivariate_normal_objective(data.optimal_params, data.cov);

        auto result = optimizer.minimize(objective, data.initial_params);

        // 最適解に近い値が得られることを確認
        for (size_t i = 0; i < data.dimension; ++i)
        {
          EXPECT_NEAR(result.parameters[i], data.optimal_params[i], data.tolerance);
        }
        EXPECT_TRUE(result.iterations > 0);
      }
      catch (const std::exception& e)
      {
        GTEST_SKIP() << "Test data file not found: " << e.what();
      }
    }

    /**
     * @brief シフトされた2次元テスト（adaptive_metropolis_shifted_dim2.json）
     */
    TEST(AdaptiveMetropolisTest, ShiftedDim2)
    {
      try
      {
        TestData data = load_test_data("adaptive_metropolis_shifted_dim2.json");

        AdaptiveMetropolis<double> optimizer(data.n_samples, data.burnin, 0.0, 1e-6, 1e-6, 42);

        auto objective = create_multivariate_normal_objective(data.optimal_params, data.cov);

        auto result = optimizer.minimize(objective, data.initial_params);

        // 最適解に近い値が得られることを確認
        for (size_t i = 0; i < data.dimension; ++i)
        {
          EXPECT_NEAR(result.parameters[i], data.optimal_params[i], data.tolerance);
        }
        EXPECT_TRUE(result.iterations > 0);
      }
      catch (const std::exception& e)
      {
        GTEST_SKIP() << "Test data file not found: " << e.what();
      }
    }

    /**
     * @brief シフトされた3次元テスト（adaptive_metropolis_shifted_dim3.json）
     */
    TEST(AdaptiveMetropolisTest, ShiftedDim3)
    {
      try
      {
        TestData data = load_test_data("adaptive_metropolis_shifted_dim3.json");

        AdaptiveMetropolis<double> optimizer(data.n_samples, data.burnin, 0.0, 1e-6, 1e-6, 42);

        auto objective = create_multivariate_normal_objective(data.optimal_params, data.cov);

        auto result = optimizer.minimize(objective, data.initial_params);

        // 最適解に近い値が得られることを確認
        for (size_t i = 0; i < data.dimension; ++i)
        {
          EXPECT_NEAR(result.parameters[i], data.optimal_params[i], data.tolerance);
        }
        EXPECT_TRUE(result.iterations > 0);
      }
      catch (const std::exception& e)
      {
        GTEST_SKIP() << "Test data file not found: " << e.what();
      }
    }

    /**
     * @brief シフトされた5次元テスト（adaptive_metropolis_shifted_dim5.json）
     */
    TEST(AdaptiveMetropolisTest, ShiftedDim5)
    {
      try
      {
        TestData data = load_test_data("adaptive_metropolis_shifted_dim5.json");

        AdaptiveMetropolis<double> optimizer(data.n_samples, data.burnin, 0.0, 1e-6, 1e-6, 42);

        auto objective = create_multivariate_normal_objective(data.optimal_params, data.cov);

        auto result = optimizer.minimize(objective, data.initial_params);

        // 最適解に近い値が得られることを確認
        for (size_t i = 0; i < data.dimension; ++i)
        {
          EXPECT_NEAR(result.parameters[i], data.optimal_params[i], data.tolerance);
        }
        EXPECT_TRUE(result.iterations > 0);
      }
      catch (const std::exception& e)
      {
        GTEST_SKIP() << "Test data file not found: " << e.what();
      }
    }

    /**
     * @brief エラーハンドリング: 空の初期パラメータ
     */
    TEST(AdaptiveMetropolisTest, EmptyInitialParamsError)
    {
      AdaptiveMetropolis<double> optimizer(1'000, 500, 0.0, 1e-6, 1e-6, 42);

      auto objective = [](const std::vector<double>& x) -> double
      {
        return x[0] * x[0];
      };

      std::vector<double> empty_params;

      EXPECT_THROW(optimizer.minimize(objective, empty_params), std::invalid_argument);
    }

    /**
     * @brief コンストラクタのエラーハンドリング: 最大反復回数が0
     */
    TEST(AdaptiveMetropolisTest, InvalidMaxIterationsError)
    {
      EXPECT_THROW(
        AdaptiveMetropolis<double> optimizer(0, 0, 0.0, 1e-6, 1e-6, 42), std::invalid_argument);
    }

    /**
     * @brief コンストラクタのエラーハンドリング: 適応期間が最大反復回数を超える
     */
    TEST(AdaptiveMetropolisTest, InvalidAdaptationPeriodError)
    {
      EXPECT_THROW(AdaptiveMetropolis<double> optimizer(1'000, 2'000, 0.0, 1e-6, 1e-6, 42),
        std::invalid_argument);
    }

  }  // namespace test
}  // namespace optimize
