#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace iocsv
{

  /**
   * @brief ベクトルデータをCSV形式でファイルに書き出す
   * @param data 書き出すデータ（行優先順序で格納された2次元配列）
   * @param rows 行数
   * @param cols 列数
   * @param filename 出力ファイル名
   * @param precision 浮動小数点数の精度（デフォルト: 6）
   * @return 成功した場合true、失敗した場合false
   */
  inline bool write_csv(
      const std::vector<float>& data,
      size_t rows,
      size_t cols,
      const std::string& filename,
      int precision = 6)
  {
    if (data.size() != rows * cols)
    {
      return false;
    }

    std::ofstream file(filename);
    if (!file.is_open())
    {
      return false;
    }

    file << std::fixed << std::setprecision(precision);

    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        file << data[i * cols + j];
        if (j < cols - 1)
        {
          file << ",";
        }
      }
      file << "\n";
    }

    file.close();
    return true;
  }

  /**
   * @brief ベクトルデータをCSV形式でファイルに書き出す（1次元配列用）
   * @param data 書き出すデータ
   * @param filename 出力ファイル名
   * @param precision 浮動小数点数の精度（デフォルト: 6）
   * @return 成功した場合true、失敗した場合false
   */
  inline bool write_csv_1d(
      const std::vector<float>& data, const std::string& filename, int precision = 6)
  {
    std::ofstream file(filename);
    if (!file.is_open())
    {
      return false;
    }

    file << std::fixed << std::setprecision(precision);

    for (size_t i = 0; i < data.size(); ++i)
    {
      file << data[i];
      if (i < data.size() - 1)
      {
        file << ",";
      }
    }
    file << "\n";

    file.close();
    return true;
  }

}  // namespace iocsv
