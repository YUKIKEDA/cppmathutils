# 単回帰 (Simple Linear Regression)

## Overview

単回帰（Simple Linear Regression）は、1つの説明変数（独立変数）$x$ を用いて、目的変数（従属変数）$y$ を予測する線形回帰モデルです。2つの変数間の線形関係をモデル化し、最小二乗法を用いて最適なパラメータを推定します。

## Mathematical Formulation

### Model

単回帰モデルは以下の線形方程式で表されます：

$$y = w_0 + w_1 x + \epsilon$$

ここで：
- $y$: 目的変数（従属変数）
- $x$: 説明変数（独立変数）
- $w_0$: 切片（y-intercept）
- $w_1$: 傾き（slope）
- $\epsilon$: 誤差項（残差）

### Estimated Values

実際のデータから推定されたモデルは以下のように表されます：

$$\hat{y} = \hat{w_0} + \hat{w_1} x$$

ここで $\hat{y}$ は予測値、$\hat{w_0}$ と $\hat{w_1}$ は推定されたパラメータです。

## Parameter Estimation by Least Squares

最小二乗法（Ordinary Least Squares, OLS）は、残差平方和（Sum of Squared Residuals, SSR）を最小化することでパラメータを推定します。

### Sum of Squared Residuals

真のモデル $y_i = w_0 + w_1 x_i + \epsilon_i$ において、誤差項 $\epsilon_i$ は観測できないため、推定されたパラメータを用いて残差（residual）$e_i$ として推定します：

$$e_i = y_i - \hat{y}_i = y_i - \hat{w_0} - \hat{w_1} x_i$$

この残差は真の誤差項 $\epsilon_i$ の推定値とみなすことができます。残差平方和（Sum of Squared Residuals, SSR）は以下のように定義されます：

$$SSR = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - \hat{w_0} - \hat{w_1} x_i)^2$$

ここで $n$ はサンプル数、$y_i$ と $x_i$ は $i$ 番目の観測値です。

### Parameter Estimation Formulas

残差平方和を最小化することで、以下の正規方程式が得られます：

#### Slope Estimation

$$\hat{w_1} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{Cov(x, y)}{Var(x)} = \frac{S_{xy}}{S_{xx}}$$

ここで：
- $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$: $x$ の標本平均
- $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$: $y$ の標本平均
- $S_{xy} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$: 共分散の分子
- $S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2$: $x$ の偏差平方和

#### Intercept Estimation

$$\hat{w_0} = \bar{y} - \hat{w_1} \bar{x}$$

### Derivation Overview

パラメータ $w_0$ と $w_1$ について偏微分を計算し、0とおくことで正規方程式が得られます：

$$\frac{\partial SSR}{\partial w_0} = -2\sum_{i=1}^{n}(y_i - w_0 - w_1 x_i) = 0$$

$$\frac{\partial SSR}{\partial w_1} = -2\sum_{i=1}^{n}x_i(y_i - w_0 - w_1 x_i) = 0$$

これらの連立方程式を解くことで、上記の推定式が導出されます。

## Evaluation Metrics

### Coefficient of Determination (R²)

決定係数は、モデルがデータの変動をどれだけ説明できるかを示します：

$$R^2 = 1 - \frac{SSR}{SST} = \frac{SSM}{SST}$$

ここで：
- $SSR = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$: 残差平方和
- $SST = \sum_{i=1}^{n}(y_i - \bar{y})^2$: 総平方和（Total Sum of Squares）
- $SSM = \sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2$: 回帰平方和（Model Sum of Squares）

$R^2$ の範囲は $[0, 1]$ で、1に近いほどモデルの説明力が高いことを示します。

### Relationship with Correlation Coefficient

単回帰において、決定係数は相関係数の二乗と等しくなります：

$$R^2 = r_{xy}^2$$

ここで $r_{xy}$ は $x$ と $y$ のピアソン相関係数です：

$$r_{xy} = \frac{Cov(x, y)}{\sqrt{Var(x)Var(y)}} = \frac{S_{xy}}{\sqrt{S_{xx} S_{yy}}}$$

### Mean Squared Error (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{SSR}{n}$$

### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE} = \sqrt{\frac{SSR}{n}}$$

### Mean Absolute Error (MAE)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

## Statistical Assumptions

単回帰モデルが有効であるためには、以下の仮定が満たされる必要があります：

1. **線形性**: $x$ と $y$ の間に線形関係が存在する
2. **独立性**: 観測値は互いに独立である
3. **等分散性（均一分散性）**: 誤差項の分散は一定である（$\text{Var}(\epsilon_i) = \sigma^2$）
4. **正規性**: 誤差項は正規分布に従う（$\epsilon_i \sim N(0, \sigma^2)$）
5. **外生性**: 説明変数 $x$ は誤差項と無相関である

これらの仮定が満たされない場合、推定結果の信頼性が低下する可能性があります。

## Statistical Inference for Parameters

### Standard Error

パラメータ推定値の標準誤差は以下のように計算されます：

$$\text{SE}(\hat{w_1}) = \frac{\sigma}{\sqrt{S_{xx}}}$$

$$\text{SE}(\hat{w_0}) = \sigma\sqrt{\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}}$$

ここで $\sigma^2$ は誤差分散の不偏推定量：

$$\hat{\sigma}^2 = \frac{SSR}{n-2} = MSE \cdot \frac{n}{n-2}$$

### t-test

帰無仮説 $H_0: w_1 = 0$ を検定する場合、以下のt統計量を使用します：

$$t = \frac{\hat{w_1}}{\text{SE}(\hat{w_1})}$$

この統計量は自由度 $n-2$ のt分布に従います。

### Confidence Intervals

パラメータの $(1-\alpha)$ 信頼区間は以下のように計算されます：

$$\hat{w_1} \pm t_{\alpha/2, n-2} \cdot \text{SE}(\hat{w_1})$$

$$\hat{w_0} \pm t_{\alpha/2, n-2} \cdot \text{SE}(\hat{w_0})$$

## Prediction Intervals

新しい観測値 $x_0$ に対する予測値 $\hat{y}_0$ の信頼区間は：

$$\hat{y}_0 \pm t_{\alpha/2, n-2} \cdot \hat{\sigma} \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}}$$

## Computational Efficiency

### Formula Transformations

実装においては、以下の変形式を使用することで計算効率を向上させることができます：

$$\hat{w_1} = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{n\sum x_i^2 - (\sum x_i)^2}$$

$$\hat{w_0} = \frac{\sum y_i - \hat{w_1}\sum x_i}{n}$$

ただし、数値的安定性を考慮すると、平均からの偏差を用いた元の式の方が推奨されます。

## Limitations and Constraints

1. **線形関係の仮定**: 非線形関係には対応できない
2. **外れ値への敏感性**: 外れ値の影響を受けやすい
3. **多重共線性**: 単回帰では該当しないが、重回帰への拡張時に問題となる
4. **因果関係の誤解**: 相関関係と因果関係を混同しないよう注意が必要

## Application Examples

- 売上と広告費の関係の分析
- 気温とアイスクリームの売上の関係
- 学習時間とテストスコアの関係
- 身長と体重の関係

## References

- James, G., et al. (2021). *An Introduction to Statistical Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Montgomery, D. C., et al. (2021). *Introduction to Linear Regression Analysis*. Wiley.
