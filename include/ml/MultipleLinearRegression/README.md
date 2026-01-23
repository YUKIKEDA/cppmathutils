# 重回帰 (Multiple Linear Regression)

## Overview

重回帰（Multiple Linear Regression）は、複数の説明変数（独立変数）$x_1, x_2, \ldots, x_p$ を用いて、目的変数（従属変数）$y$ を予測する線形回帰モデルです。単回帰の拡張として、複数の要因を同時に考慮することで、より現実的な予測モデルを構築できます。最小二乗法を用いて最適なパラメータを推定します。

## Mathematical Formulation

### Model

重回帰モデルは以下の線形方程式で表されます：

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$$

ここで：
- $y$: 目的変数（従属変数）
- $x_1, x_2, \ldots, x_p$: 説明変数（独立変数）
- $\beta_0$: 切片（y-intercept）
- $\beta_1, \beta_2, \ldots, \beta_p$: 回帰係数（regression coefficients）
- $\epsilon$: 誤差項（残差）

### Matrix Notation

重回帰モデルは行列形式でより簡潔に表現できます。$n$ 個の観測値と $p$ 個の説明変数がある場合：

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

ここで：

$\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}$: 目的変数のベクトル（$n \times 1$）

$\mathbf{X} = \begin{pmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix}$: 計画行列（design matrix, $n \times (p+1)$）

$\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_p \end{pmatrix}$: パラメータベクトル（$(p+1) \times 1$）

$\boldsymbol{\epsilon} = \begin{pmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{pmatrix}$: 誤差項のベクトル（$n \times 1$）

計画行列 $\mathbf{X}$ の最初の列はすべて1で、切片項 $\beta_0$ に対応します。

### Estimated Values

実際のデータから推定されたモデルは以下のように表されます：

$$\hat{y} = \hat{\beta_0} + \hat{\beta_1} x_1 + \hat{\beta_2} x_2 + \cdots + \hat{\beta_p} x_p$$

行列形式では：

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$$

ここで $\hat{\mathbf{y}}$ は予測値のベクトル、$\hat{\boldsymbol{\beta}}$ は推定されたパラメータベクトルです。

## Parameter Estimation by Least Squares

最小二乗法（Ordinary Least Squares, OLS）は、残差平方和（Sum of Squared Residuals, SSR）を最小化することでパラメータを推定します。

### Sum of Squared Residuals

真のモデル $y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \epsilon_i$ において、誤差項 $\epsilon_i$ は観測できないため、推定されたパラメータを用いて残差（residual）$e_i$ として推定します：

$$e_i = y_i - \hat{y}_i = y_i - \hat{\beta_0} - \hat{\beta_1} x_{i1} - \hat{\beta_2} x_{i2} - \cdots - \hat{\beta_p} x_{ip}$$

残差平方和（Sum of Squared Residuals, SSR）は以下のように定義されます：

$$SSR = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \hat{\mathbf{y}}\|^2 = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2$$

ここで $n$ はサンプル数、$\|\cdot\|$ はユークリッドノルムです。

### Parameter Estimation Formulas

残差平方和を最小化することで、以下の正規方程式（normal equations）が得られます：

$$\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$$

この連立方程式を解くことで、パラメータの推定値が得られます：

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

ここで $(\mathbf{X}^T\mathbf{X})^{-1}$ は $\mathbf{X}^T\mathbf{X}$ の逆行列です。

### Derivation Overview

パラメータベクトル $\boldsymbol{\beta}$ について偏微分を計算し、0とおくことで正規方程式が得られます：

$$\frac{\partial SSR}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0}$$

これを整理すると：

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

となり、上記の推定式が導出されます。

### Conditions for Unique Solution

$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ が一意に存在するためには、以下の条件が必要です：

1. **フルランク条件**: $\mathbf{X}$ がフルランク（$\text{rank}(\mathbf{X}) = p+1$）であること
2. **非特異性**: $\mathbf{X}^T\mathbf{X}$ が正則（可逆）であること

これらの条件が満たされない場合、多重共線性（multicollinearity）の問題が発生します。

## Evaluation Metrics

### Coefficient of Determination (R²)

決定係数は、モデルがデータの変動をどれだけ説明できるかを示します：

$$R^2 = 1 - \frac{SSR}{SST} = \frac{SSM}{SST}$$

ここで：
- $SSR = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$: 残差平方和
- $SST = \sum_{i=1}^{n}(y_i - \bar{y})^2$: 総平方和（Total Sum of Squares）
- $SSM = \sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2$: 回帰平方和（Model Sum of Squares）

$R^2$ の範囲は $[0, 1]$ で、1に近いほどモデルの説明力が高いことを示します。

### Adjusted R²

説明変数の数を考慮した調整済み決定係数は、以下のように定義されます：

$$R_{\text{adj}}^2 = 1 - \frac{SSR/(n-p-1)}{SST/(n-1)} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

調整済み $R^2$ は、説明変数を追加しても必ずしも増加しないため、モデル選択の指標として有用です。

### Mean Squared Error (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{SSR}{n}$$

### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE} = \sqrt{\frac{SSR}{n}}$$

### Mean Absolute Error (MAE)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

## Statistical Assumptions

重回帰モデルが有効であるためには、以下の仮定が満たされる必要があります：

1. **線形性**: 説明変数と目的変数の間に線形関係が存在する
2. **独立性**: 観測値は互いに独立である
3. **等分散性（均一分散性）**: 誤差項の分散は一定である（$\text{Var}(\epsilon_i) = \sigma^2$）
4. **正規性**: 誤差項は正規分布に従う（$\epsilon_i \sim N(0, \sigma^2)$）
5. **外生性**: 説明変数は誤差項と無相関である
6. **多重共線性の不存在**: 説明変数間に完全な線形関係が存在しない（$\text{rank}(\mathbf{X}) = p+1$）

これらの仮定が満たされない場合、推定結果の信頼性が低下する可能性があります。

## Statistical Inference for Parameters

### Variance-Covariance Matrix

パラメータ推定値の分散共分散行列は以下のように計算されます：

$$\text{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$$

ここで $\sigma^2$ は誤差分散の不偏推定量：

$$\hat{\sigma}^2 = \frac{SSR}{n-p-1} = MSE \cdot \frac{n}{n-p-1}$$

### Standard Error

パラメータ推定値 $\hat{\beta_j}$ の標準誤差は、分散共分散行列の対角要素の平方根として計算されます：

$$\text{SE}(\hat{\beta_j}) = \hat{\sigma}\sqrt{(\mathbf{X}^T\mathbf{X})^{-1}_{jj}}$$

ここで $(\mathbf{X}^T\mathbf{X})^{-1}_{jj}$ は $(\mathbf{X}^T\mathbf{X})^{-1}$ の $(j+1, j+1)$ 要素です（$j=0$ は切片に対応）。

### t-test

帰無仮説 $H_0: \beta_j = 0$ を検定する場合、以下のt統計量を使用します：

$$t = \frac{\hat{\beta_j}}{\text{SE}(\hat{\beta_j})}$$

この統計量は自由度 $n-p-1$ のt分布に従います。

### F-test for Overall Significance

すべての回帰係数が同時に0であるという帰無仮説 $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$ を検定する場合、以下のF統計量を使用します：

$$F = \frac{SSM/p}{SSR/(n-p-1)} = \frac{R^2/p}{(1-R^2)/(n-p-1)}$$

この統計量は自由度 $(p, n-p-1)$ のF分布に従います。

### Confidence Intervals

パラメータの $(1-\alpha)$ 信頼区間は以下のように計算されます：

$$\hat{\beta_j} \pm t_{\alpha/2, n-p-1} \cdot \text{SE}(\hat{\beta_j})$$

## Prediction Intervals

新しい観測値 $\mathbf{x}_0 = (1, x_{01}, x_{02}, \ldots, x_{0p})^T$ に対する予測値 $\hat{y}_0$ の信頼区間は：

$$\hat{y}_0 \pm t_{\alpha/2, n-p-1} \cdot \hat{\sigma} \sqrt{1 + \mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0}$$

## Computational Efficiency

### Matrix Operations

実装においては、以下の点に注意が必要です：

1. **逆行列の計算**: $(\mathbf{X}^T\mathbf{X})^{-1}$ の直接計算は数値的に不安定な場合があります。代わりに、QR分解やコレスキー分解を用いた解法が推奨されます。

2. **QR分解による解法**: $\mathbf{X} = \mathbf{Q}\mathbf{R}$ と分解すると（$\mathbf{Q}$ は直交行列、$\mathbf{R}$ は上三角行列）、正規方程式は以下のように簡略化されます：

   $$\mathbf{R}\hat{\boldsymbol{\beta}} = \mathbf{Q}^T\mathbf{y}$$

   この連立方程式は後退代入により効率的に解けます。

   **QR分解の詳細**:
   - **Householder変換**: 本実装では、Householder変換を用いたQR分解（HouseholderQR）を使用しています。これは数値的に安定で、計算効率も良好です。
   - **計算の流れ**:
     1. 計画行列 $\mathbf{X}$（$n \times (p+1)$）をQR分解: $\mathbf{X} = \mathbf{Q}\mathbf{R}$
     2. 正規方程式 $\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$ を変形:
        $$\mathbf{R}^T\mathbf{Q}^T\mathbf{Q}\mathbf{R}\hat{\boldsymbol{\beta}} = \mathbf{R}^T\mathbf{Q}^T\mathbf{y}$$
     3. $\mathbf{Q}$ が直交行列（$\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$）であることを利用:
        $$\mathbf{R}^T\mathbf{R}\hat{\boldsymbol{\beta}} = \mathbf{R}^T\mathbf{Q}^T\mathbf{y}$$
     4. $\mathbf{R}$ が上三角行列で正則と仮定すると、両辺に $(\mathbf{R}^T)^{-1}$ を左から掛けて:
        $$\mathbf{R}\hat{\boldsymbol{\beta}} = \mathbf{Q}^T\mathbf{y}$$
     5. 後退代入により $\hat{\boldsymbol{\beta}}$ を求める
   - **利点**:
     - 逆行列の計算が不要（数値的安定性が向上）
     - $\mathbf{X}^T\mathbf{X}$ を計算する必要がない（条件数が改善）
     - 多重共線性が存在する場合でも、より安定した結果が得られる

3. **数値的安定性**: 多重共線性が存在する場合、$\mathbf{X}^T\mathbf{X}$ が特異に近くなり、逆行列の計算が不安定になります。この場合、正則化手法（リッジ回帰、Lasso回帰など）の使用を検討します。

## Multicollinearity

### Definition

多重共線性（multicollinearity）は、説明変数間に強い線形関係が存在する場合に発生します。完全な多重共線性では、$\mathbf{X}^T\mathbf{X}$ が特異となり、パラメータの一意な推定が不可能になります。

### Detection

多重共線性の検出には以下の方法があります：

1. **分散拡大係数（VIF, Variance Inflation Factor）**:

   $$VIF_j = \frac{1}{1-R_j^2}$$

   ここで $R_j^2$ は、$x_j$ を目的変数として他の説明変数で回帰した場合の決定係数です。$VIF_j > 10$ の場合、多重共線性が疑われます。

2. **条件数（Condition Number）**: $\mathbf{X}^T\mathbf{X}$ の最大固有値と最小固有値の比。値が大きい場合（例：$> 30$）、多重共線性が疑われます。

### Solutions

多重共線性の問題を解決する方法：

1. **変数の削除**: 相関の高い変数の一方を削除
2. **主成分回帰（PCR）**: 主成分分析を用いて変数を変換
3. **リッジ回帰**: L2正則化を適用
4. **Lasso回帰**: L1正則化を適用

## Limitations and Constraints

1. **線形関係の仮定**: 非線形関係には対応できない
2. **外れ値への敏感性**: 外れ値の影響を受けやすい
3. **多重共線性**: 説明変数間の相関が高い場合、推定が不安定になる
4. **因果関係の誤解**: 相関関係と因果関係を混同しないよう注意が必要
5. **過学習**: 説明変数が多すぎる場合、過学習が発生する可能性がある
6. **サンプルサイズ**: パラメータ数に対してサンプル数が少ない場合、推定が不安定になる

## Application Examples

- 住宅価格の予測（面積、築年数、立地など）
- 売上の予測（広告費、価格、季節要因など）
- 医療診断（複数の検査値から疾患の可能性を予測）
- 気象予測（気温、湿度、気圧などから降水量を予測）
- 教育効果の分析（学習時間、宿題量、家庭環境などから成績を予測）

## References

- James, G., et al. (2021). *An Introduction to Statistical Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Montgomery, D. C., et al. (2021). *Introduction to Linear Regression Analysis*. Wiley.
- Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis*. Wiley.
