# 線形回帰 (Linear Regression)

## Overview

線形回帰（Linear Regression）は、パラメータ（重み）に対して線形なモデルを用いて、目的変数を予測する統計的手法です。ここで「線形」とは、説明変数が線形であることを意味するのではなく、**パラメータ（重み）に対して線形**であることを意味します。

線形回帰モデルは、基底関数（basis functions）を用いて以下のように一般化できます：

$$\hat{y} = w_0 + w_1 \phi_1(\mathbf{x}) + w_2 \phi_2(\mathbf{x}) + \cdots + w_p \phi_p(\mathbf{x})$$

ここで $\phi_i(\mathbf{x})$ は任意の基底関数であり、$x$, $x^2$, $\sin(x)$, $\exp(x)$, $\log(x)$ など、非線形関数でも構いません。重要なのは、重み $w_i$ に対して線形であることです。

単回帰（Simple Linear Regression）や重回帰（Multiple Linear Regression）は、基底関数が $\phi_i(\mathbf{x}) = x_i$ という特殊ケースとして位置づけられます。

## Mathematical Formulation

### General Model

線形回帰モデルは以下のように表されます：

$$\hat{y} = w_0 + w_1 \phi_1(\mathbf{x}) + w_2 \phi_2(\mathbf{x}) + \cdots + w_p \phi_p(\mathbf{x})$$

または、より一般的に：

$$\hat{y} = \sum_{j=0}^{p} w_j \phi_j(\mathbf{x})$$

ここで $\phi_0(\mathbf{x}) = 1$ と定義し、$w_0$ は切片項に対応します。

ここで：
- $\hat{y}$: 予測値（目的変数）
- $\mathbf{x}$: 入力ベクトル（元の説明変数）
- $\phi_1(\mathbf{x}), \phi_2(\mathbf{x}), \ldots, \phi_p(\mathbf{x})$: 基底関数（basis functions）
- $w_0, w_1, w_2, \ldots, w_p$: パラメータ（重み、回帰係数）
- $p$: 基底関数の数

### Basis Functions

基底関数 $\phi_i(\mathbf{x})$ は任意の関数を取ることができます。以下に代表的な例を示します：

#### 多項式基底（Polynomial Basis）

$$\phi_1(x) = x, \quad \phi_2(x) = x^2, \quad \phi_3(x) = x^3, \ldots$$

これにより多項式回帰が実現されます：
$$\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \cdots$$

#### 三角関数基底（Trigonometric Basis）

$$\phi_1(x) = \sin(x), \quad \phi_2(x) = \cos(x), \quad \phi_3(x) = \sin(2x), \ldots$$

#### 指数・対数基底（Exponential/Logarithmic Basis）

$$\phi_1(x) = e^x, \quad \phi_2(x) = e^{-x}, \quad \phi_3(x) = \log(x), \ldots$$

#### ガウス基底（Gaussian Basis）

$$\phi_i(x) = \exp\left(-\frac{(x - \mu_i)^2}{2\sigma^2}\right)$$

#### 恒等関数（Identity Function）: 単回帰・重回帰

$$\phi_1(\mathbf{x}) = x_1, \quad \phi_2(\mathbf{x}) = x_2, \quad \ldots, \quad \phi_p(\mathbf{x}) = x_p$$

これが通常の単回帰・重回帰に対応します。

### Matrix Notation

$n$ 個の観測値がある場合、線形回帰モデルは行列形式で表現できます：

$$\hat{\mathbf{y}} = \mathbf{\Phi}\mathbf{w}$$

ここで：

$\hat{\mathbf{y}} = \begin{pmatrix} \hat{y}_1 \\ \hat{y}_2 \\ \vdots \\ \hat{y}_n \end{pmatrix}$: 予測値のベクトル（$n \times 1$）

$\mathbf{\Phi} = \begin{pmatrix} \phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \phi_2(\mathbf{x}_1) & \cdots & \phi_p(\mathbf{x}_1) \\ \phi_0(\mathbf{x}_2) & \phi_1(\mathbf{x}_2) & \phi_2(\mathbf{x}_2) & \cdots & \phi_p(\mathbf{x}_2) \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \phi_0(\mathbf{x}_n) & \phi_1(\mathbf{x}_n) & \phi_2(\mathbf{x}_n) & \cdots & \phi_p(\mathbf{x}_n) \end{pmatrix}$: 計画行列（design matrix, $n \times (p+1)$）

$\mathbf{w} = \begin{pmatrix} w_0 \\ w_1 \\ w_2 \\ \vdots \\ w_p \end{pmatrix}$: パラメータベクトル（$(p+1) \times 1$）

ここで $\phi_0(\mathbf{x}) = 1$ と定義し、計画行列 $\mathbf{\Phi}$ の最初の列はすべて1で、切片項 $w_0$ に対応します。

### Special Cases

#### Simple Linear Regression

基底関数が $\phi_1(x) = x$ の場合：

$$\hat{y} = w_0 + w_1 x$$

これは通常の単回帰モデルです。

#### Multiple Linear Regression

基底関数が $\phi_i(\mathbf{x}) = x_i$ ($i = 1, 2, \ldots, p$) の場合：

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_p x_p$$

これは通常の重回帰モデルです。

#### Polynomial Regression

基底関数が $\phi_i(x) = x^i$ の場合：

$$\hat{y} = w_0 + w_1 x + w_2 x^2 + \cdots + w_p x^p$$

これは多項式回帰モデルです。説明変数 $x$ は非線形ですが、重み $w_i$ に対して線形であるため、線形回帰の範疇です。

## Parameter Estimation by Least Squares

最小二乗法（Ordinary Least Squares, OLS）は、残差平方和（Sum of Squared Residuals, SSR）を最小化することでパラメータを推定します。

### Sum of Squared Residuals

真のモデル $y_i = \sum_{j=0}^{p} w_j \phi_j(\mathbf{x}_i) + \epsilon_i$ において、誤差項 $\epsilon_i$ は観測できないため、推定されたパラメータを用いて残差（residual）$e_i$ として推定します：

$$e_i = y_i - \hat{y}_i = y_i - \sum_{j=0}^{p} w_j \phi_j(\mathbf{x}_i)$$

残差平方和（Sum of Squared Residuals, SSR）は以下のように定義されます：

$$SSR = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \hat{\mathbf{y}}\|^2 = \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2$$

ここで $n$ はサンプル数、$\|\cdot\|$ はユークリッドノルムです。

### Parameter Estimation Formula

残差平方和を最小化することで、以下の正規方程式（normal equations）が得られます：

$$\mathbf{\Phi}^T\mathbf{\Phi}\hat{\mathbf{w}} = \mathbf{\Phi}^T\mathbf{y}$$

この連立方程式を解くことで、パラメータの推定値が得られます：

$$\hat{\mathbf{w}} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{y}$$

ここで $(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}$ は $\mathbf{\Phi}^T\mathbf{\Phi}$ の逆行列です。

### Derivation

パラメータベクトル $\mathbf{w}$ について偏微分を計算し、0とおくことで正規方程式が得られます：

$$\frac{\partial SSR}{\partial \mathbf{w}} = -2\mathbf{\Phi}^T(\mathbf{y} - \mathbf{\Phi}\mathbf{w}) = \mathbf{0}$$

これを整理すると：

$$\mathbf{\Phi}^T\mathbf{\Phi}\mathbf{w} = \mathbf{\Phi}^T\mathbf{y}$$

となり、上記の推定式が導出されます。

### Conditions for Unique Solution

$\hat{\mathbf{w}} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{y}$ が一意に存在するためには、以下の条件が必要です：

1. **フルランク条件**: $\mathbf{\Phi}$ がフルランク（$\text{rank}(\mathbf{\Phi}) = p+1$）であること
2. **非特異性**: $\mathbf{\Phi}^T\mathbf{\Phi}$ が正則（可逆）であること
3. **サンプルサイズ**: $n \geq p+1$ であること（パラメータ数以上）

これらの条件が満たされない場合、多重共線性（multicollinearity）の問題が発生するか、解が一意に定まりません。

### Geometric Interpretation

最小二乗法は、幾何学的には目的変数ベクトル $\mathbf{y}$ を計画行列 $\mathbf{\Phi}$ の列空間（column space）に射影する操作として解釈できます。推定されたパラメータ $\hat{\mathbf{w}}$ は、$\mathbf{y}$ を $\mathbf{\Phi}$ の列空間に射影したベクトル $\hat{\mathbf{y}} = \mathbf{\Phi}\hat{\mathbf{w}}$ を求めるための係数です。

## Evaluation Metrics

### Coefficient of Determination (R²)

決定係数は、モデルがデータの変動をどれだけ説明できるかを示します：

$$R^2 = 1 - \frac{SSR}{SST} = \frac{SSM}{SST}$$

ここで：
- $SSR = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$: 残差平方和（Sum of Squared Residuals）
- $SST = \sum_{i=1}^{n}(y_i - \bar{y})^2$: 総平方和（Total Sum of Squares）
- $SSM = \sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2$: 回帰平方和（Model Sum of Squares）

$R^2$ の範囲は $[0, 1]$ で、1に近いほどモデルの説明力が高いことを示します。

### Adjusted R²

基底関数の数を考慮した調整済み決定係数は、以下のように定義されます：

$$R_{\text{adj}}^2 = 1 - \frac{SSR/(n-p-1)}{SST/(n-1)} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

調整済み $R^2$ は、基底関数を追加しても必ずしも増加しないため、モデル選択の指標として有用です。

### Mean Squared Error (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{SSR}{n}$$

### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE} = \sqrt{\frac{SSR}{n}}$$

RMSEは目的変数と同じ単位を持つため、解釈が容易です。

### Mean Absolute Error (MAE)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

MAEは外れ値に対してより頑健（robust）です。

## Statistical Assumptions

線形回帰モデルが有効であるためには、以下の仮定が満たされる必要があります。これらの仮定は、ガウス・マルコフの定理（Gauss-Markov Theorem）により、最小二乗推定量が最良線形不偏推定量（BLUE, Best Linear Unbiased Estimator）となるための条件です。

1. **線形性**: パラメータ（重み）に対して線形である
   - モデル形式: $E[y|\mathbf{x}] = \sum_{j=0}^{p} w_j \phi_j(\mathbf{x})$
   - 注意: 基底関数 $\phi_j(\mathbf{x})$ 自体は非線形でも構いません

2. **独立性**: 観測値は互いに独立である
   - $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ for $i \neq j$

3. **等分散性（均一分散性）**: 誤差項の分散は一定である
   - $\text{Var}(\epsilon_i) = \sigma^2$ for all $i$（ホモスケダスティック）

4. **正規性**: 誤差項は正規分布に従う（推論のため）
   - $\epsilon_i \sim N(0, \sigma^2)$

5. **外生性**: 基底関数の値は誤差項と無相関である
   - $E[\epsilon_i|\phi_j(\mathbf{x}_i)] = 0$ または $\text{Cov}(\phi_j(\mathbf{x}_i), \epsilon_i) = 0$

6. **多重共線性の不存在**: 基底関数間に完全な線形関係が存在しない
   - $\text{rank}(\mathbf{\Phi}) = p+1$

これらの仮定が満たされない場合、推定結果の信頼性が低下する可能性があります。

## Statistical Inference for Parameters

### Variance-Covariance Matrix

パラメータ推定値の分散共分散行列は以下のように計算されます：

$$\text{Cov}(\hat{\mathbf{w}}) = \sigma^2(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}$$

ここで $\sigma^2$ は誤差分散の不偏推定量：

$$\hat{\sigma}^2 = \frac{SSR}{n-p-1} = MSE \cdot \frac{n}{n-p-1}$$

### Standard Error

パラメータ推定値 $\hat{w_j}$ の標準誤差は、分散共分散行列の対角要素の平方根として計算されます：

$$\text{SE}(\hat{w_j}) = \hat{\sigma}\sqrt{(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}_{jj}}$$

ここで $(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}_{jj}$ は $(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}$ の $(j+1, j+1)$ 要素です（$j=0$ は切片に対応）。

### t-test

帰無仮説 $H_0: w_j = 0$ を検定する場合、以下のt統計量を使用します：

$$t = \frac{\hat{w_j}}{\text{SE}(\hat{w_j})}$$

この統計量は自由度 $n-p-1$ のt分布に従います。

### F-test for Overall Significance

すべての回帰係数（切片を除く）が同時に0であるという帰無仮説 $H_0: w_1 = w_2 = \cdots = w_p = 0$ を検定する場合、以下のF統計量を使用します：

$$F = \frac{SSM/p}{SSR/(n-p-1)} = \frac{R^2/p}{(1-R^2)/(n-p-1)}$$

この統計量は自由度 $(p, n-p-1)$ のF分布に従います。

### Confidence Intervals

パラメータの $(1-\alpha)$ 信頼区間は以下のように計算されます：

$$\hat{w_j} \pm t_{\alpha/2, n-p-1} \cdot \text{SE}(\hat{w_j})$$

## Prediction Intervals

### Point Prediction

新しい観測値 $\mathbf{x}_0$ に対する点予測は：

$$\hat{y}_0 = \sum_{j=0}^{p} \hat{w_j} \phi_j(\mathbf{x}_0) = \boldsymbol{\phi}_0^T\hat{\mathbf{w}}$$

ここで $\boldsymbol{\phi}_0 = (\phi_0(\mathbf{x}_0), \phi_1(\mathbf{x}_0), \phi_2(\mathbf{x}_0), \ldots, \phi_p(\mathbf{x}_0))^T$ です。$\phi_0(\mathbf{x}) = 1$ であるため、$\boldsymbol{\phi}_0 = (1, \phi_1(\mathbf{x}_0), \phi_2(\mathbf{x}_0), \ldots, \phi_p(\mathbf{x}_0))^T$ となります。

### Prediction Interval

新しい観測値 $\mathbf{x}_0$ に対する予測値 $\hat{y}_0$ の $(1-\alpha)$ 予測区間は：

$$\hat{y}_0 \pm t_{\alpha/2, n-p-1} \cdot \hat{\sigma} \sqrt{1 + \boldsymbol{\phi}_0^T(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\boldsymbol{\phi}_0}$$

## Computational Efficiency

### Matrix Operations

実装においては、以下の点に注意が必要です：

1. **逆行列の計算**: $(\mathbf{\Phi}^T\mathbf{\Phi})^{-1}$ の直接計算は数値的に不安定な場合があります。代わりに、QR分解やコレスキー分解を用いた解法が推奨されます。

2. **QR分解による解法**: $\mathbf{\Phi} = \mathbf{Q}\mathbf{R}$ と分解すると（$\mathbf{Q}$ は直交行列、$\mathbf{R}$ は上三角行列）、正規方程式は以下のように簡略化されます：

   $$\mathbf{R}\hat{\mathbf{w}} = \mathbf{Q}^T\mathbf{y}$$

   この連立方程式は後退代入により効率的に解けます。

   **QR分解の詳細**:
   - **Householder変換**: Householder変換を用いたQR分解（HouseholderQR）を使用することで、数値的に安定で、計算効率も良好です。
   - **計算の流れ**:
     1. 計画行列 $\mathbf{\Phi}$（$n \times (p+1)$）をQR分解: $\mathbf{\Phi} = \mathbf{Q}\mathbf{R}$
     2. 正規方程式 $\mathbf{\Phi}^T\mathbf{\Phi}\hat{\mathbf{w}} = \mathbf{\Phi}^T\mathbf{y}$ を変形:
        $$\mathbf{R}^T\mathbf{Q}^T\mathbf{Q}\mathbf{R}\hat{\mathbf{w}} = \mathbf{R}^T\mathbf{Q}^T\mathbf{y}$$
     3. $\mathbf{Q}$ が直交行列（$\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$）であることを利用:
        $$\mathbf{R}^T\mathbf{R}\hat{\mathbf{w}} = \mathbf{R}^T\mathbf{Q}^T\mathbf{y}$$
     4. $\mathbf{R}$ が上三角行列で正則と仮定すると、両辺に $(\mathbf{R}^T)^{-1}$ を左から掛けて:
        $$\mathbf{R}\hat{\mathbf{w}} = \mathbf{Q}^T\mathbf{y}$$
     5. 後退代入により $\hat{\mathbf{w}}$ を求める
   - **利点**:
     - 逆行列の計算が不要（数値的安定性が向上）
     - $\mathbf{\Phi}^T\mathbf{\Phi}$ を計算する必要がない（条件数が改善）
     - 多重共線性が存在する場合でも、より安定した結果が得られる

3. **数値的安定性**: 多重共線性が存在する場合、$\mathbf{\Phi}^T\mathbf{\Phi}$ が特異に近くなり、逆行列の計算が不安定になります。この場合、正則化手法（リッジ回帰、Lasso回帰など）の使用を検討します。

4. **計算量**: 
   - 直接的な逆行列計算: $O((p+1)^3)$
   - QR分解による解法: $O(n(p+1)^2)$

## Basis Function Selection

基底関数の選択は、線形回帰モデルの性能に大きく影響します。以下に代表的な選択方法を示します：

### Polynomial Basis

多項式基底は、非線形関係をモデル化する際によく用いられます。ただし、次数が高くなると過学習のリスクが増加します。

### Radial Basis Functions (RBF)

ガウス基底関数などの放射基底関数は、局所的な非線形関係をモデル化する際に有効です。

### Feature Engineering

ドメイン知識に基づいて、適切な基底関数を設計することが重要です。例えば：
- 周期性がある場合: $\sin(x)$, $\cos(x)$
- 指数関数的な関係: $\exp(x)$, $\log(x)$
- 交互作用項: $x_1 x_2$, $x_1^2 x_2$

## Multicollinearity

### Definition

多重共線性（multicollinearity）は、基底関数間に強い線形関係が存在する場合に発生します。完全な多重共線性では、$\mathbf{\Phi}^T\mathbf{\Phi}$ が特異となり、パラメータの一意な推定が不可能になります。不完全な多重共線性では、推定は可能ですが、標準誤差が大きくなり、推定値が不安定になります。

### Detection

多重共線性の検出には以下の方法があります：

1. **分散拡大係数（VIF, Variance Inflation Factor）**:

   $$VIF_j = \frac{1}{1-R_j^2}$$

   ここで $R_j^2$ は、$\phi_j(\mathbf{x})$ を目的変数として他の基底関数で回帰した場合の決定係数です。$VIF_j > 10$ の場合、多重共線性が疑われます。

2. **条件数（Condition Number）**: $\mathbf{\Phi}^T\mathbf{\Phi}$ の最大固有値と最小固有値の比。値が大きい場合（例：$> 30$）、多重共線性が疑われます。

### Solutions

多重共線性の問題を解決する方法：

1. **基底関数の削除**: 相関の高い基底関数の一方を削除
2. **正則化**: リッジ回帰（L2正則化）やLasso回帰（L1正則化）を適用
3. **直交基底の使用**: 直交多項式など、互いに直交する基底関数を使用
4. **主成分回帰（PCR）**: 主成分分析を用いて基底関数を変換

## Limitations and Constraints

1. **パラメータに対する線形性の仮定**: パラメータに対して非線形なモデル（例: $y = w_0 + w_1 e^{w_2 x}$）には対応できない。

2. **基底関数の選択**: 適切な基底関数の選択が困難な場合がある。ドメイン知識や探索的なデータ解析が必要。

3. **外れ値への敏感性**: 外れ値の影響を受けやすい。頑健回帰（robust regression）手法の使用を検討します。

4. **多重共線性**: 基底関数間の相関が高い場合、推定が不安定になる。

5. **過学習**: 基底関数が多すぎる場合、過学習が発生する可能性がある。特に $p$ が $n$ に近い場合や $p > n$ の場合には注意が必要です。

6. **サンプルサイズ**: パラメータ数に対してサンプル数が少ない場合、推定が不安定になる。一般に $n \geq 10(p+1)$ が推奨されます。

7. **仮定の違反**: 等分散性や独立性の仮定が満たされない場合、標準誤差の推定が不正確になる可能性があります。

## Extensions and Variations

線形回帰には様々な拡張が存在します：

1. **リッジ回帰**: L2正則化により多重共線性を緩和
2. **Lasso回帰**: L1正則化により基底関数選択を同時に行う
3. **Elastic Net**: L1とL2正則化を組み合わせた手法
4. **ロバスト回帰**: 外れ値に対して頑健な手法（例：Huber回帰、RANSAC）
5. **一般化最小二乗法（GLS）**: 等分散性の仮定を緩和
6. **一般化推定方程式（GEE）**: 相関のある観測値を扱う
7. **ベイジアン線形回帰**: ベイズ推論を用いた線形回帰

## Application Examples

線形回帰は様々な分野で応用されています：

- **多項式回帰**: 非線形関係のモデル化（例: 物理現象の近似）
- **フーリエ解析**: 周期関数の近似（$\sin$, $\cos$ 基底）
- **スプライン回帰**: 区分多項式による滑らかな関数近似
- **カーネル回帰**: カーネル関数を基底関数として使用
- **特徴量エンジニアリング**: ドメイン知識に基づく基底関数の設計

## Relationship with Other Methods

線形回帰は、多くの機械学習手法の基礎となっています：

- **ロジスティック回帰**: 線形回帰を分類問題に拡張
- **ニューラルネットワーク**: 線形回帰は単層ニューラルネットワークと等価
- **サポートベクターマシン**: 線形カーネルを使用した場合、線形回帰と関連
- **カーネル法**: カーネル関数を基底関数として使用する線形回帰の拡張

## Key Insight: What Makes It "Linear"?

線形回帰の「線形」とは、**パラメータ（重み）に対して線形**であることを意味します。これは以下のことを意味します：

- ✅ **線形回帰**: $y = w_0 + w_1 x + w_2 x^2 + w_3 \sin(x)$
  - 重み $w_i$ に対して線形
  - 説明変数 $x$ は非線形でも構わない

- ❌ **非線形回帰**: $y = w_0 + w_1 e^{w_2 x}$
  - 重み $w_2$ に対して非線形
  - これは非線形回帰モデル

この区別が、線形回帰の本質的な理解において重要です。

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning: with Applications in R* (2nd ed.). Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). *Introduction to Linear Regression Analysis* (6th ed.). Wiley.
