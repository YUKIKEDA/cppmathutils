# リッジ回帰 (Ridge Regression)

## Overview

リッジ回帰（Ridge Regression）は、線形回帰にL2正則化（L2 regularization）を追加した手法です。最小二乗法の目的関数に、パラメータのL2ノルム（二乗和）をペナルティ項として加えることで、パラメータの大きさを制約し、過学習や多重共線性の問題を緩和します。

リッジ回帰は、1960年代にHoerlとKennardによって提案され、特に以下の問題に対処するために開発されました：

1. **多重共線性（Multicollinearity）**: 説明変数間の相関が高い場合、通常の最小二乗法では推定値が不安定になる
2. **過学習（Overfitting）**: パラメータ数が多い場合、訓練データに過度に適合し、汎化性能が低下する
3. **数値的不安定性**: 計画行列が特異に近い場合、逆行列の計算が不安定になる

リッジ回帰は、パラメータを0に近づけることで、モデルの複雑さを制御し、より汎化性能の高いモデルを構築します。

## Mathematical Formulation

### Objective Function

リッジ回帰の目的関数は、残差平方和（SSR）にL2正則化項を加えたものです：

$$J(\mathbf{w}) = \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2$$

ここで：
- $\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2$: 残差平方和（Sum of Squared Residuals, SSR）
- $\lambda\|\mathbf{w}\|^2$: L2正則化項（L2 penalty term）
- $\lambda \geq 0$: 正則化パラメータ（regularization parameter, ハイパーパラメータ）
- $\|\mathbf{w}\|^2 = \sum_{j=0}^{d} w_j^2$: パラメータベクトルのL2ノルムの二乗（$d$ は説明変数の数）

### Regularization Parameter

正則化パラメータ $\lambda$ は、データへの適合度とパラメータの大きさのバランスを制御します：

- **$\lambda = 0$**: 通常の最小二乗法（OLS）と等価
- **$\lambda > 0$**: パラメータが0に近づくように制約
- **$\lambda \to \infty$**: すべてのパラメータが0に近づく（切片項を除く場合もある）

### Matrix Notation

目的関数を行列形式で表現すると：

$$J(\mathbf{w}) = (\mathbf{y} - \mathbf{\Phi}\mathbf{w})^T(\mathbf{y} - \mathbf{\Phi}\mathbf{w}) + \lambda\mathbf{w}^T\mathbf{w}$$

ここで：
- $\mathbf{y}$: 目的変数ベクトル（$n \times 1$）
- $\mathbf{\Phi}$: 計画行列（design matrix, $n \times (d+1)$、$d$ は説明変数の数）
- $\mathbf{w}$: パラメータベクトル（$(d+1) \times 1$）
- $\lambda$: 正則化パラメータ（スカラー）

### Intercept Handling

切片項 $w_0$ を正則化に含めるかどうかは実装によって異なります：

1. **切片項も正則化**: $\lambda\|\mathbf{w}\|^2 = \lambda\sum_{j=0}^{d} w_j^2$（$d$ は説明変数の数）
2. **切片項を除外**: $\lambda\|\mathbf{w}\|^2 = \lambda\sum_{j=1}^{d} w_j^2$（切片項 $w_0$ は正則化しない）

一般的には、切片項を正則化から除外することが多いです。これは、切片項はデータの平均的なオフセットを表すため、正則化の対象としない方が自然だからです。

## Parameter Estimation

### Normal Equations

目的関数 $J(\mathbf{w})$ を $\mathbf{w}$ について偏微分し、0とおくことで、以下の正規方程式が得られます：

$$\frac{\partial J(\mathbf{w})}{\partial \mathbf{w}} = -2\mathbf{\Phi}^T(\mathbf{y} - \mathbf{\Phi}\mathbf{w}) + 2\lambda\mathbf{w} = \mathbf{0}$$

これを整理すると：

$$(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})\hat{\mathbf{w}} = \mathbf{\Phi}^T\mathbf{y}$$

ここで $\mathbf{I}$ は $(d+1) \times (d+1)$ の単位行列です（$d$ は説明変数の数）。

### Parameter Estimation Formula

正規方程式を解くことで、リッジ回帰のパラメータ推定値が得られます：

$$\hat{\mathbf{w}} = (\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}\mathbf{\Phi}^T\mathbf{y}$$

### Intercept Exclusion

切片項を正則化から除外する場合、単位行列 $\mathbf{I}$ の最初の対角要素（切片項に対応）を0にします：

$$\mathbf{I}_{\text{ridge}} = \begin{pmatrix} 0 & 0 & 0 & \cdots & 0 \\ 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \end{pmatrix}$$

この場合、正規方程式は：

$$(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I}_{\text{ridge}})\hat{\mathbf{w}} = \mathbf{\Phi}^T\mathbf{y}$$

となります。

### Comparison with OLS

通常の最小二乗法（OLS）の推定式：

$$\hat{\mathbf{w}}_{\text{OLS}} = (\mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{y}$$

リッジ回帰の推定式：

$$\hat{\mathbf{w}}_{\text{Ridge}} = (\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}\mathbf{\Phi}^T\mathbf{y}$$

リッジ回帰では、$\mathbf{\Phi}^T\mathbf{\Phi}$ に $\lambda\mathbf{I}$ を加えることで、行列が正則（可逆）になり、多重共線性の問題を緩和します。

### Uniqueness of Solution

$\lambda > 0$ の場合、$(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})$ は常に正定値（positive definite）であり、逆行列が一意に存在します。これは、$\mathbf{\Phi}^T\mathbf{\Phi}$ が特異（singular）であっても、リッジ回帰は解を持つことを意味します。

## Probabilistic Model and Bayesian Derivation

リッジ回帰は、確率モデルの観点から自然に導出されます。パラメータに正規分布の事前分布を仮定したベイズ線形回帰において、最大事後確率（Maximum A Posteriori, MAP）推定を行うと、リッジ回帰の目的関数が自然に導かれます。この導出により、リッジ回帰が単なる正則化手法ではなく、統計的に正当化された推定手法であることが理解できます。

### Probabilistic Model

通常の線形回帰と同様に、誤差項 $\epsilon_i$ が独立に正規分布 $N(0, \sigma^2)$ に従うと仮定します：

$$\epsilon_i \sim N(0, \sigma^2), \quad i = 1, 2, \ldots, n$$

この仮定により、目的変数 $y_i$ は以下の正規分布に従います：

$$y_i \sim N(\mathbf{\phi}_i^T\mathbf{w}, \sigma^2)$$

ここで $\mathbf{\phi}_i^T$ は計画行列 $\mathbf{\Phi}$ の $i$ 行目です。

**補足：目的変数の分布の導出**

線形回帰モデル $y_i = \mathbf{\phi}_i^T\mathbf{w} + \epsilon_i$ において、誤差項 $\epsilon_i \sim N(0, \sigma^2)$ が正規分布に従うため、$y_i$ は確率変数 $\epsilon_i$ の線形変換として表されます。正規分布の性質により、$y_i$ も正規分布に従い、その平均は $\mathbf{\phi}_i^T\mathbf{w}$、分散は $\sigma^2$ となります。

確率密度関数は以下のように表されます：

$$f(y_i | \mathbf{\phi}_i, \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{\phi}_i^T\mathbf{w})^2}{2\sigma^2}\right)$$

### Likelihood Function

$n$ 個の観測値が独立であると仮定すると、同時確率密度関数（尤度関数）は各観測値の確率密度関数の積として表されます：

$$p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) = \prod_{i=1}^{n} f(y_i | \mathbf{\phi}_i, \mathbf{w}, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{\phi}_i^T\mathbf{w})^2}{2\sigma^2}\right)$$

**補足：尤度関数の導出**

独立性の仮定により、$n$ 個の観測値の同時確率密度関数は、各観測値の確率密度関数の積として表されます。これは、各観測値が互いに影響を与えないということを意味します。

積を整理すると：

$$p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) = \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{\phi}_i^T\mathbf{w})^2\right)$$

$$= \left(\frac{1}{2\pi\sigma^2}\right)^{n/2} \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2\right)$$

ここで、$\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 = \sum_{i=1}^{n}(y_i - \mathbf{\phi}_i^T\mathbf{w})^2$ は残差平方和です。

### Log-Likelihood Function

計算の簡便性のため、対数尤度関数を考えます：

$$\log p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) = \log \left[\left(\frac{1}{2\pi\sigma^2}\right)^{n/2} \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2\right)\right]$$

**補足：対数尤度関数の導出**

対数関数の性質 $\log(ab) = \log a + \log b$ と $\log(a^c) = c\log a$、$\log(e^x) = x$ を用いて展開すると：

$$\log p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) = \log \left(\frac{1}{2\pi\sigma^2}\right)^{n/2} + \log \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2\right)$$

$$= \frac{n}{2}\log\left(\frac{1}{2\pi\sigma^2}\right) - \frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2$$

$$= \frac{n}{2}\left[\log 1 - \log(2\pi\sigma^2)\right] - \frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2$$

対数関数は単調増加関数であるため、尤度関数の最大化と対数尤度関数の最大化は等価です。

### Prior Distribution

これまで、最尤推定（Maximum Likelihood Estimation, MLE）の枠組みで、尤度関数 $p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2)$ を最大化することでパラメータ $\mathbf{w}$ を推定してきました。この方法では、パラメータ $\mathbf{w}$ に関する事前知識を一切仮定しません。

#### 最尤推定の限界

対数尤度関数を最大化する最尤推定は、観測データ $\mathbf{y}$ が与えられた下で、そのデータを最もよく説明するパラメータ $\mathbf{w}$ を推定します。しかし、この方法には以下のような問題があります：

1. **過学習（Overfitting）**: パラメータ数が多い場合、訓練データに過度に適合し、汎化性能が低下する可能性があります。特に、観測データ数 $n$ がパラメータ数 $d+1$ に近い、または $n < d+1$ の場合、最尤推定は不安定になります。

2. **多重共線性（Multicollinearity）**: 説明変数間の相関が高い場合、$\mathbf{\Phi}^T\mathbf{\Phi}$ が特異に近くなり、逆行列の計算が不安定になります。この場合、パラメータ推定値の分散が非常に大きくなり、推定値が不安定になります。

3. **事前知識の欠如**: 最尤推定では、パラメータに関する事前知識（例えば、「パラメータは0に近い値であることが期待される」など）を活用できません。

#### ベイジアンアプローチの導入

これらの問題を解決するために、ベイジアンアプローチを導入します。ベイジアンアプローチでは、パラメータ $\mathbf{w}$ を確率変数として扱い、 **事前分布（prior distribution）** $p(\mathbf{w})$ を仮定します。これにより、パラメータに関する事前知識を統計モデルに組み込むことができます。

ベイズの定理により、観測データ $\mathbf{y}$ が与えられた下でのパラメータ $\mathbf{w}$ の**事後分布（posterior distribution）** は、以下のように表されます：

$$p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2) = \frac{p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) \cdot p(\mathbf{w})}{p(\mathbf{y} | \mathbf{\Phi}, \sigma^2)}$$

ここで：
- $p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2)$: 尤度関数（likelihood）
- $p(\mathbf{w})$: 事前分布（prior distribution）
- $p(\mathbf{y} | \mathbf{\Phi}, \sigma^2)$: 正規化定数（normalizing constant, エビデンス）

分母の $p(\mathbf{y} | \mathbf{\Phi}, \sigma^2)$ は $\mathbf{w}$ に依存しないため、事後分布の最大化（最大事後確率推定、Maximum A Posteriori, MAP）を考える際には、以下の比例関係が重要です：

$$p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2) \propto p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) \cdot p(\mathbf{w})$$

つまり、**事後分布は尤度関数と事前分布の積に比例**します。この関係により、尤度関数を最大化するだけではなく、事前分布も考慮した推定が可能になります。

#### なぜ同時確率を考えるのか

最尤推定では、尤度関数 $p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2)$ のみを最大化します。しかし、ベイジアンアプローチでは、**同時確率** $p(\mathbf{y}, \mathbf{w} | \mathbf{\Phi}, \sigma^2) = p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) \cdot p(\mathbf{w})$ を最大化します。

この違いは重要です：

- **最尤推定**: 観測データ $\mathbf{y}$ が与えられた下で、そのデータを最もよく説明する $\mathbf{w}$ を探します。これは、$p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2)$ を最大化することに対応します。

- **MAP推定**: 観測データ $\mathbf{y}$ と事前知識 $p(\mathbf{w})$ の両方を考慮して、最も妥当な $\mathbf{w}$ を探します。これは、$p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) \cdot p(\mathbf{w})$ を最大化することに対応します。

事前分布 $p(\mathbf{w})$ を導入することで、パラメータが「合理的な範囲」に制約され、過学習や多重共線性の問題を緩和できます。特に、パラメータが0に近い値であることを期待する場合、事前分布として0を中心とした分布を選ぶことで、パラメータの大きさを制約できます。

#### ガウス分布を事前分布として選ぶ理由

リッジ回帰では、パラメータ $\mathbf{w}$ の事前分布として、**多変量正規分布（multivariate normal distribution）** を仮定します：

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \alpha^{-1}\mathbf{I}) = \left(\frac{\alpha}{2\pi}\right)^{(d+1)/2} \exp\left(-\frac{\alpha}{2}\mathbf{w}^T\mathbf{w}\right)$$

ここで：
- $\mathbf{0}$: 平均ベクトル（すべての要素が0）
- $\alpha^{-1}\mathbf{I}$: 共分散行列（$\alpha > 0$ は精度パラメータ、$\mathbf{I}$ は単位行列）
- $d+1$: パラメータの数（切片項を含む）

> [!NOTE]
> **多変量正規分布の補足**
> 
> 多変量正規分布は、機械学習や統計学において非常に重要な分布です。以下では、1次元のガウス分布から多変量正規分布への導出、周辺化、条件付き分布について説明します。
> 
> #### 1次元のガウス分布から多変量正規分布への導出
> 
> **1次元のガウス分布（正規分布）**
> 
> 1次元の確率変数 $x \in \mathbb{R}$ がガウス分布に従う場合、その確率密度関数は：
> 
> $$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) = \mathcal{N}(x | \mu, \sigma^2)$$
> 
> ここで、$\mu$ は平均、$\sigma^2$ は分散です。
> 
> **多変量正規分布への拡張**
> 
> $d$ 次元の確率変数ベクトル $\mathbf{x} = (x_1, x_2, \ldots, x_d)^T \in \mathbb{R}^d$ が多変量正規分布に従う場合、その確率密度関数は：
> 
> $$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \mathbf{\Sigma})$$
> 
> ここで：
> - $\boldsymbol{\mu} \in \mathbb{R}^d$: 平均ベクトル
> - $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$: 共分散行列（正定値対称行列）
> - $|\mathbf{\Sigma}|$: 共分散行列の行列式（determinant）
> 
> **精度行列（Precision Matrix）**
> 
> 多変量正規分布では、共分散行列の逆行列 $\mathbf{\Lambda} = \mathbf{\Sigma}^{-1}$ を**精度行列（precision matrix）** と呼びます。確率密度関数は、精度行列を用いて以下のようにも表すことができます：
> 
> $$p(\mathbf{x}) = \frac{|\mathbf{\Lambda}|^{1/2}}{(2\pi)^{d/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\mathbf{\Lambda}(\mathbf{x} - \boldsymbol{\mu})\right)$$
> 
> ここで、$|\mathbf{\Lambda}| = |\mathbf{\Sigma}^{-1}| = |\mathbf{\Sigma}|^{-1}$ です。
> 
> **精度行列の意味**
> 
> 1. **1次元の場合**: 精度は分散の逆数 $\lambda = \sigma^{-2}$ で、精度が高いほど分散が小さく、確率密度が平均の周りに集中します。
> 
> 2. **多変量の場合**: 精度行列 $\mathbf{\Lambda}$ の対角要素 $\Lambda_{ii}$ は、$i$ 番目の変数の精度（分散の逆数）を表します。非対角要素 $\Lambda_{ij}$（$i \neq j$）は、変数間の条件付き相関を表します。
> 
> 3. **独立性との関係**: 精度行列が対角行列である場合（$\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$）、各変数は独立になります。これは、共分散行列が対角行列であることと等価です。
> 
> **共分散行列と精度行列の関係**
> 
> - $\mathbf{\Lambda} = \mathbf{\Sigma}^{-1}$ および $\mathbf{\Sigma} = \mathbf{\Lambda}^{-1}$
> - 共分散行列が対角行列 $\mathbf{\Sigma} = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_d^2)$ の場合、精度行列も対角行列 $\mathbf{\Lambda} = \text{diag}(\sigma_1^{-2}, \sigma_2^{-2}, \ldots, \sigma_d^{-2})$ になります。
> 
> **リッジ回帰での使用**
> 
> リッジ回帰の事前分布 $p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \alpha^{-1}\mathbf{I})$ では、精度行列は $\mathbf{\Lambda} = \alpha\mathbf{I}$ となります。これは、各パラメータが独立で、精度 $\alpha$ を持つことを意味します。$\alpha$ が大きいほど、パラメータが0に近い値になることを強く期待します。
> 
> **導出の考え方**
> 
> 多変量正規分布は、以下の2つのアプローチから導出できます：
> 
> 1. **独立な1次元ガウス分布の積**: $d$ 個の独立な確率変数 $x_1, x_2, \ldots, x_d$ がそれぞれ $\mathcal{N}(\mu_i, \sigma_i^2)$ に従う場合、同時確率密度関数は各確率密度関数の積になります。共分散行列が対角行列 $\mathbf{\Sigma} = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_d^2)$ の場合、多変量正規分布は独立な1次元ガウス分布の積として表されます。
> 
> 2. **線形変換による導出**: 標準多変量正規分布 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（各成分が独立に標準正規分布に従う）に対して、線形変換 $\mathbf{x} = \mathbf{A}\mathbf{z} + \boldsymbol{\mu}$ を適用すると、$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{A}\mathbf{A}^T)$ となります。ここで、$\mathbf{A}\mathbf{A}^T = \mathbf{\Sigma}$ とすることで、任意の共分散行列を持つ多変量正規分布が得られます。
> 
> **指数部分の解釈**
> 
> 1次元の場合の指数部分 $-\frac{(x-\mu)^2}{2\sigma^2}$ は、多変量の場合では $-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ に拡張されます。これは、**マハラノビス距離（Mahalanobis distance）** の二乗に相当します。$\mathbf{\Sigma}$ が対角行列の場合、これは各成分の標準化された距離の二乗和になります。
> 
> #### 多変量正規分布の周辺化
> 
> **周辺分布の性質**
> 
> 多変量正規分布の重要な性質として、**周辺分布（marginal distribution）もガウス分布に従う**ことが挙げられます。
> 
> 確率変数ベクトル $\mathbf{x} = (\mathbf{x}_1^T, \mathbf{x}_2^T)^T$ が多変量正規分布に従うとします：
> 
> $$\mathbf{x} = \begin{pmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \begin{pmatrix} \mathbf{\Sigma}_{11} & \mathbf{\Sigma}_{12} \\ \mathbf{\Sigma}_{21} & \mathbf{\Sigma}_{22} \end{pmatrix}\right)$$
> 
> ここで、$\mathbf{x}_1$ と $\mathbf{x}_2$ は部分ベクトル、$\boldsymbol{\mu}_1$ と $\boldsymbol{\mu}_2$ は対応する平均ベクトル、$\mathbf{\Sigma}_{11}$、$\mathbf{\Sigma}_{12}$、$\mathbf{\Sigma}_{21}$、$\mathbf{\Sigma}_{22}$ は共分散行列のブロックです。
> 
> **周辺分布の導出**
> 
> $\mathbf{x}_1$ の周辺分布は、$\mathbf{x}_2$ について積分することで得られます：
> 
> $$p(\mathbf{x}_1) = \int p(\mathbf{x}_1, \mathbf{x}_2) d\mathbf{x}_2$$
> 
> この積分を実行すると、$\mathbf{x}_1$ の周辺分布は以下の多変量正規分布になります：
> 
> $$p(\mathbf{x}_1) = \mathcal{N}(\mathbf{x}_1 | \boldsymbol{\mu}_1, \mathbf{\Sigma}_{11})$$
> 
> 同様に、$\mathbf{x}_2$ の周辺分布は：
> 
> $$p(\mathbf{x}_2) = \mathcal{N}(\mathbf{x}_2 | \boldsymbol{\mu}_2, \mathbf{\Sigma}_{22})$$
> 
> **重要な性質**
> 
> この性質により、多変量正規分布に従う確率変数の任意の部分集合の周辺分布も、多変量正規分布（1次元の場合は1次元のガウス分布）になります。これは、ベイズ推論において非常に有用な性質です。
> 
> #### 多変量正規分布の条件付き分布
> 
> **条件付き分布の導出**
> 
> $\mathbf{x}_2$ が与えられた下での $\mathbf{x}_1$ の条件付き分布 $p(\mathbf{x}_1 | \mathbf{x}_2)$ も、多変量正規分布になります。これは、多変量正規分布のもう一つの重要な性質です。
> 
> 条件付き分布は以下のように表されます：
> 
> $$p(\mathbf{x}_1 | \mathbf{x}_2) = \mathcal{N}(\mathbf{x}_1 | \boldsymbol{\mu}_{1|2}, \mathbf{\Sigma}_{1|2})$$
> 
> ここで、条件付き平均と条件付き共分散行列は：
> 
> $$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$
> 
> $$\mathbf{\Sigma}_{1|2} = \mathbf{\Sigma}_{11} - \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}\mathbf{\Sigma}_{21}$$
> 
> **導出の考え方**
> 
> 条件付き分布は、同時分布を周辺分布で割ることで得られます：
> 
> $$p(\mathbf{x}_1 | \mathbf{x}_2) = \frac{p(\mathbf{x}_1, \mathbf{x}_2)}{p(\mathbf{x}_2)}$$
> 
> 多変量正規分布の場合、この比を計算すると、指数部分の2次形式を完成平方することで、条件付き分布のパラメータが導出されます。具体的には：
> 
> 1. 同時分布の指数部分を $\mathbf{x}_1$ について整理
> 2. $\mathbf{x}_1$ について2次形式を完成平方
> 3. 完成平方した項から条件付き平均と条件付き共分散行列を読み取る
> 
> **条件付き平均の解釈**
> 
> 条件付き平均 $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ は、以下のように解釈できます：
> 
> - $\boldsymbol{\mu}_1$: $\mathbf{x}_1$ の無条件平均
> - $\mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$: $\mathbf{x}_2$ の観測値が平均からどれだけ離れているかに基づく補正項
> 
> $\mathbf{x}_1$ と $\mathbf{x}_2$ が相関している場合（$\mathbf{\Sigma}_{12} \neq \mathbf{0}$）、$\mathbf{x}_2$ の観測値に基づいて $\mathbf{x}_1$ の期待値が更新されます。
> 
> **条件付き共分散行列の解釈**
> 
> 条件付き共分散行列 $\mathbf{\Sigma}_{1|2} = \mathbf{\Sigma}_{11} - \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}\mathbf{\Sigma}_{21}$ は、$\mathbf{x}_2$ の情報により、$\mathbf{x}_1$ の不確実性が減少することを示しています。$\mathbf{x}_1$ と $\mathbf{x}_2$ の相関が強いほど、条件付き分散は小さくなります。
> 
> **独立性との関係**
> 
> $\mathbf{x}_1$ と $\mathbf{x}_2$ が独立である場合（$\mathbf{\Sigma}_{12} = \mathbf{0}$）、条件付き分布は周辺分布と一致します：
> 
> $$p(\mathbf{x}_1 | \mathbf{x}_2) = \mathcal{N}(\mathbf{x}_1 | \boldsymbol{\mu}_1, \mathbf{\Sigma}_{11}) = p(\mathbf{x}_1)$$
> 
> これは、$\mathbf{x}_2$ の情報が $\mathbf{x}_1$ の分布に影響を与えないことを意味します。
> 
> #### リッジ回帰への応用
> 
> これらの性質は、リッジ回帰のベイズ推論において重要です：
> 
> 1. **事後分布の計算**: 尤度関数と事前分布がともにガウス分布であるため、事後分布もガウス分布になります（共役性）。
> 2. **周辺化**: 不要なパラメータを周辺化することで、必要なパラメータの分布を求めることができます。
> 3. **条件付き分布**: 一部のパラメータが観測された下で、他のパラメータの分布を更新できます。

**なぜガウス分布を選ぶのか：**

1. **数学的簡潔性**: ガウス分布は、尤度関数もガウス分布であるため、事後分布もガウス分布になります（共役事前分布、conjugate prior）。これにより、解析的な導出が可能になります。

2. **正則化との対応**: ガウス分布の対数を取ると、$\mathbf{w}^T\mathbf{w}$ の項が現れます。これは、リッジ回帰のL2正則化項 $\lambda\|\mathbf{w}\|^2$ と対応します。

3. **解釈の容易さ**: 平均0、等方性の共分散行列（$\alpha^{-1}\mathbf{I}$）を仮定することで、「すべてのパラメータは0に近い値であることが期待される」という事前知識を自然に表現できます。

4. **計算の効率性**: ガウス分布の性質により、MAP推定の計算が効率的に行えます。

**補足：精度パラメータ $\alpha$ と正則化パラメータ $\lambda$ の関係**

後述するように、$\alpha$ と $\lambda$ の間には $\lambda = \alpha\sigma^2$ という関係があります。$\alpha$ が大きいほど、事前分布の分散が小さく（精度が高く）、パラメータが0に近い値になることを強く期待します。

#### 事後分布の導出

ベイズの定理により、事後分布は以下のように表されます：

$$p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2) \propto p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) \cdot p(\mathbf{w})$$

尤度関数と事前分布を代入すると：

$$p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2) \propto \left(\frac{1}{2\pi\sigma^2}\right)^{n/2} \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2\right) \cdot \left(\frac{\alpha}{2\pi}\right)^{(d+1)/2} \exp\left(-\frac{\alpha}{2}\mathbf{w}^T\mathbf{w}\right)$$

定数項を除いて整理すると：

$$p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2) \propto \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 - \frac{\alpha}{2}\mathbf{w}^T\mathbf{w}\right)$$

$$= \exp\left(-\frac{1}{2\sigma^2}\left[\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \alpha\sigma^2\|\mathbf{w}\|^2\right]\right)$$

ここで、$\lambda = \alpha\sigma^2$ とおくと：

$$p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2) \propto \exp\left(-\frac{1}{2\sigma^2}\left[\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2\right]\right)$$

#### 最大事後確率（MAP）推定によるリッジ回帰の導出

MAP推定では、事後分布 $p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2)$ を最大化する $\mathbf{w}$ を求めます。対数関数は単調増加であるため、対数事後分布を最大化することと等価です：

$$\log p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2) \propto -\frac{1}{2\sigma^2}\left[\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2\right]$$

定数項を除くと、以下の最大化問題に帰着します：

$$\max_{\mathbf{w}} \left[-\frac{1}{2\sigma^2}\left[\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2\right]\right]$$

これは、以下の最小化問題と等価です：

$$\min_{\mathbf{w}} \left[\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2\right]$$

これが、まさにリッジ回帰の目的関数です！

**補足：最尤推定との比較**

- **最尤推定**: $\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2$（残差平方和のみを最小化）
- **MAP推定（リッジ回帰）**: $\min_{\mathbf{w}} \left[\|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2\right]$（残差平方和と正則化項の和を最小化）

MAP推定では、残差平方和を小さくするだけでなく、パラメータの大きさも制約することで、過学習や多重共線性の問題を緩和します。

#### まとめ

リッジ回帰は、以下の確率モデルから自然に導出されます：

1. **尤度関数**: $p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) = \mathcal{N}(\mathbf{y} | \mathbf{\Phi}\mathbf{w}, \sigma^2\mathbf{I})$（観測データの分布）
2. **事前分布**: $p(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \alpha^{-1}\mathbf{I})$（パラメータの分布）
3. **事後分布**: $p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi}, \sigma^2) \propto p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) \cdot p(\mathbf{w})$（ベイズの定理）
4. **MAP推定**: 事後分布を最大化することで、リッジ回帰の目的関数が導出される

この導出により、リッジ回帰が単なる正則化手法ではなく、統計的に正当化された推定手法であることが理解できます。正則化パラメータ $\lambda$ は、事前分布の精度パラメータ $\alpha$ と観測ノイズの分散 $\sigma^2$ の積として解釈でき、$\lambda = \alpha\sigma^2$ という関係があります。


## Regularization Parameter Selection

正則化パラメータ $\lambda$ の選択は、リッジ回帰の性能に大きく影響します。以下に代表的な選択方法を示します。

### Cross-Validation

最も一般的な方法は、交差検証（cross-validation）を用いて $\lambda$ を選択することです：

1. **K-fold Cross-Validation**: データをK個のフォールドに分割し、各フォールドを検証データとして使用
2. **Leave-One-Out Cross-Validation (LOOCV)**: 各観測値を順番に検証データとして使用
3. **Grid Search**: 候補となる $\lambda$ の値をグリッド状に設定し、各値について交差検証を実行

### Information Criteria

情報量規準を用いて $\lambda$ を選択する方法もあります：

1. **AIC (Akaike Information Criterion)**
2. **BIC (Bayesian Information Criterion)**
3. **Generalized Cross-Validation (GCV)**

### Typical Range

$\lambda$ の典型的な範囲は、データのスケールに依存しますが、以下のような範囲がよく使用されます：

- $10^{-6}$ から $10^6$ の範囲
- 対数スケールで探索: $\lambda \in \{10^{-6}, 10^{-5}, \ldots, 10^5, 10^6\}$

### Effect of Regularization Parameter

$\lambda$ の値による影響：

- **$\lambda$ が小さい**: パラメータの制約が弱く、OLSに近い結果（過学習のリスク）
- **$\lambda$ が適切**: バランスの取れたモデル（汎化性能が高い）
- **$\lambda$ が大きい**: パラメータが0に近づき、モデルが単純化（未学習のリスク）

## Standardization and Normalization

### Importance of Preprocessing

リッジ回帰では、説明変数の前処理（preprocessing）が重要です。これは、L2正則化項がすべてのパラメータに等しく適用されるため、スケールが異なる説明変数がある場合、スケールの大きい変数のパラメータが不適切に制約される可能性があるからです。

### Standardization（標準化）

**Standardization（標準化）** は、データを平均0、標準偏差1に変換する手法です。統計学や機械学習において、最も一般的に使用される前処理手法の一つです。

#### Z-score Standardization

Z-score標準化（Z-score standardization）は、以下の式で定義されます：

$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

ここで：

$\bar{x}_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}$: $j$ 番目の説明変数の平均

$s_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_{ij} - \bar{x}_j)^2}$: $j$ 番目の説明変数の標準偏差（不偏推定量）

標準化後、各説明変数の平均は0、標準偏差（分散）は1になります。

### Normalization（正規化）

**Normalization（正規化）** は、データを特定の範囲（通常は0から1の間）にスケーリングする手法です。Standardizationとは異なり、データの分布形状は保持されますが、スケールが変更されます。

#### Min-Max Normalization

Min-Max正規化は、データを0から1の範囲にスケーリングします：

$$z_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}$$

ここで $\min(x_j)$ と $\max(x_j)$ は、それぞれ $j$ 番目の説明変数の最小値と最大値です。

#### Unit Vector Normalization

単位ベクトル正規化は、各ベクトルをそのノルムで割ることで、単位ベクトルに変換します：

$$z_{ij} = \frac{x_{ij}}{\|\mathbf{x}_j\|}$$

ここで $\|\mathbf{x}_j\| = \sqrt{\sum_{i=1}^{n} x_{ij}^2}$ は $j$ 番目の説明変数のL2ノルムです。

### Standardization vs Normalization

- **Standardization（標準化）**: 平均0、標準偏差1に変換。データの分布形状が変わる可能性がある（外れ値の影響を受けやすい）。
- **Normalization（正規化）**: 特定の範囲（通常0-1）にスケーリング。データの分布形状は保持される。

### Standardization in Ridge Regression

リッジ回帰では、通常 **Z-score標準化（Z-score standardization）** が使用されます。これは、以下の理由によるものです：

1. **L2正則化との整合性**: L2正則化項は、すべてのパラメータに等しく適用されるため、説明変数が同じスケール（平均0、標準偏差1）であることが望ましい
2. **統計的性質**: 標準化により、各説明変数が同じ分散を持つため、パラメータの解釈が容易になる
3. **数値的安定性**: 標準化により、数値計算がより安定する

標準化後、計画行列 $\mathbf{\Phi}$ の各列（切片項を除く）の平均が0、分散が1になります。

**注意**: 
- 切片項は標準化しません
- 予測時にも同じ標準化パラメータ（平均と標準偏差）を使用する必要があります
- 訓練データから計算した平均と標準偏差を保存し、テストデータや新しいデータにも同じ変換を適用します

## Bias-Variance Trade-off

### Bias

リッジ回帰は、パラメータを0に近づけることで、推定値にバイアス（偏り）を導入します。$\lambda > 0$ の場合、リッジ回帰の推定量は不偏推定量ではありません。

$$\text{Bias}(\hat{\mathbf{w}}_{\text{Ridge}}) = E[\hat{\mathbf{w}}_{\text{Ridge}}] - \mathbf{w}_{\text{true}} \neq \mathbf{0}$$

### Variance

一方で、リッジ回帰はパラメータ推定値の分散を減少させます：

$$\text{Var}(\hat{\mathbf{w}}_{\text{Ridge}}) = \sigma^2(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}\mathbf{\Phi}^T\mathbf{\Phi}(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}$$

$\lambda$ が大きくなるほど、分散は減少します。

### Trade-off

リッジ回帰は、バイアスと分散のトレードオフを制御します：

- **$\lambda = 0$（OLS）**: バイアスは小さいが、分散が大きい（特に多重共線性がある場合）
- **$\lambda > 0$（Ridge）**: バイアスは増加するが、分散が減少する
- **最適な $\lambda$**: バイアスと分散のバランスが取れ、予測誤差が最小になる

### Mean Squared Error

予測誤差の期待値（Mean Squared Error, MSE）は、バイアスと分散の和として表現できます：

$$\text{MSE} = \text{Bias}^2 + \text{Var}$$

リッジ回帰は、バイアスの増加よりも分散の減少の方が大きい場合、OLSよりも優れた予測性能を示します。

## Geometric Interpretation

### Constrained Optimization View

リッジ回帰は、以下の制約付き最適化問題として解釈できます：

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 \quad \text{subject to} \quad \|\mathbf{w}\|^2 \leq t$$

ここで $t$ は制約の半径で、$\lambda$ と対応関係があります。$\lambda$ が大きいほど、$t$ は小さくなります。

### Shrinkage

リッジ回帰は、パラメータを0方向に「縮小（shrinkage）」させる効果があります。このため、リッジ回帰は「縮小推定量（shrinkage estimator）」とも呼ばれます。

$$\hat{w}_j^{\text{Ridge}} = \frac{\hat{w}_j^{\text{OLS}}}{1 + \lambda / d_j}$$

ここで $d_j$ は $\mathbf{\Phi}^T\mathbf{\Phi}$ の $j$ 番目の固有値に関連する項です。$\lambda$ が大きいほど、パラメータは0に近づきます。

## Multicollinearity

### Problem with OLS

多重共線性が存在する場合、$\mathbf{\Phi}^T\mathbf{\Phi}$ が特異に近くなり、OLSの推定値が不安定になります：

- パラメータ推定値の分散が大きくなる
- 推定値がデータの小さな変化に対して敏感になる
- 逆行列の計算が数値的に不安定になる

### Solution by Ridge Regression

リッジ回帰は、$\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I}$ を正則化することで、多重共線性の問題を緩和します：

- 行列が常に正定値となり、逆行列が一意に存在する
- パラメータ推定値の分散が減少する
- 推定値がより安定する

### Condition Number

条件数（condition number）は、行列の数値的安定性を測る指標です：

$$\kappa(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I}) = \frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}$$

$\lambda > 0$ の場合、条件数は改善され、数値的安定性が向上します。

## Statistical Properties

### Variance-Covariance Matrix

リッジ回帰のパラメータ推定値の分散共分散行列は：

$$\text{Cov}(\hat{\mathbf{w}}_{\text{Ridge}}) = \sigma^2(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}\mathbf{\Phi}^T\mathbf{\Phi}(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}$$

### Bias

リッジ回帰の推定量のバイアスは：

$$\text{Bias}(\hat{\mathbf{w}}_{\text{Ridge}}) = -\lambda(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}\mathbf{w}_{\text{true}}$$

$\lambda > 0$ の場合、バイアスは0ではありません。

### Mean Squared Error

パラメータ推定値の平均二乗誤差（MSE）は：

$$\text{MSE}(\hat{\mathbf{w}}_{\text{Ridge}}) = \text{Bias}^2 + \text{Var}$$

適切な $\lambda$ を選択することで、OLSよりも小さいMSEを達成できる場合があります。

## Computational Efficiency

### Matrix Operations

リッジ回帰の実装においては、以下の点に注意が必要です：

1. **逆行列の計算**: $(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})^{-1}$ の直接計算は、$\lambda > 0$ の場合、数値的に安定です。

2. **Cholesky分解**: $\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I}$ は正定値行列であるため、コレスキー分解（Cholesky decomposition）を用いて効率的に解くことができます：
   - $\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I} = \mathbf{L}\mathbf{L}^T$ と分解
   - $\mathbf{L}\mathbf{L}^T\hat{\mathbf{w}} = \mathbf{\Phi}^T\mathbf{y}$ を解く

3. **SVD分解**: 特異値分解（Singular Value Decomposition, SVD）を用いる方法もあります：
   - $\mathbf{\Phi} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$ と分解
   - $\hat{\mathbf{w}} = \mathbf{V}(\mathbf{\Sigma}^2 + \lambda\mathbf{I})^{-1}\mathbf{\Sigma}\mathbf{U}^T\mathbf{y}$

4. **計算量**: 
   - 直接的な逆行列計算: $O((d+1)^3)$（$d$ は説明変数の数）
   - コレスキー分解: $O((d+1)^3)$（ただし、数値的に安定）
   - SVD分解: $O(n(d+1)^2)$（$n > d+1$ の場合）

### Efficient Cross-Validation

交差検証を効率的に実行するため、以下の手法が使用されます：

1. **Warm Start**: 前の $\lambda$ の解を初期値として使用
2. **Path Algorithm**: $\lambda$ の値を順次変化させながら、解のパスを追跡
3. **Approximate Methods**: 近似手法を用いて計算を高速化

## Evaluation Metrics

リッジ回帰の評価には、通常の線形回帰と同様の指標を使用します：

### Coefficient of Determination (R²)

$$R^2 = 1 - \frac{SSR}{SST}$$

ただし、リッジ回帰では $R^2$ が負の値になる場合があります（特に $\lambda$ が大きい場合）。

### Mean Squared Error (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE}$$

### Mean Absolute Error (MAE)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

### Cross-Validation Score

交差検証スコア（CV score）は、汎化性能を評価するための重要な指標です：

$$\text{CV Score} = \frac{1}{K}\sum_{k=1}^{K} \text{MSE}_k$$

ここで $\text{MSE}_k$ は $k$ 番目のフォールドでのMSEです。

## Relationship with Other Methods

### Ordinary Least Squares (OLS)

リッジ回帰は、$\lambda = 0$ の場合、OLSと等価です。

### Lasso Regression

Lasso回帰は、L1正則化（$\lambda\|\mathbf{w}\|_1$）を使用します。Lassoはスパース性（多くのパラメータが0になる）を誘発しますが、リッジ回帰はすべてのパラメータを0に近づけるだけです。

### Elastic Net

Elastic Netは、L1とL2正則化を組み合わせた手法です：

$$J(\mathbf{w}) = \|\mathbf{y} - \mathbf{\Phi}\mathbf{w}\|^2 + \lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|^2$$

### Bayesian Linear Regression

リッジ回帰は、ベイジアン線形回帰の特殊ケースとして解釈できます。パラメータに正規分布の事前分布を仮定した場合、事後分布の最頻値（MAP推定）がリッジ回帰の解と一致します。

### Principal Component Regression (PCR)

主成分回帰（PCR）は、主成分分析を用いて説明変数を変換してから回帰を行う手法です。リッジ回帰と関連がありますが、異なるアプローチです。

## Advantages and Disadvantages

### Advantages

1. **多重共線性への対処**: 多重共線性が存在する場合でも、安定した推定が可能
2. **過学習の防止**: パラメータの大きさを制約することで、過学習を抑制
3. **数値的安定性**: $\lambda > 0$ の場合、常に解が存在し、数値的に安定
4. **汎化性能**: 適切な $\lambda$ を選択することで、汎化性能が向上
5. **計算効率**: 比較的効率的に計算可能

### Disadvantages

1. **バイアスの導入**: パラメータ推定値にバイアスが生じる
2. **正則化パラメータの選択**: 適切な $\lambda$ の選択が必要（交差検証など）
3. **特徴選択の欠如**: Lassoとは異なり、特徴選択（変数選択）を行わない
4. **標準化の必要性**: 説明変数の標準化が推奨される
5. **解釈の困難さ**: パラメータの解釈がOLSほど直接的ではない

## Application Examples

リッジ回帰は以下のような場面で使用されます：

1. **多重共線性が存在する場合**: 相関の高い説明変数が多数ある場合
2. **パラメータ数が多い場合**: $d$ が $n$ に近い、または $d > n$ の場合（$d$ は説明変数の数）
3. **過学習を防ぎたい場合**: 訓練データに過度に適合することを避けたい場合
4. **予測精度を重視する場合**: パラメータの解釈よりも予測精度を重視する場合

## Limitations and Constraints

1. **バイアスの存在**: パラメータ推定値にバイアスが生じるため、統計的推論が複雑になる
2. **正則化パラメータの選択**: 適切な $\lambda$ の選択が困難な場合がある
3. **特徴選択の欠如**: Lassoとは異なり、自動的な特徴選択を行わない
4. **標準化の必要性**: 説明変数の標準化が必要（実装によっては自動的に行われる場合もある）
5. **解釈の困難さ**: パラメータの解釈がOLSほど直接的ではない

## Extensions and Variations

リッジ回帰には様々な拡張が存在します：

1. **Adaptive Ridge**: パラメータごとに異なる正則化パラメータを使用
2. **Group Ridge**: グループ単位で正則化を適用
3. **Kernel Ridge Regression**: カーネル関数を用いた非線形拡張
4. **Ridge Regression with Constraints**: 追加の制約を課したリッジ回帰

## Key Insight: Regularization as Prior

リッジ回帰は、ベイジアンの観点から、パラメータに正規分布の事前分布を仮定した場合のMAP推定と等価です：

$$p(\mathbf{w}|\mathbf{y}, \mathbf{\Phi}) \propto p(\mathbf{y}|\mathbf{\Phi}, \mathbf{w}) \cdot p(\mathbf{w})$$

ここで $p(\mathbf{w}) = \mathcal{N}(\mathbf{0}, \frac{1}{\lambda}\mathbf{I})$ とすると、事後分布の最頻値がリッジ回帰の解と一致します。

この解釈により、$\lambda$ は事前分布の精度（precision）パラメータとして理解できます。$\lambda$ が大きいほど、事前分布の分散が小さく、パラメータが0に近い値になることを強く期待します。

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning: with Applications in R* (2nd ed.). Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). *Introduction to Linear Regression Analysis* (6th ed.). Wiley.
