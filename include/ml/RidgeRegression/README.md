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
- $\|\mathbf{w}\|^2 = \sum_{j=0}^{p} w_j^2$: パラメータベクトルのL2ノルムの二乗

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
- $\mathbf{\Phi}$: 計画行列（design matrix, $n \times (p+1)$）
- $\mathbf{w}$: パラメータベクトル（$(p+1) \times 1$）
- $\lambda$: 正則化パラメータ（スカラー）

### Intercept Handling

切片項 $w_0$ を正則化に含めるかどうかは実装によって異なります：

1. **切片項も正則化**: $\lambda\|\mathbf{w}\|^2 = \lambda\sum_{j=0}^{p} w_j^2$
2. **切片項を除外**: $\lambda\|\mathbf{w}\|^2 = \lambda\sum_{j=1}^{p} w_j^2$（切片項 $w_0$ は正則化しない）

一般的には、切片項を正則化から除外することが多いです。これは、切片項はデータの平均的なオフセットを表すため、正則化の対象としない方が自然だからです。

## Parameter Estimation

### Normal Equations

目的関数 $J(\mathbf{w})$ を $\mathbf{w}$ について偏微分し、0とおくことで、以下の正規方程式が得られます：

$$\frac{\partial J(\mathbf{w})}{\partial \mathbf{w}} = -2\mathbf{\Phi}^T(\mathbf{y} - \mathbf{\Phi}\mathbf{w}) + 2\lambda\mathbf{w} = \mathbf{0}$$

これを整理すると：

$$(\mathbf{\Phi}^T\mathbf{\Phi} + \lambda\mathbf{I})\hat{\mathbf{w}} = \mathbf{\Phi}^T\mathbf{y}$$

ここで $\mathbf{I}$ は $(p+1) \times (p+1)$ の単位行列です。

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
   - 直接的な逆行列計算: $O((p+1)^3)$
   - コレスキー分解: $O((p+1)^3)$（ただし、数値的に安定）
   - SVD分解: $O(n(p+1)^2)$（$n > p+1$ の場合）

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
2. **パラメータ数が多い場合**: $p$ が $n$ に近い、または $p > n$ の場合
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
