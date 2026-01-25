# 最急降下法 (Gradient Descent)

## Overview

最急降下法（Gradient Descent）は、目的関数を最小化するための最も基本的で広く使用されている最適化アルゴリズムです。目的関数の勾配（gradient）の反対方向にパラメータを更新することで、関数の最小値に向かって反復的に探索します。

最急降下法は、機械学習、深層学習、統計的最適化など、様々な分野で広く応用されています。特に、解析的な解が得られない非線形最適化問題において、数値的に最適解を求める際に使用されます。

## Mathematical Formulation

### Objective Function

最急降下法は、以下のような目的関数 $f(\mathbf{x})$ の最小化を目的とします：

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$$

ここで：
- $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$: 最適化するパラメータベクトル（$n$次元）
- $f: \mathbb{R}^n \to \mathbb{R}$: 目的関数（スカラー値を返す）

### Gradient

目的関数 $f(\mathbf{x})$ の勾配ベクトル $\nabla f(\mathbf{x})$ は、各変数に関する偏微分のベクトルとして定義されます：

$$\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)^T$$

勾配ベクトルは、関数値が最も急激に増加する方向を指します。したがって、その反対方向（負の勾配方向）は、関数値が最も急激に減少する方向となります。

### Update Rule

最急降下法の更新式は以下のように表されます：

$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \alpha \nabla f(\mathbf{x}^{(t)})$$

ここで：
- $\mathbf{x}^{(t)}$: $t$ 回目の反復におけるパラメータベクトル
- $\alpha > 0$: 学習率（learning rate）またはステップサイズ（step size）
- $\nabla f(\mathbf{x}^{(t)})$: $\mathbf{x}^{(t)}$ における勾配ベクトル

### Component-wise Update

各成分について明示的に書くと：

$$x_i^{(t+1)} = x_i^{(t)} - \alpha \frac{\partial f}{\partial x_i}(\mathbf{x}^{(t)}), \quad i = 1, 2, \ldots, n$$

## Algorithm Description

### Basic Algorithm

最急降下法の基本的なアルゴリズムは以下の通りです：

1. **初期化**: 初期パラメータ $\mathbf{x}^{(0)}$ を設定
2. **反復**: 以下のステップを収束するまで繰り返す：
   a. 現在のパラメータ $\mathbf{x}^{(t)}$ における勾配 $\nabla f(\mathbf{x}^{(t)})$ を計算
   b. パラメータを更新: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \alpha \nabla f(\mathbf{x}^{(t)})$
   c. 収束判定: 目的関数値の変化が許容誤差以下になった場合、または最大反復回数に達した場合、終了
3. **出力**: 最適化されたパラメータ $\mathbf{x}^*$ を返す

### Convergence Criterion

収束判定には、以下のような基準が使用されます：

1. **目的関数値の変化**: 
   $$|f(\mathbf{x}^{(t+1)}) - f(\mathbf{x}^{(t)})| < \epsilon$$
   ここで $\epsilon$ は許容誤差（tolerance）です。

2. **勾配のノルム**:
   $$\|\nabla f(\mathbf{x}^{(t)})\| < \epsilon$$
   最適解では勾配がゼロベクトルになるため、勾配のノルムが十分に小さくなった場合に収束とみなします。

3. **パラメータの変化**:
   $$\|\mathbf{x}^{(t+1)} - \mathbf{x}^{(t)}\| < \epsilon$$
   パラメータの更新量が十分に小さくなった場合に収束とみなします。

本実装では、目的関数値の変化による収束判定を使用しています。

## Numerical Differentiation

解析的な勾配が提供されない場合、数値微分（numerical differentiation）を用いて勾配を近似します。

### Central Difference Method

本実装では、中央差分法（central difference method）を使用しています。各変数 $x_i$ に関する偏微分は以下のように近似されます：

$$\frac{\partial f}{\partial x_i}(\mathbf{x}) \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h}$$

ここで：
- $h > 0$: 微小値（epsilon）
- $\mathbf{e}_i$: $i$ 番目の要素が1、それ以外が0の単位ベクトル

### Advantages of Central Difference

中央差分法は、前進差分法や後退差分法と比較して、以下の利点があります：

1. **精度**: 誤差が $O(h^2)$ のオーダーとなり、前進差分法や後退差分法の $O(h)$ よりも高精度です。
2. **対称性**: 関数の評価点が対称的であるため、数値誤差が相殺されやすくなります。

### Computational Cost

数値微分による勾配計算には、各次元について2回の関数評価が必要です。したがって、$n$ 次元のパラメータベクトルに対しては、$2n$ 回の関数評価が必要となります。これは、解析的な勾配が利用可能な場合と比較して計算コストが高いため、可能な限り解析的な勾配を提供することが推奨されます。

## Parameters

### Learning Rate (α)

学習率 $\alpha$ は、各反復におけるパラメータの更新量を制御する最も重要なハイパーパラメータです。

- **大きすぎる場合**: 最適解を飛び越えて発散する可能性があります（オーバーシュート）
- **小さすぎる場合**: 収束が遅くなり、局所最適解に留まる可能性があります

適切な学習率の選択は、問題の性質や目的関数の形状に依存します。一般的には、$0.001$ から $0.1$ の範囲で試行錯誤により決定されます。

### Maximum Iterations

最大反復回数は、アルゴリズムが収束しない場合に無限ループを防ぐための安全装置です。目的関数の形状や初期値によっては、収束までに多くの反復が必要になる場合があります。

### Tolerance (ε)

許容誤差 $\epsilon$ は、収束判定の基準として使用されます。目的関数値の変化がこの値以下になった場合、収束とみなします。一般的には、$10^{-6}$ から $10^{-8}$ 程度の値が使用されます。

### Epsilon (h)

数値微分における微小値 $h$ は、勾配の近似精度に影響します。

- **大きすぎる場合**: 近似誤差が大きくなります
- **小さすぎる場合**: 浮動小数点演算の丸め誤差が大きくなります

一般的には、$10^{-8}$ から $10^{-6}$ 程度の値が使用されます。目的関数のスケールに応じて調整が必要な場合があります。

## Convergence Analysis

### Convergence Conditions

最急降下法が収束するための十分条件は以下の通りです：

1. **目的関数の凸性**: 目的関数 $f(\mathbf{x})$ が凸関数である場合、大域的最適解に収束することが保証されます。
2. **リプシッツ連続性**: 勾配 $\nabla f(\mathbf{x})$ がリプシッツ連続である場合、適切な学習率を選択することで収束が保証されます。
3. **学習率の条件**: 学習率が以下の条件を満たす場合、収束が保証されます：
   $$0 < \alpha < \frac{2}{L}$$
   ここで $L$ は勾配のリプシッツ定数です。

### Convergence Rate

最急降下法の収束速度は、目的関数の性質に依存します：

- **強凸関数**: $O(\log(1/\epsilon))$ の反復回数で $\epsilon$-最適解に到達
- **一般の凸関数**: $O(1/\epsilon)$ の反復回数で $\epsilon$-最適解に到達
- **非凸関数**: 局所最適解への収束が保証される（大域的最適解ではない可能性がある）

## Advantages and Disadvantages

### Advantages

1. **実装の簡単さ**: アルゴリズムが単純で実装が容易です。
2. **汎用性**: 様々な目的関数に適用可能です。
3. **メモリ効率**: 勾配ベクトルのみを保持すればよいため、メモリ使用量が少ないです。
4. **数値微分のサポート**: 解析的な勾配がなくても、数値微分により勾配を近似できます。

### Disadvantages

1. **収束速度**: 目的関数の形状によっては、収束が非常に遅くなる場合があります（特に、条件数が大きい場合）。
2. **局所最適解**: 非凸関数の場合、大域的最適解ではなく局所最適解に収束する可能性があります。
3. **学習率の調整**: 適切な学習率の選択が困難な場合があります。
4. **振動**: 学習率が大きすぎる場合、最適解の周りで振動する可能性があります。
5. **計算コスト**: 数値微分を使用する場合、各反復で $2n$ 回の関数評価が必要となり、計算コストが高くなります。

## Variants and Extensions

### Stochastic Gradient Descent (SGD)

確率的勾配降下法は、全データではなく、ランダムに選択したサンプル（または1つのサンプル）を用いて勾配を計算します。これにより、計算コストを削減し、大規模データセットに適用可能になります。

### Mini-batch Gradient Descent

ミニバッチ勾配降下法は、全データと1サンプルの間を取った手法で、小さなバッチ（例：32、64、128サンプル）を用いて勾配を計算します。SGDとバッチ勾配降下法の良いバランスを提供します。

### Momentum

モーメンタム法は、過去の更新方向を考慮することで、振動を抑制し、収束を加速します。更新式は以下のようになります：

$$\mathbf{v}^{(t+1)} = \beta \mathbf{v}^{(t)} - \alpha \nabla f(\mathbf{x}^{(t)})$$
$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + \mathbf{v}^{(t+1)}$$

ここで $\beta$ はモーメンタム係数（通常0.9程度）です。

### Adaptive Learning Rate Methods

適応的学習率法は、各パラメータに対して異なる学習率を動的に調整します。代表的な手法には以下があります：

- **AdaGrad**: 過去の勾配の二乗和に基づいて学習率を調整
- **RMSprop**: AdaGradの改良版で、指数移動平均を使用
- **Adam**: モーメンタムとRMSpropを組み合わせた手法

### Constrained Optimization

本実装では、境界制約（bounds）をサポートしています。各パラメータに対して上限と下限を設定することで、制約付き最適化問題に対応できます。更新後のパラメータは、以下のようにクリッピングされます：

$$x_i = \text{clamp}(x_i, \text{lower}_i, \text{upper}_i)$$

## Application Examples

最急降下法は、以下のような問題に適用されます：

1. **機械学習**: 線形回帰、ロジスティック回帰、ニューラルネットワークのパラメータ最適化
2. **統計的最適化**: 最尤推定、ベイズ推論
3. **工学最適化**: 設計パラメータの最適化
4. **信号処理**: フィルタ設計、信号復元

## Relationship with Other Optimization Methods

最急降下法は、多くの最適化手法の基礎となっています：

- **共役勾配法**: 最急降下法を改良し、共役方向を探索することで収束を加速
- **準ニュートン法（BFGS, L-BFGS）**: ヘッセ行列の近似を用いて、より効率的な探索方向を計算
- **ニュートン法**: ヘッセ行列の逆行列を用いて、より直接的に最適解に近づく

## Implementation Notes

### Maximization

本実装では、最大化問題もサポートしています。最大化は、目的関数の符号を反転させた最小化問題として実装されています：

$$\max_{\mathbf{x}} f(\mathbf{x}) = \min_{\mathbf{x}} (-f(\mathbf{x}))$$

勾配も同様に符号を反転させます：

$$\nabla(-f)(\mathbf{x}) = -\nabla f(\mathbf{x})$$

### Boundary Constraints

境界制約が指定された場合、各反復後にパラメータが指定された範囲内にクリッピングされます。これにより、制約付き最適化問題に対応できます。

### Numerical Stability

数値微分を使用する際は、以下の点に注意が必要です：

1. **微小値の選択**: 目的関数のスケールに応じて適切な微小値を選択する必要があります。
2. **丸め誤差**: 浮動小数点演算の丸め誤差が、特に微小値が小さすぎる場合に問題となる可能性があります。
3. **計算コスト**: 数値微分は計算コストが高いため、可能な限り解析的な勾配を提供することが推奨されます。

## References

- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). Optimization methods for large-scale machine learning. *SIAM Review*, 60(2), 223-311.
