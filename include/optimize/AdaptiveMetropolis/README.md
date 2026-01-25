# Adaptive Metropolis (AM) Algorithm

## Overview

Adaptive Metropolis (AM) は、Metropolis-Hastingsアルゴリズムの改良版で、MCMC（Markov Chain Monte Carlo）法の一種です。Haario et al. (2001)によって提案され、提案分布の共分散行列を適応的に更新することで、サンプリング効率を大幅に向上させます。

## Theoretical Background

### Metropolis-Hastings Algorithm

Metropolis-Hastingsアルゴリズムは、目標分布 $\pi(x)$ からサンプルを生成するためのMCMC法です。アルゴリズムは以下の手順で動作します：

1. **初期化**: 初期状態 $x_0$ を設定
2. **反復**: 各反復 $t$ において：
   - 現在の状態 $x_t$ から提案分布 $q(x'|x_t)$ を用いて候補 $x'$ を生成
   - 受理確率 $\alpha(x_t, x')$ を計算：
     $$
     \alpha(x_t, x') = \min\left(1, \frac{\pi(x') q(x_t|x')}{\pi(x_t) q(x'|x_t)}\right)
     $$
   - 確率 $\alpha(x_t, x')$ で $x_{t+1} = x'$、そうでなければ $x_{t+1} = x_t$

### Improvements in Adaptive Metropolis

従来のMetropolis-Hastingsアルゴリズムでは、提案分布のパラメータ（特に共分散行列）を事前に固定する必要がありました。これにより、以下の問題が発生します：

- 提案分布が目標分布に適していない場合、受理率が低くなり、サンプリング効率が低下
- 異なるスケールのパラメータに対して、固定の提案分布では効率的に探索できない

Adaptive Metropolisは、**バーンイン期間中に提案分布の共分散行列を適応的に更新**することで、これらの問題を解決します。

## Algorithm Details

### Adaptive Update of Proposal Distribution

Adaptive Metropolisでは、提案分布として多変量正規分布を使用します：

$$
q(x'|x_t) = \mathcal{N}(x_t, C_t)
$$

ここで、$C_t$ は反復 $t$ における共分散行列です。

### Covariance Matrix Update Formula

Haario et al. (2001)のオリジナル手法では、共分散行列を以下のように更新します：

$$
C_{t+1} = s_d \left( \varepsilon I_d + \frac{1}{t} \sum_{i=1}^{t} (x_i - \bar{x}_t)(x_i - \bar{x}_t)^T \right)
$$

ここで：
- $s_d$ はスケーリングパラメータ（通常 $s_d = \frac{2.38^2}{d}$、$d$ は次元数）
- $\varepsilon > 0$ は小さな正定数（数値安定性のため）
- $I_d$ は $d \times d$ 単位行列
- $\bar{x}_t = \frac{1}{t} \sum_{i=1}^{t} x_i$ は現在までのサンプルの平均

### Efficient Update (Online Update)

共分散行列を毎回再計算するのではなく、以下の再帰的更新式を使用することで計算コストを削減できます：

$$
\bar{x}_{t+1} = \bar{x}_t + \frac{1}{t+1}(x_{t+1} - \bar{x}_t)
$$

$$
C_{t+1} = \frac{t-1}{t} C_t + \frac{s_d}{t} \left( t \bar{x}_t \bar{x}_t^T - (t+1) \bar{x}_{t+1} \bar{x}_{t+1}^T + x_{t+1} x_{t+1}^T + \varepsilon I_d \right)
$$

### Scaling Parameter

スケーリングパラメータ $s_d$ は、最適な受理率（通常0.234）を達成するために調整されます。Roberts et al. (1997)の理論的結果に基づき、高次元では：

$$
s_d = \frac{2.38^2}{d}
$$

が推奨されます。ただし、実際の適用では、目標分布の形状に応じて調整が必要な場合があります。

## Algorithm Procedure

1. **初期化**:
   - 初期状態 $x_0$ を設定
   - 初期共分散行列 $C_0$ を設定（通常は単位行列のスケール版）
   - サンプル平均 $\bar{x}_0 = x_0$ を初期化

2. **適応期間**（$t = 1, 2, \ldots, t_{adapt}$）:
   - 現在の共分散行列 $C_t$ を用いて提案分布から候補 $x' \sim \mathcal{N}(x_t, C_t)$ を生成
   - 受理確率 $\alpha(x_t, x')$ を計算
   - 確率 $\alpha$ で $x_{t+1} = x'$、そうでなければ $x_{t+1} = x_t$
   - 共分散行列 $C_{t+1}$ を更新
   - サンプル平均 $\bar{x}_{t+1}$ を更新

3. **固定期間**（$t > t_{adapt}$）:
   - 共分散行列を固定し、通常のMetropolis-Hastingsアルゴリズムを実行

## Convergence and Theoretical Guarantees

### Ergodicity

適応的アルゴリズムでは、提案分布が変化し続けるため、Markov性が失われる可能性があります。しかし、以下の条件を満たせば、エルゴード性が保証されます：

1. **有界性**: 共分散行列の固有値が有界である
2. **適応の停止**: 適応期間が有限である（$t_{adapt} < \infty$）

### Weak Ergodicity

適応期間中も共分散行列を更新し続ける場合、**弱いエルゴード性**（weak ergodicity）が保証されます。これは、適応が十分に遅い場合（例：$C_t$ の更新頻度が $t$ に対して十分に低い）に成立します。

## Advantages and Disadvantages

### Advantages

1. **高いサンプリング効率**: 適応的な提案分布により、受理率が最適に近づく
2. **次元スケーリング**: 異なるスケールのパラメータに対して自動的に調整
3. **実装の容易さ**: 比較的シンプルな実装で効果的な改善が得られる
4. **事前知識が不要**: 目標分布の形状に関する事前知識が少なくて済む

### Disadvantages

1. **バーンイン期間の必要性**: 適応期間中は、共分散行列が収束するまでサンプルを破棄する必要がある
2. **計算コスト**: 共分散行列の更新に計算コストがかかる（ただし、オンライン更新により軽減可能）
3. **高次元での性能**: 非常に高次元（$d > 100$）では、共分散行列の推定が困難になる場合がある

## Parameter Selection

### Scaling Parameter $s_d$

- **推奨値**: $s_d = \frac{2.38^2}{d}$（Roberts et al., 1997）
- **調整**: 受理率が0.2-0.4の範囲になるように調整

### Adaptation Period $t_{adapt}$

- **推奨値**: 総反復回数の10-50%
- **目安**: 共分散行列が安定するまで（通常、数千から数万回）

### Regularization Parameter $\varepsilon$

- **目的**: 数値安定性と正定値性の保証
- **推奨値**: $10^{-6}$ から $10^{-3}$ の範囲

## Implementation Considerations

1. **共分散行列の正定値性**: 更新後の共分散行列が正定値であることを確認
2. **数値安定性**: 共分散行列の計算における数値誤差に注意
3. **メモリ効率**: オンライン更新を使用してメモリ使用量を削減
4. **並列化**: 各サンプルの生成は独立であるため、並列化が可能

## Applications

- **ベイジアン推論**: 事後分布からのサンプリング
- **統計的モデリング**: 複雑なモデルのパラメータ推定
- **機械学習**: ベイジアンニューラルネットワークの学習
- **物理シミュレーション**: 統計力学系のサンプリング

## References

1. Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. *Bernoulli*, 7(2), 223-242.

2. Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*, 7(1), 110-120.

3. Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC. *Statistics and Computing*, 18(4), 343-373.

4. Rosenthal, J. S. (2011). Optimal proposal distributions and adaptive MCMC. *Handbook of Markov Chain Monte Carlo*, 93-112.
