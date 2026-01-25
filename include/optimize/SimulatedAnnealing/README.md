# 焼きなまし法 (Simulated Annealing)

## Overview

焼きなまし法（Simulated Annealing, SA）は、金属の焼きなまし（annealing）過程にヒントを得た確率的な最適化アルゴリズムです。Kirkpatrick et al. (1983)によって提案され、組み合わせ最適化問題や連続最適化問題に対して、局所最適解から脱出して大域最適解に近づくことを目的としています。

焼きなまし法は、初期には高い「温度」で広く探索し、徐々に温度を下げる（冷却）ことで、最終的に最適解に収束させる手法です。この温度パラメータにより、悪い解も一定の確率で受理することで、局所最適解から脱出できる可能性があります。

## Theoretical Background

### Physical Analogy: Annealing Process

焼きなまし法は、金属工学における**焼きなまし（annealing）** 過程に基づいています。金属を高温から徐々に冷却することで、結晶構造が最安定状態（エネルギー最小状態）に到達します。

**物理的過程**：
1. **高温状態**: 原子は激しく動き、高いエネルギー状態を取る
2. **冷却過程**: 温度を徐々に下げることで、原子の動きが制限される
3. **平衡状態**: 十分に冷却されると、原子は低エネルギー状態（最安定状態）に到達

**最適化への対応**：
- **エネルギー**: 目的関数の値 $f(\mathbf{x})$
- **状態**: パラメータベクトル $\mathbf{x}$
- **温度**: 制御パラメータ $T$
- **最安定状態**: 大域最適解 $\mathbf{x}^*$

### Metropolis Algorithm

焼きなまし法は、**メトロポリスアルゴリズム（Metropolis Algorithm）** を基盤としています。メトロポリスアルゴリズムは、統計力学における平衡分布からのサンプリング手法です。

**メトロポリスアルゴリズム**：

現在の状態 $\mathbf{x}_t$ から、提案分布 $q(\mathbf{x}' | \mathbf{x}_t)$ を用いて候補状態 $\mathbf{x}'$ を生成します。この候補を受理する確率は：

$$P_{\text{accept}} = \min\left(1, \exp\left(-\frac{\Delta E}{k_B T}\right)\right)$$

ここで：
- $\Delta E = f(\mathbf{x}') - f(\mathbf{x}_t)$: エネルギー差（目的関数の差）
- $k_B$: ボルツマン定数（通常、$k_B = 1$ として正規化）
- $T$: 温度パラメータ

**重要な性質**：
- $\Delta E < 0$（改善）: 常に受理（$P_{\text{accept}} = 1$）
- $\Delta E > 0$（悪化）: 確率 $\exp(-\Delta E / T)$ で受理
- 温度 $T$ が高い: 悪い解も高い確率で受理（広い探索）
- 温度 $T$ が低い: 悪い解は低い確率で受理（局所探索）

### Connection to Markov Chain Monte Carlo

焼きなまし法は、**マルコフ連鎖モンテカルロ法（Markov Chain Monte Carlo, MCMC）** の一種として理解できます。各反復で、現在の状態から次の状態への遷移が行われ、この遷移はマルコフ性（現在の状態のみに依存）を持ちます。

**平衡分布**：

温度 $T$ が固定されている場合、平衡分布は**ボルツマン分布（Boltzmann distribution）** に従います：

$$p(\mathbf{x}) \propto \exp\left(-\frac{f(\mathbf{x})}{T}\right)$$

この分布は、目的関数の値が小さい（エネルギーが低い）状態ほど高い確率で出現することを意味します。

**温度の効果**：
- **高温（$T \to \infty$）**: すべての状態がほぼ等確率（一様分布に近い）
- **低温（$T \to 0$）**: 最適解付近の状態が高い確率で出現（デルタ分布に近い）

## Algorithm Details

### Basic Simulated Annealing Algorithm

焼きなまし法の基本的なアルゴリズムは以下の通りです：

1. **初期化**:
   - 初期状態 $\mathbf{x}_0$ を設定
   - 初期温度 $T_0$ を設定
   - 冷却スケジュールを設定

2. **反復**（各反復 $t = 1, 2, \ldots$）:
   - 現在の状態 $\mathbf{x}_t$ の近傍から候補状態 $\mathbf{x}'$ を生成
   - エネルギー差 $\Delta E = f(\mathbf{x}') - f(\mathbf{x}_t)$ を計算
   - 受理確率 $P_{\text{accept}} = \min(1, \exp(-\Delta E / T_t))$ を計算
   - 確率 $P_{\text{accept}}$ で $\mathbf{x}_{t+1} = \mathbf{x}'$、そうでなければ $\mathbf{x}_{t+1} = \mathbf{x}_t$
   - 温度を更新: $T_{t+1} = \text{cooling}(T_t)$

3. **終了条件**:
   - 温度が十分に低くなった（$T_t < T_{\min}$）
   - 最大反復回数に達した
   - 収束判定（一定回数改善がない）

### Proposal Distribution (Neighborhood Function)

候補状態 $\mathbf{x}'$ の生成方法は、問題の種類によって異なります：

#### 連続最適化問題

**ガウス分布による提案**：

$$\mathbf{x}' = \mathbf{x}_t + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$

ここで $\sigma$ はステップサイズパラメータです。通常、温度に応じて調整されます：

$$\sigma_t = \sigma_0 \sqrt{T_t}$$

**一様分布による提案**：

$$\mathbf{x}' = \mathbf{x}_t + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \text{Uniform}([-\delta, \delta]^d)$$

ここで $\delta$ はステップサイズ、$d$ は次元数です。

#### 組み合わせ最適化問題

**近傍操作（neighborhood operator）**：

組み合わせ最適化問題では、現在の解から「近傍」の解を生成する操作を定義します。例：
- **巡回セールスマン問題**: 2つの都市を入れ替える、経路の一部を反転する
- **ナップサック問題**: アイテムの追加・削除
- **グラフ彩色問題**: 1つの頂点の色を変更

### Cooling Schedule (Temperature Schedule)

**冷却スケジュール（cooling schedule）** は、温度をどのように下げるかを決定します。適切な冷却スケジュールの選択が、アルゴリズムの性能に大きく影響します。

#### 1. 線形冷却（Linear Cooling）

$$T_{t+1} = T_t - \alpha$$

ここで $\alpha > 0$ は冷却率です。シンプルですが、収束が遅い場合があります。

#### 2. 指数冷却（Exponential Cooling）

$$T_{t+1} = \alpha T_t$$

ここで $0 < \alpha < 1$ は冷却率です。最も一般的な冷却スケジュールです。

**推奨値**: $\alpha = 0.85$ から $0.99$ の範囲

#### 3. 対数冷却（Logarithmic Cooling）

$$T_t = \frac{T_0}{\log(t + 1)}$$

理論的には最適な冷却スケジュールですが、実用的には収束が遅すぎる場合があります。

#### 4. 幾何冷却（Geometric Cooling）

$$T_t = T_0 \alpha^t$$

指数冷却と同様ですが、反復回数 $t$ を指数として使用します。

#### 5. 適応的冷却（Adaptive Cooling）

温度を固定し、一定回数の反復で受理率を監視します。受理率が目標値（例：0.4-0.6）より低い場合、温度を下げます。

### Acceptance Probability

受理確率は、以下のように定義されます：

$$P_{\text{accept}}(\mathbf{x}_t, \mathbf{x}', T_t) = \begin{cases}
1 & \text{if } f(\mathbf{x}') \leq f(\mathbf{x}_t) \\
\exp\left(-\frac{f(\mathbf{x}') - f(\mathbf{x}_t)}{T_t}\right) & \text{if } f(\mathbf{x}') > f(\mathbf{x}_t)
\end{cases}$$

**最小化問題の場合**：
- $f(\mathbf{x}') < f(\mathbf{x}_t)$（改善）: 常に受理
- $f(\mathbf{x}') > f(\mathbf{x}_t)$（悪化）: 確率 $\exp(-\Delta f / T_t)$ で受理

**最大化問題の場合**：
目的関数を $-f(\mathbf{x})$ に変換するか、受理確率を以下のように変更します：

$$P_{\text{accept}} = \min\left(1, \exp\left(\frac{f(\mathbf{x}') - f(\mathbf{x}_t)}{T_t}\right)\right)$$

## Algorithm Procedure

### Detailed Algorithm Steps

1. **初期化**:
   - 初期解 $\mathbf{x}_0$ を設定（ランダムまたはヒューリスティック）
   - 初期温度 $T_0$ を設定
   - 冷却率 $\alpha$ を設定
   - 最小温度 $T_{\min}$ を設定
   - 最大反復回数 $N_{\max}$ を設定
   - 各温度での反復回数 $L$ を設定（マルコフ連鎖の長さ）
   - 最良解 $\mathbf{x}_{\text{best}} = \mathbf{x}_0$ を記録

2. **外側ループ**（温度ループ、$k = 0, 1, 2, \ldots$）:
   - 現在の温度 $T_k$ で以下を実行
   
3. **内側ループ**（各温度での反復、$t = 1, 2, \ldots, L$）:
   - 現在の解 $\mathbf{x}_t$ の近傍から候補解 $\mathbf{x}'$ を生成
   - 目的関数の差 $\Delta f = f(\mathbf{x}') - f(\mathbf{x}_t)$ を計算
   - 受理確率 $P = \min(1, \exp(-\Delta f / T_k))$ を計算
   - 一様乱数 $u \sim \text{Uniform}(0, 1)$ を生成
   - もし $u < P$ なら:
     - $\mathbf{x}_{t+1} = \mathbf{x}'$（受理）
     - もし $f(\mathbf{x}') < f(\mathbf{x}_{\text{best}})$ なら:
       - $\mathbf{x}_{\text{best}} = \mathbf{x}'$（最良解を更新）
   - そうでなければ:
     - $\mathbf{x}_{t+1} = \mathbf{x}_t$（棄却）

4. **温度更新**:
   - $T_{k+1} = \alpha T_k$（指数冷却の場合）
   - もし $T_{k+1} < T_{\min}$ なら終了

5. **終了判定**:
   - 温度が最小温度以下になった
   - 最大反復回数に達した
   - 収束判定（一定回数改善がない）

6. **結果の返却**:
   - 最良解 $\mathbf{x}_{\text{best}}$ を返す

### Pseudocode

```
function SimulatedAnnealing(f, x0, T0, α, Tmin, L, Nmax):
    x_current = x0
    x_best = x0
    f_best = f(x0)
    T = T0
    k = 0
    
    while T > Tmin and k < Nmax:
        for t = 1 to L:
            x_candidate = generate_neighbor(x_current)
            Δf = f(x_candidate) - f(x_current)
            
            if Δf < 0:  // 改善
                x_current = x_candidate
                if f(x_candidate) < f_best:
                    x_best = x_candidate
                    f_best = f(x_candidate)
            else:  // 悪化
                P = exp(-Δf / T)
                u = random_uniform(0, 1)
                if u < P:
                    x_current = x_candidate
        
        T = α * T  // 温度を下げる
        k = k + 1
    
    return x_best
```

## Convergence and Theoretical Guarantees

### Asymptotic Convergence

焼きなまし法の理論的な収束性は、冷却スケジュールに大きく依存します。

#### 必要条件

**対数冷却スケジュール**：

$$T_t = \frac{T_0}{\log(t + 1)}$$

この冷却スケジュールを使用すると、**大域最適解に確率1で収束**することが理論的に保証されます（Hajek, 1988）。

**証明の概要**：

1. **マルコフ連鎖のエルゴード性**: 各温度でマルコフ連鎖がエルゴード的である
2. **平衡分布への収束**: 各温度で平衡分布（ボルツマン分布）に収束
3. **温度の減少**: 温度が十分に遅く減少する場合、平衡分布が最適解に集中
4. **大域収束**: 対数冷却により、すべての状態から最適解に到達可能

#### 実用的な冷却スケジュール

実用的には、対数冷却は収束が遅すぎるため、**指数冷却**が広く使用されます：

$$T_t = T_0 \alpha^t, \quad 0 < \alpha < 1$$

指数冷却では、理論的な大域収束は保証されませんが、多くの実用的な問題で良好な結果が得られます。

### Convergence Rate

収束速度は、以下の要因に依存します：

1. **初期温度 $T_0$**: 高いほど広く探索できるが、収束が遅い
2. **冷却率 $\alpha$**: 小さいほど速く冷却するが、局所最適解に陥りやすい
3. **マルコフ連鎖の長さ $L$**: 長いほど各温度で十分に探索できるが、計算コストが増加
4. **提案分布**: 適切な近傍関数が収束を加速

### Local vs. Global Optima

焼きなまし法の重要な特徴は、**局所最適解から脱出できる**ことです。これは、温度が高い間は悪い解も受理するためです。

**局所最適解からの脱出**：

温度 $T$ が高い場合、エネルギー差 $\Delta E$ が大きい解でも、確率 $\exp(-\Delta E / T)$ で受理されます。これにより、局所最適解の「谷」から脱出できる可能性があります。

**温度の役割**：
- **高温**: 広い探索、局所最適解からの脱出が容易
- **低温**: 局所探索、最適解付近での微調整

## Advantages and Disadvantages

### Advantages

1. **大域最適解への収束可能性**: 適切な冷却スケジュールにより、理論的に大域最適解に収束可能
2. **局所最適解からの脱出**: 温度パラメータにより、局所最適解から脱出できる
3. **汎用性**: 連続最適化、組み合わせ最適化の両方に適用可能
4. **勾配情報が不要**: 目的関数の値のみが必要で、勾配やヘッセ行列は不要
5. **実装の容易さ**: 比較的シンプルなアルゴリズムで実装可能
6. **制約条件の扱い**: 制約条件を受理確率に組み込むことで、制約付き最適化にも適用可能

### Disadvantages

1. **計算コスト**: 多くの反復が必要で、計算コストが高い場合がある
2. **パラメータの調整**: 初期温度、冷却率、マルコフ連鎖の長さなどのパラメータの調整が必要
3. **収束の保証がない**: 実用的な冷却スケジュール（指数冷却）では、大域収束が保証されない
4. **収束速度**: 最適解への収束が遅い場合がある
5. **確率的な性質**: 実行ごとに異なる結果が得られる可能性がある
6. **高次元問題**: 次元が高い場合、探索空間が広すぎて効率が低下する可能性がある

## Parameter Selection

### Initial Temperature $T_0$

初期温度は、アルゴリズムの性能に大きく影響します。

#### 経験的な方法

**受理率に基づく方法**：

1. ランダムな状態遷移を多数（例：100-1000回）実行
2. 悪化した遷移の平均エネルギー差 $\overline{\Delta E_+}$ を計算
3. 初期温度を以下のように設定：

$$T_0 = \frac{\overline{\Delta E_+}}{\log(P_0^{-1})}$$

ここで $P_0$ は目標受理率（通常、0.8-0.95）です。

**推奨値**: $P_0 = 0.8$ の場合、$T_0 \approx 4.5 \overline{\Delta E_+}$

#### 目的関数の範囲に基づく方法

目的関数の値の範囲 $[f_{\min}, f_{\max}]$ が既知の場合：

$$T_0 = f_{\max} - f_{\min}$$

または：

$$T_0 = 10 \times |f(\mathbf{x}_0)|$$

#### 推奨値

- **小規模問題**: $T_0 = 100$ から $1000$
- **中規模問題**: $T_0 = 1000$ から $10000$
- **大規模問題**: $T_0 = 10000$ から $100000$

### Cooling Rate $\alpha$

冷却率は、温度をどの程度速く下げるかを制御します。

**推奨値**:
- **速い冷却**: $\alpha = 0.80$ から $0.90$（収束が速いが、局所最適解に陥りやすい）
- **中程度の冷却**: $\alpha = 0.90$ から $0.95$（バランスが良い）
- **遅い冷却**: $\alpha = 0.95$ から $0.99$（大域最適解に近づきやすいが、収束が遅い）

**一般的な推奨値**: $\alpha = 0.85$ から $0.95$

### Minimum Temperature $T_{\min}$

最小温度は、アルゴリズムの終了条件として使用されます。

**推奨値**:
- $T_{\min} = 10^{-3}$ から $10^{-6}$
- または、$T_{\min} = T_0 \times 10^{-6}$

### Markov Chain Length $L$

各温度での反復回数（マルコフ連鎖の長さ）は、各温度で十分に探索するために重要です。

**固定長**:
- **小規模問題**: $L = 100$ から $1000$
- **中規模問題**: $L = 1000$ から $10000$
- **大規模問題**: $L = 10000$ から $100000$

**適応的長さ**:
- 各温度で、一定回数（例：10回）の受理が発生するまで反復
- または、受理率が目標値（例：0.4-0.6）に達するまで反復

### Step Size (for Continuous Optimization)

連続最適化問題では、提案分布のステップサイズも重要です。

**推奨値**:
- **固定ステップサイズ**: パラメータの範囲の1-10%
- **温度依存ステップサイズ**: $\sigma_t = \sigma_0 \sqrt{T_t / T_0}$

### Parameter Tuning Strategy

1. **粗い調整**: まず広い範囲でパラメータを試す
2. **細かい調整**: 良好な結果が得られた範囲で細かく調整
3. **交差検証**: 複数の問題インスタンスで性能を評価
4. **適応的パラメータ**: アルゴリズムの実行中にパラメータを調整

## Implementation Considerations

### Numerical Stability

**指数関数のオーバーフロー**：

受理確率の計算で、$\exp(-\Delta f / T)$ が数値的に不安定になる場合があります。

**対処法**:
- **対数空間での計算**: $\log P = -\Delta f / T$ を計算し、$P = \exp(\log P)$ を計算（ただし、$\log P < 0$ の場合のみ）
- **クリッピング**: $\Delta f / T$ が大きすぎる場合、$P = 0$ と設定

**実装例**:
```cpp
double acceptance_probability = 1.0;
if (delta_f > 0) {
    double exponent = -delta_f / temperature;
    if (exponent > -50) {  // オーバーフローを防ぐ
        acceptance_probability = std::exp(exponent);
    } else {
        acceptance_probability = 0.0;
    }
}
```

### Efficient Random Number Generation

乱数生成は、アルゴリズムの計算コストの大部分を占める可能性があります。

**推奨事項**:
- 高品質な乱数生成器を使用（例：Mersenne Twister）
- 乱数生成器を適切に初期化
- 並列化する場合、各スレッドで独立な乱数生成器を使用

### Memory Management

大規模問題では、メモリ使用量に注意が必要です。

**考慮事項**:
- 現在の解と最良解のみを保持
- 提案分布の生成に必要なメモリを最小化
- 必要に応じて、解の履歴を保存（デバッグ用）

### Parallelization

焼きなまし法は、本質的に逐次的なアルゴリズムですが、以下の方法で並列化できます：

1. **独立実行**: 複数の独立な実行を並列に実行し、最良解を選択
2. **並列近傍探索**: 複数の候補解を並列に生成・評価
3. **複数温度**: 異なる温度で複数のマルコフ連鎖を並列実行

### Restart Strategy

局所最適解に陥った場合、**再起動（restart）** 戦略が有効です。

**方法**:
- 一定回数改善がない場合、温度を上げる（reheating）
- または、新しい初期解から再開

### Adaptive Parameters

実行中にパラメータを適応的に調整することで、性能を向上できます。

**適応的冷却**:
- 受理率を監視し、目標値（例：0.4-0.6）より低い場合、冷却を遅くする
- 改善が続く場合、温度を維持または上げる

**適応的ステップサイズ**:
- 受理率が高い場合、ステップサイズを増やす
- 受理率が低い場合、ステップサイズを減らす

## Variants and Extensions

### Fast Simulated Annealing

**高速焼きなまし法（Fast Simulated Annealing）** は、冷却スケジュールを改良した手法です。

**冷却スケジュール**:
$$T_t = \frac{T_0}{1 + t}$$

この冷却スケジュールは、対数冷却よりも速く、指数冷却よりも理論的な保証があります。

### Adaptive Simulated Annealing (ASA)

**適応的焼きなまし法（Adaptive Simulated Annealing）** は、各次元に異なる温度パラメータを使用する手法です。

**特徴**:
- 各次元 $i$ に温度 $T_i$ を割り当て
- 次元ごとに異なる冷却スケジュールを適用
- 高次元問題で有効

### Quantum Annealing

**量子焼きなまし法（Quantum Annealing）** は、量子効果を利用した最適化手法です。D-Waveなどの量子コンピュータで実装されています。

### Hybrid Methods

焼きなまし法を他の最適化手法と組み合わせることで、性能を向上できます。

**例**:
- **焼きなまし法 + 局所探索**: 各温度で局所最適化を実行
- **焼きなまし法 + 遺伝的アルゴリズム**: 集団ベースの探索と組み合わせ
- **焼きなまし法 + タブーサーチ**: タブーリストと組み合わせて探索を改善

## Applications

焼きなまし法は、以下のような問題に適用されています：

### 1. 組み合わせ最適化問題

- **巡回セールスマン問題（TSP）**: 都市間の最短経路を見つける
- **ナップサック問題**: 制約条件下で価値を最大化
- **グラフ彩色問題**: 最小色数でグラフを彩色
- **スケジューリング問題**: ジョブの最適なスケジュールを見つける
- **レイアウト問題**: VLSI設計、施設配置など

### 2. 連続最適化問題

- **非凸関数の最適化**: 多峰性関数の大域最適解を見つける
- **機械学習**: ハイパーパラメータ最適化、ニューラルネットワークの学習
- **信号処理**: フィルタ設計、パラメータ推定
- **制御システム**: 最適制御問題

### 3. 統計的推論

- **ベイジアン推論**: 事後分布からのサンプリング（MCMCの一種）
- **モデル選択**: 最適なモデル構造を見つける
- **ハイパーパラメータ最適化**: ガウス過程回帰などのハイパーパラメータを最適化

### 4. 画像処理

- **画像復元**: ノイズ除去、デブラー
- **画像セグメンテーション**: 領域分割
- **ステレオビジョン**: 視差マップの推定

### 5. 物理シミュレーション

- **分子動力学**: タンパク質の構造予測
- **スピングラスモデル**: 統計力学モデルの最適化
- **材料設計**: 最適な材料組成を見つける

## Comparison with Other Optimization Methods

### vs. Gradient-Based Methods

**勾配ベース手法（例：勾配降下法、ニュートン法）**:
- **利点**: 収束が速い、局所最適解への収束が保証される
- **欠点**: 勾配が必要、局所最適解に陥りやすい、非凸問題には不適切

**焼きなまし法**:
- **利点**: 勾配が不要、局所最適解から脱出可能、非凸問題に適用可能
- **欠点**: 収束が遅い、確率的な性質

### vs. Genetic Algorithms

**遺伝的アルゴリズム（GA）**:
- **利点**: 集団ベースの探索、多様性が高い
- **欠点**: パラメータが多い、収束が遅い場合がある

**焼きなまし法**:
- **利点**: パラメータが少ない、実装が簡単
- **欠点**: 単一解の探索、多様性が低い

### vs. Particle Swarm Optimization

**粒子群最適化（PSO）**:
- **利点**: 集団ベース、収束が速い場合がある
- **欠点**: パラメータの調整が必要、局所最適解に陥りやすい

**焼きなまし法**:
- **利点**: 理論的な保証、パラメータが少ない
- **欠点**: 収束が遅い、単一解の探索

## Best Practices

1. **適切な初期温度の選択**: 受理率に基づいて初期温度を設定
2. **冷却スケジュールの調整**: 問題に応じて冷却率を調整
3. **マルコフ連鎖の長さ**: 各温度で十分に探索するように設定
4. **複数回の実行**: 確率的な性質のため、複数回実行して最良解を選択
5. **ヒューリスティックな初期解**: ランダムな初期解ではなく、ヒューリスティックな初期解を使用
6. **適応的パラメータ**: 実行中にパラメータを適応的に調整
7. **並列化**: 可能な場合、独立実行を並列化
8. **可視化**: 温度、目的関数値、受理率などを可視化してデバッグ

## References

1. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.

2. Černý, V. (1985). Thermodynamical approach to the traveling salesman problem: An efficient simulation algorithm. *Journal of Optimization Theory and Applications*, 45(1), 41-51.

3. Hajek, B. (1988). Cooling schedules for optimal annealing. *Mathematics of Operations Research*, 13(2), 311-329.

4. van Laarhoven, P. J., & Aarts, E. H. (1987). *Simulated Annealing: Theory and Applications*. Springer.

5. Henderson, D., Jacobson, S. H., & Johnson, A. W. (2003). The theory and practice of simulated annealing. In *Handbook of Metaheuristics* (pp. 287-319). Springer.

6. Ingber, L. (1993). Simulated annealing: Practice versus theory. *Mathematical and Computer Modelling*, 18(11), 29-57.

7. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics*, 21(6), 1087-1092.
