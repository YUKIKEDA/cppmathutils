# ガウス過程回帰 (Gaussian Process Regression)

## Overview

ガウス過程回帰（Gaussian Process Regression, GPR）は、非線形回帰問題に対して柔軟なモデルを提供するベイジアン機械学習手法です。ガウス過程は、任意の有限個の入力点に対する出力の同時分布が多変量正規分布に従う確率過程として定義されます。この性質により、ガウス過程回帰は不確実性を自然に表現でき、予測の信頼区間を提供することができます。

ガウス過程回帰は、従来の基底関数による線形回帰モデルを拡張し、**次元の呪い（curse of dimensionality）** を回避しながら、高次元入力空間でも柔軟な非線形関数を表現できる手法として発展しました。

## What is Machine Learning?

機械学習（Machine Learning）は、データからパターンを学習し、新しいデータに対する予測や分類を行う技術です。回帰問題では、入力 $\mathbf{x}$ と出力 $y$ の関係を学習し、新しい入力に対する出力を予測します。

従来の線形回帰モデルは、入力と出力の関係が線形であることを仮定しますが、現実の多くの問題では非線形な関係が存在します。このような場合、基底関数を用いて非線形な関係を表現する必要があります。

## Review: From Simple to Multiple Linear Regression

### Simple Linear Regression

単回帰（Simple Linear Regression）は、1つの説明変数 $x$ を用いて目的変数 $y$ を予測するモデルです：

$$y = \beta_0 + \beta_1 x + \epsilon$$

ここで $\beta_0$ は切片、$\beta_1$ は回帰係数、$\epsilon$ は誤差項です。

### Multiple Linear Regression

重回帰（Multiple Linear Regression）は、複数の説明変数 $\mathbf{x} = (x_1, x_2, \ldots, x_p)^T$ を用いて目的変数 $y$ を予測するモデルです：

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon = \boldsymbol{\beta}^T\mathbf{x} + \epsilon$$

ここで $\boldsymbol{\beta} = (\beta_0, \beta_1, \ldots, \beta_p)^T$ はパラメータベクトル、$\mathbf{x} = (1, x_1, x_2, \ldots, x_p)^T$ は特徴ベクトルです。

### Linear Regression with Basis Functions

非線形な関係を表現するために、基底関数（basis functions）を用いた線形回帰モデルを考えます：

$$y = \sum_{j=0}^{p} w_j \phi_j(\mathbf{x}) = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})$$

ここで：
- $\boldsymbol{\phi}(\mathbf{x}) = (\phi_0(\mathbf{x}), \phi_1(\mathbf{x}), \ldots, \phi_p(\mathbf{x}))^T$: 特徴ベクトル（feature vector）
- $\mathbf{w} = (w_0, w_1, \ldots, w_p)^T$: パラメータベクトル（重みベクトル）
- $\phi_0(\mathbf{x}) = 1$: 切片項に対応する基底関数

この表現により、多項式、三角関数、指数関数など、様々な非線形関数を線形回帰の枠組みで扱うことができます。

## The Wall of Feature Engineering

基底関数による線形回帰モデルでは、適切な基底関数を選択する必要があります。しかし、複雑な非線形関係を表現するためには、多くの基底関数が必要になり、以下の問題が発生します：

1. **基底関数の選択**: どの基底関数を用いるべきか、その数はいくつか
2. **パラメータ数の増加**: 基底関数の数が増えると、推定すべきパラメータ数も増加
3. **過学習のリスク**: パラメータ数が多すぎると、訓練データに過度に適合し、汎化性能が低下

これらの問題は、**特徴量エンジニアリング（feature engineering）** の壁として知られています。

## Treating Basis Functions as Random Variables (Non-parametric Models)

従来の線形回帰モデルでは、基底関数の数と種類を事前に決定する必要がありました。しかし、ガウス過程回帰では、**基底関数を確率変数として捉え、パラメータを周辺化（marginalize）** することで、基底関数の数を明示的に指定する必要がなくなります。

このアプローチは、**ノンパラメトリックモデル（non-parametric model）** と呼ばれます。パラメトリックモデルでは、パラメータの数が固定されていますが、ノンパラメトリックモデルでは、データの量に応じてモデルの複雑さが自動的に調整されます。

> [!NOTE]
> **基底関数の決定方法：手動設計からデータ駆動学習へ**
>
> 基底関数をどうやって決めるかは、機械学習における重要な問題です。歴史的に、以下のようなアプローチが発展してきました：
>
> **1. 手動設計（Manual Design）**
>
> 従来の機械学習では、基底関数を**手動で設計**する必要がありました。例えば：
> - 多項式基底: $\phi_j(x) = x^j$（$j = 0, 1, 2, \ldots$）
> - 動径基底関数: $\phi_h(\mathbf{x}) = \exp(-(\mathbf{x} - \boldsymbol{\mu}_h)^2 / (2\sigma^2))$
> - 三角関数基底: $\phi_j(x) = \sin(jx), \cos(jx)$
>
> このアプローチでは、**特徴量エンジニアリング（feature engineering）** の専門知識が必要で、問題領域に応じて適切な基底関数を選択する必要があります。しかし、複雑な非線形関係を表現するには、多くの基底関数が必要になり、次元の呪いの問題に直面します。
>
> **2. 確率変数として扱う（ガウス過程）**
>
> ガウス過程回帰では、基底関数を**確率変数として扱い、パラメータを周辺化**することで、基底関数の数を明示的に指定する必要がなくなります。カーネル関数を通じて、無限次元の特徴空間を暗黙的に扱うことができます。
>
> このアプローチの特徴：
> - **パラメータフリー**: 基底関数の数を事前に決める必要がない
> - **ベイジアン推論**: 不確実性を自然に定量化できる
> - **柔軟性**: カーネル関数の選択により、様々な関数クラスを表現可能
> - **計算コスト**: $O(n^3)$ の計算量により、大規模データには不向き
>
> **3. データから学習する（特徴量表現学習）**
>
> 現代の機械学習、特に**深層学習（Deep Learning）**では、基底関数（特徴量）自体を**データから自動的に学習**します。多層ニューラルネットワークは、各層で特徴量を変換し、最終的にタスクに適した表現を学習します。
>
> 深層学習のアプローチ：
> - **表現学習（Representation Learning）**: データから有用な特徴量を自動的に抽出
> - **階層的特徴抽出**: 低層では単純な特徴（エッジなど）を、高層では複雑な特徴（物体など）を学習
> - **エンドツーエンド学習**: 特徴量抽出から予測まで、一貫して最適化
>
> 深層学習の特徴：
> - **スケーラビリティ**: 大規模データに対して効率的に学習可能
> - **汎用性**: 画像、音声、自然言語など、様々なデータタイプに対応
> - **表現力**: 非常に複雑な非線形関係を表現可能
> - **不確実性の定量化**: 従来は困難だったが、ベイジアンニューラルネットワークなどの手法で対応可能
>
> **各アプローチの比較**
>
> | アプローチ | 基底関数の決定 | 利点 | 欠点 |
> |-----------|--------------|------|------|
> | 手動設計 | 専門家が事前に設計 | 解釈しやすい、計算が高速 | 専門知識が必要、次元の呪い |
> | ガウス過程 | 確率変数として周辺化 | 不確実性の定量化、パラメータフリー | 計算量が $O(n^3)$、大規模データに不向き |
> | 深層学習 | データから自動学習 | 大規模データに適している、表現力が高い | 不確実性の定量化が困難（改善中）、解釈が困難 |
>
> **現代のトレンド**
>
> 近年では、これらのアプローチを組み合わせた手法も研究されています：
> - **深層ガウス過程（Deep Gaussian Processes）**: ガウス過程を多層に積み重ねたモデル
> - **ベイジアンニューラルネットワーク**: ニューラルネットワークのパラメータに事前分布を仮定
> - **ニューラルネットワークガウス過程**: 無限幅のニューラルネットワークがガウス過程に収束する性質を利用
>
> これらの手法により、深層学習の表現力とガウス過程の不確実性定量化の両方を活用できるようになっています。

## Linear Regression Models and the Curse of Dimensionality

### Radial Basis Function Regression

複雑な関数を表現するために、**動径基底関数（Radial Basis Function, RBF）** を用いた回帰モデルを考えます。特に、ガウス分布の形をした基底関数を用いる場合：

$$\phi_h(\mathbf{x}) = \exp\left(-\frac{(\mathbf{x} - \boldsymbol{\mu}_h)^2}{2\sigma^2}\right)$$

ここで：
- $\boldsymbol{\mu}_h$: 基底関数の中心（$h = -H, -H+1, \ldots, 0, \ldots, H-1, H$）
- $\sigma^2$: 基底関数の幅（分散パラメータ）

1次元入力の場合、中心を $\mu_h \in \{-H, -H+1, \ldots, 0, \ldots, H-1, H\}$ として配置すると、動径基底関数回帰モデルは以下のように表されます：

$$y = \sum_{h=-H}^{H} w_h \phi_h(x) = \sum_{h=-H}^{H} w_h \exp\left(-\frac{(x - \mu_h)^2}{2\sigma^2}\right)$$

このモデルは、十分な数の基底関数（$2H+1$ 個）を用いれば、任意の形状の関数を表現できることが知られています（普遍近似定理）。

### Curse of Dimensionality

動径基底関数回帰は、1次元や2次元の入力に対しては有効ですが、入力の次元が大きくなると深刻な問題に直面します。

**次元の呪いとは**

入力空間の次元が $d$ の場合、各次元に $M$ 個の基底関数の中心を配置すると、必要な基底関数の総数は $M^d$ 個になります。例えば：
- $d=1, M=10$: $10^1 = 10$ 個の基底関数
- $d=2, M=10$: $10^2 = 100$ 個の基底関数
- $d=10, M=10$: $10^{10} = 10,000,000,000$ 個の基底関数

このように、入力の次元が大きくなると、必要な基底関数の数が**指数的に増加**します。これを**次元の呪い（curse of dimensionality）** と呼びます。

**問題点**

1. **計算コスト**: 基底関数の数が指数的に増加するため、計算量が爆発的に増大します
2. **メモリ使用量**: 計画行列 $\mathbf{\Phi}$ のサイズが $n \times M^d$ となり、メモリ不足に陥ります
3. **過学習**: パラメータ数がサンプル数を超える可能性があり、過学習が発生します
4. **実用性の欠如**: 高次元データに対して実用的な手法とはなりえません

## Gaussian Process

### Extension from Bayesian Linear Regression

動径基底関数回帰モデルを再考します：

$$y = \sum_{h=-H}^{H} w_h \phi_h(\mathbf{x}) = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})$$

ここで、パラメータベクトル $\mathbf{w}$ に正規分布の事前分布を仮定します：

$$p(\mathbf{w} | \alpha) = \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$$

これは、リッジ回帰で学んだベイジアン線形回帰の設定と同じです。

### Marginalization of Parameters

ベイジアン線形回帰では、通常、事後分布 $p(\mathbf{w} | \mathbf{y}, \mathbf{\Phi})$ を計算し、MAP推定や事後分布の平均を用いて予測を行います。しかし、ガウス過程では、**パラメータ $\mathbf{w}$ を周辺化（marginalize out）** することで、パラメータに依存しない予測分布を直接得ます。

**周辺化とは**

**周辺化（marginalization）** は、確率変数の一部を積分によって消去し、残りの確率変数の分布を求める操作です。例えば、同時確率分布 $p(x, y)$ から $x$ を周辺化すると、$y$ の周辺分布 $p(y) = \int p(x, y) dx$ が得られます。

ベイジアン線形回帰では、パラメータ $\mathbf{w}$ と観測データ $\mathbf{y}$ の同時分布 $p(\mathbf{y}, \mathbf{w} | \mathbf{\Phi})$ から、$\mathbf{w}$ を周辺化することで、$\mathbf{y}$ の周辺分布 $p(\mathbf{y} | \mathbf{\Phi})$ を求めます：

$$p(\mathbf{y} | \mathbf{\Phi}) = \int p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}) p(\mathbf{w} | \alpha) d\mathbf{w}$$

この操作により、パラメータ $\mathbf{w}$ を明示的に推定する必要がなくなり、出力 $\mathbf{y}$ の分布を直接扱うことができます。

**無限次元パラメータ問題の解消**

従来の基底関数アプローチでは、無限次元の基底関数（例：無限次元の多項式基底）を使おうとすると、パラメータベクトル $\mathbf{w}$ も無限次元になり、以下の問題が発生します：

1. **推定不可能**: 無限次元のパラメータを有限個のデータから推定することは不可能
2. **計算不可能**: 無限次元のベクトルや行列を計算機で扱うことは不可能
3. **次元の呪い**: 基底関数の数が指数的に増加し、実用的でない

しかし、ガウス過程では、**パラメータ $\mathbf{w}$ を周辺化**することで、この問題を根本的に解決します：

- **パラメータの明示的な扱いが不要**: 無限次元のパラメータ $\mathbf{w}$ を明示的に扱う必要がなく、出力 $\mathbf{y}$ の分布を直接扱えます
- **カーネル関数による表現**: 無限次元の特徴空間での内積を、カーネル関数 $k(\mathbf{x}_i, \mathbf{x}_j)$ として表現できるため、無限次元のパラメータを扱う必要がありません
- **計算可能**: カーネル関数の評価は有限次元の入力空間で行われるため、計算機で扱えます

つまり、ガウス過程は、**無限次元の特徴空間を扱いながら、無限次元のパラメータを明示的に扱う必要がない**という、一見矛盾するように見える要求を満たしています。これが、ガウス過程が次元の呪いを回避できる根本的な理由です。

### Derivation of the Joint Distribution of Outputs

$n$ 個の観測データ $\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$ があるとします。計画行列を $\mathbf{\Phi} = (\boldsymbol{\phi}(\mathbf{x}_1), \boldsymbol{\phi}(\mathbf{x}_2), \ldots, \boldsymbol{\phi}(\mathbf{x}_n))^T$ とすると、出力ベクトルは：

$$\mathbf{y} = \mathbf{\Phi}\mathbf{w} + \boldsymbol{\epsilon}$$

ここで $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ は誤差項です。

パラメータ $\mathbf{w}$ の事前分布が $p(\mathbf{w} | \alpha) = \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$ であるとき、$\mathbf{w}$ を周辺化した出力 $\mathbf{y}$ の分布を求めます。

**導出：出力の同時分布**

$\mathbf{w}$ が与えられたときの $\mathbf{y}$ の条件付き分布は：

$$p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) = \mathcal{N}(\mathbf{\Phi}\mathbf{w}, \sigma^2\mathbf{I})$$

$\mathbf{w}$ の事前分布は：

$$p(\mathbf{w} | \alpha) = \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$$

$\mathbf{y}$ の周辺分布は、$\mathbf{w}$ について積分することで得られます：

$$p(\mathbf{y} | \mathbf{\Phi}, \sigma^2, \alpha) = \int p(\mathbf{y} | \mathbf{\Phi}, \mathbf{w}, \sigma^2) p(\mathbf{w} | \alpha) d\mathbf{w}$$

この積分を計算すると、$\mathbf{y}$ は以下の多変量正規分布に従うことが示されます：

$$p(\mathbf{y} | \mathbf{\Phi}, \sigma^2, \alpha) = \mathcal{N}(\mathbf{0}, \mathbf{C})$$

ここで、共分散行列 $\mathbf{C}$ は：

$$\mathbf{C} = \sigma^2\mathbf{I} + \alpha^{-1}\mathbf{\Phi}\mathbf{\Phi}^T$$

これは、正規分布の線形変換の性質から導かれます。$\mathbf{y} = \mathbf{\Phi}\mathbf{w} + \boldsymbol{\epsilon}$ において、$\mathbf{w}$ と $\boldsymbol{\epsilon}$ が独立な正規分布に従うとき、$\mathbf{y}$ も正規分布に従い、その平均と共分散は：

$$\mathbb{E}[\mathbf{y}] = \mathbf{\Phi}\mathbb{E}[\mathbf{w}] + \mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$$

$$\text{Cov}[\mathbf{y}] = \mathbf{\Phi}\text{Cov}[\mathbf{w}]\mathbf{\Phi}^T + \text{Cov}[\boldsymbol{\epsilon}] = \alpha^{-1}\mathbf{\Phi}\mathbf{\Phi}^T + \sigma^2\mathbf{I}$$

### Introduction of Kernel Functions

共分散行列 $\mathbf{C}$ の $(i, j)$ 要素は：

$$C_{ij} = \sigma^2\delta_{ij} + \alpha^{-1}\boldsymbol{\phi}(\mathbf{x}_i)^T\boldsymbol{\phi}(\mathbf{x}_j)$$

ここで $\delta_{ij}$ はクロネッカーのデルタ（$i=j$ のとき1、それ以外0）です。

重要な観察は、共分散行列の要素が**特徴ベクトルの内積** $\boldsymbol{\phi}(\mathbf{x}_i)^T\boldsymbol{\phi}(\mathbf{x}_j)$ に依存していることです。この内積を**カーネル関数（kernel function）** として定義します：

$$k(\mathbf{x}_i, \mathbf{x}_j) = \alpha^{-1}\boldsymbol{\phi}(\mathbf{x}_i)^T\boldsymbol{\phi}(\mathbf{x}_j)$$

すると、共分散行列は：

$$C_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) + \sigma^2\delta_{ij}$$

と表されます。

カーネル関数 $k(\mathbf{x}_i, \mathbf{x}_j)$ は、2つの入力点 $\mathbf{x}_i$ と $\mathbf{x}_j$ の「類似度」を測る関数です。特徴ベクトルの内積として定義されるため、入力が似ていれば（特徴ベクトルが似ていれば）、カーネル関数の値は大きくなります。

### Definition of Gaussian Process

これまでの導出から、重要な結論が得られます：

**任意の有限個の入力点の集合 $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$ に対して、対応する出力の同時分布 $p(y_1, y_2, \ldots, y_n)$ が多変量正規分布に従うとき、入力 $\mathbf{x}$ と出力 $y$ の関係は **ガウス過程（Gaussian Process）** であるといいます。**

より形式的には、ガウス過程は以下のように定義されます：

**定義：ガウス過程（Gaussian Process）**

確率過程 $\{f(\mathbf{x}) : \mathbf{x} \in \mathcal{X}\}$ がガウス過程であるとは、任意の有限個の入力点 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n \in \mathcal{X}$ に対して、確率変数のベクトル $(f(\mathbf{x}_1), f(\mathbf{x}_2), \ldots, f(\mathbf{x}_n))^T$ が多変量正規分布に従うことをいいます。

$$(f(\mathbf{x}_1), f(\mathbf{x}_2), \ldots, f(\mathbf{x}_n))^T \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K})$$

ここで：
- $\boldsymbol{\mu} = (\mu(\mathbf{x}_1), \mu(\mathbf{x}_2), \ldots, \mu(\mathbf{x}_n))^T$: 平均関数（mean function）
- $\mathbf{K}$: 共分散行列（covariance matrix）、その $(i, j)$ 要素は $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$

ガウス過程は、平均関数 $\mu(\mathbf{x})$ とカーネル関数（共分散関数）$k(\mathbf{x}_i, \mathbf{x}_j)$ によって完全に特徴づけられます。これを以下のように表記します：

$$f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}_i, \mathbf{x}_j))$$

> [!NOTE]
> **基底関数にガウス関数を用いる理由**
>
> 動径基底関数としてガウス関数を用いる理由は、以下の通りです：
>
> 1. **滑らかさ**: ガウス関数は無限回微分可能であり、滑らかな関数を表現できます
> 2. **局所性**: 中心からの距離に応じて値が減衰し、局所的な非線形関係をモデル化できます
> 3. **正定値性**: ガウス関数から構成されるカーネル関数は正定値であり、有効なカーネル関数となります
>
> **無限次元の多項式基底との等価性**
>
> RBFカーネル（ガウスカーネル）は、無限次元の多項式基底に対応していることが知られています。これは、RBFカーネルが無限次元の特徴空間での内積を、元の入力空間でのカーネル関数の評価に置き換えることができることを意味します。
>
> **証明：RBFカーネルと無限次元多項式基底の等価性**
>
> 1次元の場合を考えます。RBFカーネルは：
>
> $$k(x_i, x_j) = \sigma_f^2 \exp\left(-\frac{(x_i - x_j)^2}{2\ell^2}\right)$$
>
> これを展開するために、まず指数関数の性質を利用します：
>
> $$\exp\left(-\frac{(x_i - x_j)^2}{2\ell^2}\right) = \exp\left(-\frac{x_i^2}{2\ell^2}\right) \exp\left(-\frac{x_j^2}{2\ell^2}\right) \exp\left(\frac{x_i x_j}{\ell^2}\right)$$
>
> 最後の項 $\exp(x_i x_j / \ell^2)$ をテイラー展開します：
>
> $$\exp\left(\frac{x_i x_j}{\ell^2}\right) = \sum_{n=0}^{\infty} \frac{1}{n!} \left(\frac{x_i x_j}{\ell^2}\right)^n = \sum_{n=0}^{\infty} \frac{1}{n! \ell^{2n}} (x_i x_j)^n$$
>
> したがって、RBFカーネルは以下のように展開できます：
>
> $$k(x_i, x_j) = \sigma_f^2 \exp\left(-\frac{x_i^2}{2\ell^2}\right) \exp\left(-\frac{x_j^2}{2\ell^2}\right) \sum_{n=0}^{\infty} \frac{1}{n! \ell^{2n}} (x_i x_j)^n$$
>
> これを整理すると：
>
> $$k(x_i, x_j) = \sigma_f^2 \sum_{n=0}^{\infty} \frac{1}{n! \ell^{2n}} \left[\exp\left(-\frac{x_i^2}{2\ell^2}\right) x_i^n\right] \left[\exp\left(-\frac{x_j^2}{2\ell^2}\right) x_j^n\right]$$
>
> ここで、特徴関数を以下のように定義します：
>
> $$\phi_n(x) = \sqrt{\frac{\sigma_f^2}{n! \ell^{2n}}} \exp\left(-\frac{x^2}{2\ell^2}\right) x^n$$
>
> すると、RBFカーネルは：
>
> $$k(x_i, x_j) = \sum_{n=0}^{\infty} \phi_n(x_i) \phi_n(x_j)$$
>
> これは、**無限次元の特徴空間での内積**として表現されています。各 $\phi_n(x)$ は、$x^n$ にガウス重み関数 $\exp(-x^2/(2\ell^2))$ をかけたもので、実質的に多項式基底に対応しています。
>
> **一般次元への拡張**
>
> $d$ 次元入力の場合、RBFカーネルは：
>
> $$k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\ell^2}\right)$$
>
> 同様の展開により、以下のように表現できます：
>
> $$k(\mathbf{x}_i, \mathbf{x}_j) = \sum_{\mathbf{n} \in \mathbb{N}_0^d} \phi_{\mathbf{n}}(\mathbf{x}_i) \phi_{\mathbf{n}}(\mathbf{x}_j)$$
>
> ここで $\mathbf{n} = (n_1, n_2, \ldots, n_d)$ は多項式の次数を表す多重指数、$\phi_{\mathbf{n}}(\mathbf{x})$ は：
>
> $$\phi_{\mathbf{n}}(\mathbf{x}) = \sqrt{\frac{\sigma_f^2}{\mathbf{n}! \ell^{2|\mathbf{n}|}}} \exp\left(-\frac{\|\mathbf{x}\|^2}{2\ell^2}\right) \mathbf{x}^{\mathbf{n}}$$
>
> ここで $\mathbf{n}! = n_1! n_2! \cdots n_d!$、$|\mathbf{n}| = n_1 + n_2 + \cdots + n_d$、$\mathbf{x}^{\mathbf{n}} = x_1^{n_1} x_2^{n_2} \cdots x_d^{n_d}$ です。
>
> **マーサーの定理との関係**
>
> この結果は、**マーサーの定理（Mercer's theorem）** の具体例です。マーサーの定理によると、正定値カーネル $k(\mathbf{x}_i, \mathbf{x}_j)$ は、適切な特徴関数 $\{\phi_n(\mathbf{x})\}_{n=0}^{\infty}$ を用いて：
>
> $$k(\mathbf{x}_i, \mathbf{x}_j) = \sum_{n=0}^{\infty} \lambda_n \phi_n(\mathbf{x}_i) \phi_n(\mathbf{x}_j)$$
>
> と表現できます。RBFカーネルの場合、固有値は $\lambda_n = \sigma_f^2 / (n! \ell^{2n})$ に対応し、特徴関数は多項式基底にガウス重みをかけたものになります。
>
> **意味**
>
> この結果により、RBFカーネルを使用することは、**無限次元の多項式特徴空間**で線形回帰を行うことと数学的に等価であることが示されます。しかし、カーネルトリックにより、この無限次元空間を明示的に計算する必要がなく、元の入力空間でのカーネル関数の評価だけで済みます。これが、次元の呪いを回避できる理由です。

### Kernel Trick

**カーネルトリック（kernel trick）** は、特徴ベクトルの内積を直接計算せずに、カーネル関数を通じて計算する手法です。これにより、高次元（または無限次元）の特徴空間での計算を、元の入力空間での計算に置き換えることができます。

**メリット**

1. **計算効率**: 高次元特徴空間での内積計算を、低次元入力空間でのカーネル関数の評価に置き換えられます
2. **無限次元特徴空間**: 無限次元の特徴空間に対応できるカーネル関数（例：RBFカーネル）を使用できます
3. **柔軟性**: 特徴ベクトルを明示的に定義せずに、カーネル関数を直接設計できます

**カーネル関数の例**

#### RBF Kernel (Gaussian Kernel)

$$k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\ell^2}\right)$$

ここで：
- $\sigma_f^2$: 信号分散（signal variance）
- $\ell$: 長さスケール（length scale）

このカーネルは、無限次元の特徴空間に対応しています。長さスケール $\ell$ は、入力空間における「類似性」の範囲を制御します：
- $\ell$ が大きい: 遠く離れた入力でも出力が似る傾向がある（滑らかな関数）
- $\ell$ が小さい: 近い入力でのみ出力が似る傾向がある（急激に変化する関数）

#### Polynomial Kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T\mathbf{x}_j + c)^d$$

ここで $c \geq 0$ は定数、$d$ は次数です。このカーネルは、$d$ 次までの多項式特徴空間に対応しています。

#### Linear Kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$$

このカーネルを使用したガウス過程回帰は、**リッジ回帰（Ridge Regression）**や**ベイジアン線形回帰（Bayesian Linear Regression）** と等価です。

> [!NOTE]
> **リッジ回帰との等価性の導出**
>
> 線形カーネル $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$ を用いたガウス過程回帰が、リッジ回帰と等価であることを、以下のステップで丁寧に導出します。
>
> **ステップ1：リッジ回帰のベイジアン解釈**
>
> リッジ回帰のモデルは：
>
> $$y = \mathbf{w}^T\mathbf{x} + \epsilon$$
>
> ここで $\epsilon \sim \mathcal{N}(0, \sigma^2)$ は誤差項です。パラメータ $\mathbf{w}$ に正規分布の事前分布を仮定します：
>
> $$p(\mathbf{w} | \alpha) = \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$$
>
> $n$ 個の観測データ $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ があるとき、計画行列を $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n)^T$ とすると、出力ベクトルは：
>
> $$\mathbf{y} = \mathbf{X}\mathbf{w} + \boldsymbol{\epsilon}$$
>
> ここで $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ です。
>
> パラメータ $\mathbf{w}$ を周辺化すると、出力 $\mathbf{y}$ の周辺分布は：
>
> $$p(\mathbf{y} | \mathbf{X}, \sigma^2, \alpha) = \int p(\mathbf{y} | \mathbf{X}, \mathbf{w}, \sigma^2) p(\mathbf{w} | \alpha) d\mathbf{w} = \mathcal{N}(\mathbf{0}, \mathbf{C}_{\text{ridge}})$$
>
> ここで、共分散行列 $\mathbf{C}_{\text{ridge}}$ は：
>
> $$\mathbf{C}_{\text{ridge}} = \sigma^2\mathbf{I} + \alpha^{-1}\mathbf{X}\mathbf{X}^T$$
>
> **ステップ2：ガウス過程回帰（線形カーネル）の設定**
>
> ガウス過程回帰で線形カーネル $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$ を使用する場合、共分散行列 $\mathbf{K}$ の $(i, j)$ 要素は：
>
> $$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$$
>
> したがって、共分散行列は：
>
> $$\mathbf{K} = \mathbf{X}\mathbf{X}^T$$
>
> ノイズを考慮した共分散行列は：
>
> $$\mathbf{C}_{\text{GP}} = \mathbf{K} + \sigma^2\mathbf{I} = \mathbf{X}\mathbf{X}^T + \sigma^2\mathbf{I}$$
>
> **ステップ3：共分散行列の一致**
>
> リッジ回帰の共分散行列：
>
> $$\mathbf{C}_{\text{ridge}} = \sigma^2\mathbf{I} + \alpha^{-1}\mathbf{X}\mathbf{X}^T$$
>
> ガウス過程回帰（線形カーネル）の共分散行列：
>
> $$\mathbf{C}_{\text{GP}} = \mathbf{X}\mathbf{X}^T + \sigma^2\mathbf{I}$$
>
> これらを比較すると、$\alpha^{-1} = 1$、すなわち $\alpha = 1$ とすれば、両者は一致します。より一般的には、線形カーネルを $k(\mathbf{x}_i, \mathbf{x}_j) = \alpha^{-1}\mathbf{x}_i^T\mathbf{x}_j$ とスケーリングすることで、任意の $\alpha$ に対して一致させることができます。
>
> **ステップ4：予測分布の一致**
>
> 新しい入力点 $\mathbf{x}_*$ に対する予測分布を比較します。
>
> **リッジ回帰の予測分布**：
>
> リッジ回帰では、事後分布 $p(\mathbf{w} | \mathbf{y}, \mathbf{X})$ を用いて予測を行います。事後分布は：
>
> $$p(\mathbf{w} | \mathbf{y}, \mathbf{X}) = \mathcal{N}(\boldsymbol{\mu}_{\text{post}}, \boldsymbol{\Sigma}_{\text{post}})$$
>
> ここで：
>
> $$\boldsymbol{\mu}_{\text{post}} = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$
>
> $$\boldsymbol{\Sigma}_{\text{post}} = \sigma^2(\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}$$
>
> 予測分布は：
>
> $$p(y_* | \mathbf{x}_*, \mathbf{y}, \mathbf{X}) = \mathcal{N}(\mathbf{x}_*^T\boldsymbol{\mu}_{\text{post}}, \mathbf{x}_*^T\boldsymbol{\Sigma}_{\text{post}}\mathbf{x}_* + \sigma^2)$$
>
> **ガウス過程回帰（線形カーネル）の予測分布**：
>
> ガウス過程回帰の予測分布は：
>
> $$p(y_* | \mathbf{x}_*, \mathbf{y}, \mathbf{X}) = \mathcal{N}(\mu_*, \sigma_*^2)$$
>
> ここで：
>
> $$\mu_* = \mathbf{k}_*^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y}$$
>
> $$\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{k}_* + \sigma^2$$
>
> 線形カーネルの場合、$\mathbf{K} = \mathbf{X}\mathbf{X}^T$、$\mathbf{k}_* = \mathbf{X}\mathbf{x}_*$、$k(\mathbf{x}_*, \mathbf{x}_*) = \mathbf{x}_*^T\mathbf{x}_*$ です。
>
> **予測平均の一致**：
>
> $$\mu_* = \mathbf{x}_*^T\mathbf{X}^T(\mathbf{X}\mathbf{X}^T + \sigma^2\mathbf{I})^{-1}\mathbf{y}$$
>
> ウッドベリーの公式（Woodbury formula）を用いると：
>
> $$(\mathbf{X}\mathbf{X}^T + \sigma^2\mathbf{I})^{-1} = \frac{1}{\sigma^2}\mathbf{I} - \frac{1}{\sigma^4}\mathbf{X}(\mathbf{I} + \frac{1}{\sigma^2}\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$
>
> これを整理すると：
>
> $$\mu_* = \mathbf{x}_*^T(\mathbf{X}^T\mathbf{X} + \sigma^2\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$
>
> これは、$\alpha = \sigma^{-2}$ とした場合のリッジ回帰の予測平均と一致します。
>
> **予測分散の一致**：
>
> 同様に、予測分散も一致することが示せます。
>
> **結論**
>
> 以上の導出により、**線形カーネル $k(\mathbf{x}_i, \mathbf{x}_j) = \alpha^{-1}\mathbf{x}_i^T\mathbf{x}_j$ を用いたガウス過程回帰は、正則化パラメータ $\lambda = \alpha$ のリッジ回帰と数学的に等価**であることが示されました。つまり、**リッジ回帰はガウス過程回帰の特別な場合（線形カーネル）** として理解できます。

**活用シーン**:
- 線形関係が期待される場合
- 高次元データで計算効率を重視する場合
- 解釈しやすいモデルが必要な場合
- リッジ回帰のベイジアン解釈を拡張したい場合

#### Exponential Kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|}{\ell}\right)$$

ここで：
- $\sigma_f^2$: 信号分散（signal variance）
- $\ell$: 長さスケール（length scale）

RBFカーネルと似ていますが、距離の二乗ではなく距離そのものを使います。これにより、RBFカーネルよりも**滑らかさが低く**、より急激に変化する関数を表現できます。

> [!TIP]
> **ブラウン運動との関係**
>
> 1次元の場合、指数カーネルは**ブラウン運動（Brownian motion）**や**ウィーナー過程（Wiener process）**と深い関係があります。
>
> 標準的なブラウン運動 $B(t)$ の共分散関数は：
>
> $$\text{Cov}[B(t_i), B(t_j)] = \min(t_i, t_j)$$
>
> これは、時間が進むにつれて不確実性が増加する確率過程を表現します。指数カーネルは、このブラウン運動を一般化したものと解釈できます。
>
> 特に、$\ell \to 0$ の極限では、指数カーネルはブラウン運動の性質に近づきます。また、指数カーネルは $\nu = 1/2$ のMatérnカーネルと等価であり、これは**オルンシュタイン-ウーレンベック過程（Ornstein-Uhlenbeck process）**の共分散関数に対応しています。

**活用シーン**:
- 非滑らかな関数をモデル化する場合
- 急激な変化が予想される場合
- 物理現象で不連続性が存在する場合
- 時系列データでランダムウォーク的な挙動をモデル化する場合
- 金融工学で資産価格の変動をモデル化する場合（ブラウン運動との関係から）
- 拡散過程やランダムノイズを含む物理現象のモデル化

#### Periodic Kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi\|\mathbf{x}_i - \mathbf{x}_j\|/p)}{\ell^2}\right)$$

ここで：
- $\sigma_f^2$: 信号分散（signal variance）
- $\ell$: 長さスケール（length scale）
- $p$: 周期（period）

このカーネルは、**周期的なパターン**を持つ関数をモデル化するために設計されています。距離が周期 $p$ の整数倍になると、カーネル関数の値が大きくなります。

**活用シーン**:
- 時系列データで季節性や周期性がある場合（例：気温、売上データ）
- 信号処理で周期的なパターンを抽出する場合
- 天文学データで周期的な現象をモデル化する場合
- 音声データの周波数分析

#### Matérn Kernel

$$k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\|\mathbf{x}_i - \mathbf{x}_j\|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\|\mathbf{x}_i - \mathbf{x}_j\|}{\ell}\right)$$

ここで：
- $\sigma_f^2$: 信号分散（signal variance）
- $\ell$: 長さスケール（length scale）
- $\nu$: 平滑度パラメータ（smoothness parameter）
- $\Gamma(\cdot)$: ガンマ関数
- $K_\nu(\cdot)$: 修正ベッセル関数（modified Bessel function）

Matérnカーネルは、**平滑度パラメータ $\nu$** によって関数の滑らかさを制御できます。特に、$\nu = 1/2, 3/2, 5/2$ の場合、以下のように簡略化されます：

- **$\nu = 1/2$**: 指数カーネルと等価（$C^0$ 連続、滑らかでない）
- **$\nu = 3/2$**: $C^1$ 連続（1回微分可能）
- **$\nu = 5/2$**: $C^2$ 連続（2回微分可能）
- **$\nu \to \infty$**: RBFカーネルに収束（無限回微分可能）

**活用シーン**:
- 関数の滑らかさの程度が不明な場合（$\nu$ をハイパーパラメータとして学習）
- 物理現象で適切な平滑度を制御したい場合
- RBFカーネルが滑らかすぎる場合（$\nu$ を小さく設定）
- 地理統計学（geostatistics）で空間データをモデル化する場合

**有効なカーネル関数の条件**

カーネル関数が有効であるためには、**正定値性（positive definiteness）** を満たす必要があります。つまり、任意の有限個の入力点 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ に対して、グラム行列（Gram matrix）：

$$\mathbf{K} = \begin{pmatrix} k(\mathbf{x}_1, \mathbf{x}_1) & k(\mathbf{x}_1, \mathbf{x}_2) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\ k(\mathbf{x}_2, \mathbf{x}_1) & k(\mathbf{x}_2, \mathbf{x}_2) & \cdots & k(\mathbf{x}_2, \mathbf{x}_n) \\ \vdots & \vdots & \ddots & \vdots \\ k(\mathbf{x}_n, \mathbf{x}_1) & k(\mathbf{x}_n, \mathbf{x}_2) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n) \end{pmatrix}$$

が正定値行列（positive definite matrix）である必要があります。これは、ガウス過程の共分散行列が正定値である必要があるためです。

### Gaussian Process and Kernel

ガウス過程は、平均関数とカーネル関数（共分散関数）によって完全に特徴づけられます。カーネル関数は、2つの入力点の類似度を測り、それに応じて出力の共分散を決定します。

**入力が似ていれば出力も似る**

ガウス過程の重要な性質として、**入力が似ていれば出力も似る**という性質があります。これは、カーネル関数の定義から自然に導かれます。

2つの入力点 $\mathbf{x}_i$ と $\mathbf{x}_j$ に対する出力 $y_i$ と $y_j$ の共分散は、カーネル関数によって与えられます：

$$\text{Cov}[y_i, y_j] = k(\mathbf{x}_i, \mathbf{x}_j)$$

カーネル関数は、特徴ベクトルの内積として定義されるため、入力が似ていれば（特徴ベクトルが似ていれば）、カーネル関数の値は大きくなります。共分散が大きいということは、2つの出力が強い正の相関を持つことを意味します。

## Gaussian Process Regression Model

### Derivation of Predictive Distribution

訓練データ $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ が与えられたとき、新しい入力点 $\mathbf{x}_*$ に対する出力 $y_*$ の予測分布を求めます。

訓練データと新しい入力点を合わせた同時分布は、ガウス過程の定義により多変量正規分布に従います：

$$\begin{pmatrix} \mathbf{y} \\ y_* \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \mathbf{0} \\ 0 \end{pmatrix}, \begin{pmatrix} \mathbf{K} + \sigma^2\mathbf{I} & \mathbf{k}_* \\ \mathbf{k}_*^T & k(\mathbf{x}_*, \mathbf{x}_*) \end{pmatrix}\right)$$

ここで：
- $\mathbf{K}$: 訓練データ間の共分散行列（$n \times n$）、$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$
- $\mathbf{k}_*$: 新しい入力点と訓練データ間の共分散ベクトル（$n \times 1$）、$(\mathbf{k}_*)_i = k(\mathbf{x}_i, \mathbf{x}_*)$
- $k(\mathbf{x}_*, \mathbf{x}_*)$: 新しい入力点自身の共分散（スカラー）
- $\sigma^2$: ノイズ分散

**条件付き分布の計算**

多変量正規分布の条件付き分布の性質を利用すると、訓練データ $\mathbf{y}$ が与えられたときの $y_*$ の条件付き分布は：

$$y_* | \mathbf{y}, \mathbf{X}, \mathbf{x}_* \sim \mathcal{N}(\mu_*, \sigma_*^2)$$

ここで：

$$\mu_* = \mathbf{k}_*^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y}$$

$$\sigma_*^2 = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{k}_*$$

**補足：条件付き分布の導出**

多変量正規分布 $\begin{pmatrix} \mathbf{x} \\ \mathbf{y} \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} \boldsymbol{\mu}_x \\ \boldsymbol{\mu}_y \end{pmatrix}, \begin{pmatrix} \boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\ \boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy} \end{pmatrix}\right)$ において、$\mathbf{y}$ が与えられたときの $\mathbf{x}$ の条件付き分布は：

$$\mathbf{x} | \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}(\mathbf{y} - \boldsymbol{\mu}_y), \boldsymbol{\Sigma}_{xx} - \boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx})$$

この公式を、$\mathbf{x} = y_*$、$\mathbf{y} = \mathbf{y}$ として適用することで、上記の予測分布が得られます。

### Interpretation of Predictions

**平均予測**

平均予測 $\mu_*$ は、訓練データの重み付き線形結合として表されます：

$$\mu_* = \sum_{i=1}^{n} \alpha_i k(\mathbf{x}_i, \mathbf{x}_*)$$

ここで $\boldsymbol{\alpha} = (\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y}$ です。これは、新しい入力点 $\mathbf{x}_*$ に近い訓練データ点の出力が、予測に大きな影響を与えることを意味します。

**予測の不確実性**

予測分散 $\sigma_*^2$ は、以下の2つの項から構成されます：

1. $k(\mathbf{x}_*, \mathbf{x}_*)$: 事前の不確実性
2. $-\mathbf{k}_*^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{k}_*$: 訓練データによる不確実性の減少

訓練データに近い入力点では、$\mathbf{k}_*$ が大きくなるため、不確実性が減少します。逆に、訓練データから遠い入力点では、不確実性が大きくなります。

## Hyperparameter Estimation for Gaussian Process Regression

### Log Marginal Likelihood

ガウス過程回帰では、カーネル関数のパラメータ（例：RBFカーネルの $\sigma_f^2$ と $\ell$）とノイズ分散 $\sigma^2$ を**ハイパーパラメータ（hyperparameters）** として学習する必要があります。

**対数周辺尤度（log marginal likelihood）** は：

$$\log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y} - \frac{1}{2}\log|\mathbf{K} + \sigma^2\mathbf{I}| - \frac{n}{2}\log(2\pi)$$

ここで $\boldsymbol{\theta}$ はハイパーパラメータのベクトル（例：$\boldsymbol{\theta} = (\sigma_f^2, \ell, \sigma^2)^T$）です。

対数周辺尤度は、以下の3つの項から構成されます：

1. **データ適合項**: $-\frac{1}{2}\mathbf{y}^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y}$ - データへの適合度を測る
2. **複雑度ペナルティ項**: $-\frac{1}{2}\log|\mathbf{K} + \sigma^2\mathbf{I}|$ - モデルの複雑さをペナルティ
3. **正規化項**: $-\frac{n}{2}\log(2\pi)$ - 定数項

### Optimization

ハイパーパラメータは、対数周辺尤度を最大化することで推定されます：

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta})$$

この最適化には、様々な数値最適化手法が使用されます。対数周辺尤度の勾配は、以下のように計算できます：

$$\frac{\partial \log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta})}{\partial \theta_i} = \frac{1}{2}\mathbf{y}^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\frac{\partial \mathbf{K}}{\partial \theta_i}(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y} - \frac{1}{2}\text{tr}\left((\mathbf{K} + \sigma^2\mathbf{I})^{-1}\frac{\partial \mathbf{K}}{\partial \theta_i}\right)$$

**最適化手法**

ガウス過程回帰のハイパーパラメータ最適化には、以下のような手法が使用されます：

#### 1. 勾配ベースの最適化手法

勾配情報を利用する手法は、効率的で一般的に使用されます。

**勾配降下法（Gradient Descent）**

最も基本的な勾配ベースの手法です。パラメータを以下のように更新します：

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + \eta \nabla_{\boldsymbol{\theta}} \log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}^{(t)})$$

ここで $\eta$ は学習率です。シンプルですが、収束が遅い場合があります。

**共役勾配法（Conjugate Gradient Method）**

共役勾配法は、勾配降下法よりも効率的な収束を示します。特に、対数周辺尤度が二次形式に近い場合に有効です。

**L-BFGS（Limited-memory BFGS）**

準ニュートン法の一種で、ヘッセ行列の近似を利用します。メモリ効率が良く、中規模から大規模な問題に適しています。多くの実装でデフォルトの最適化手法として使用されます。

**Adam（Adaptive Moment Estimation）**

適応的な学習率を持つ最適化手法です。深層学習で広く使用されていますが、ガウス過程回帰にも適用可能です。特に、ハイパーパラメータが多い場合や、異なるスケールのパラメータがある場合に有効です。

#### 2. グリッドサーチとランダムサーチ

勾配情報を利用しない手法です。

**グリッドサーチ（Grid Search）**

ハイパーパラメータ空間を格子状に分割し、各格子点で対数周辺尤度を評価します。シンプルですが、次元が増えると計算量が指数的に増加します。

**ランダムサーチ（Random Search）**

ハイパーパラメータ空間からランダムにサンプリングして評価します。グリッドサーチよりも効率的な場合があります。

#### 3. ベイジアン最適化（Bayesian Optimization）

**ガウス過程による最適化（GP-based Optimization）**

ハイパーパラメータ最適化自体にガウス過程を使用する手法です。対数周辺尤度を評価する回数を最小化しながら、最適なハイパーパラメータを見つけます。特に、評価コストが高い場合に有効です。

**獲得関数（Acquisition Function）**

ベイジアン最適化では、次に評価すべき点を選択するために獲得関数を使用します：
- **Expected Improvement (EI)**: 期待改善量を最大化
- **Upper Confidence Bound (UCB)**: 信頼区間の上限を最大化
- **Probability of Improvement (PI)**: 改善確率を最大化

#### 4. MCMC（Markov Chain Monte Carlo）

**ベイジアン推論によるハイパーパラメータ推定**

MCMCは、ハイパーパラメータの**事後分布**をサンプリングする手法です。点推定ではなく、ハイパーパラメータの不確実性も考慮できます。

**メトロポリス・ヘイスティングス法（Metropolis-Hastings Algorithm）**

最も基本的なMCMC手法です。提案分布から候補を生成し、受理確率に基づいてサンプルを更新します。

**ハミルトニアン・モンテカルロ法（Hamiltonian Monte Carlo, HMC）**

勾配情報を利用して効率的にサンプリングする手法です。特に、高次元のハイパーパラメータ空間で有効です。

**NUTS（No-U-Turn Sampler）**

HMCの改良版で、手動でのチューニングが不要です。Stanなどのベイジアン推論フレームワークで広く使用されています。

**MCMCの利点**：
- ハイパーパラメータの不確実性を定量化できる
- 複数の局所最適解を探索できる
- ベイジアン推論の枠組みで自然に統合できる

**MCMCの欠点**：
- 計算コストが高い（多くのサンプルが必要）
- 収束の判定が難しい
- バーンイン期間の設定が必要

#### 5. シミュレーテッドアニーリング（Simulated Annealing）

**確率的な最適化手法**

シミュレーテッドアニーリングは、金属の焼きなまし（annealing）過程にヒントを得た最適化手法です。初期には高い温度で広く探索し、徐々に温度を下げて局所最適解に収束させます。

**アルゴリズム**：

1. 初期温度 $T_0$ と初期解 $\boldsymbol{\theta}^{(0)}$ を設定
2. 各反復 $t$ で：
   - 現在の解 $\boldsymbol{\theta}^{(t)}$ の近傍から候補解 $\boldsymbol{\theta}'$ を生成
   - 目的関数の差 $\Delta = \log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}') - \log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}^{(t)})$ を計算
   - 受理確率 $p_{\text{accept}} = \min(1, \exp(\Delta / T_t))$ に基づいて候補解を受理または棄却
   - 温度を更新：$T_{t+1} = \alpha T_t$（$\alpha < 1$ は冷却率）

**利点**：
- 局所最適解から脱出できる可能性がある
- 実装が比較的簡単
- 勾配情報が不要

**欠点**：
- 冷却スケジュールの設定が必要
- 収束が遅い場合がある
- 最適なパラメータ（初期温度、冷却率）の選択が難しい

#### 6. 初期値の選択

最適化の収束性と最終的な結果は、初期値に大きく依存します。以下の戦略が使用されます：

- **経験的な初期値**: 問題領域の知識に基づいた初期値
- **ランダム初期化**: 複数のランダムな初期値から最適なものを選択
- **グリッドサーチによる粗い探索**: まず粗いグリッドで探索し、最良の点を初期値として使用

#### 7. 制約条件の扱い

ハイパーパラメータには制約条件があります（例：$\sigma_f^2 > 0$、$\ell > 0$、$\sigma^2 > 0$）。以下の手法で制約を扱います：

- **変数変換**: 制約のない変数に変換（例：$\log(\sigma_f^2)$、$\log(\ell)$）
- **制約付き最適化**: 制約条件を明示的に扱う最適化手法（例：L-BFGS-B）

**推奨される手法**

一般的には、以下の組み合わせが推奨されます：

1. **小規模データ（$n < 1000$）**: L-BFGSまたは共役勾配法
2. **中規模データ（$1000 \leq n < 10000$）**: L-BFGS
3. **大規模データ（$n \geq 10000$）**: スパースガウス過程と組み合わせたL-BFGS、またはAdam
4. **評価コストが高い場合**: ベイジアン最適化
5. **複数の局所最適解が予想される場合**: 複数の初期値から最適化、ベイジアン最適化、またはシミュレーテッドアニーリング
6. **ハイパーパラメータの不確実性を考慮したい場合**: MCMC（NUTSやHMC）
7. **勾配情報が利用できない場合**: シミュレーテッドアニーリング、グリッドサーチ、またはランダムサーチ

### Computational Complexity and Efficiency

ガウス過程回帰の計算量は、主に以下の操作に依存します：

1. **共分散行列の逆行列計算**: $O(n^3)$
2. **予測**: $O(n^2)$（新しい入力点ごと）

訓練データ数 $n$ が大きくなると、計算量が急激に増加します。

大規模データに対しては、以下の効率化手法が使用されます：

1. **スパースガウス過程**: 一部のデータ点（誘導点、inducing points）のみを使用
2. **近似手法**: Nyström近似、ランダムフーリエ特徴など
3. **並列計算**: GPUを活用した並列計算

## Generalization of Gaussian Process Regression (Cauchy Distribution, Poisson Distribution, etc.)

ガウス過程回帰は、出力が正規分布に従うことを仮定していますが、他の分布（コーシー分布、ポアソン分布など）にも一般化できます。これらは**一般化ガウス過程（Generalized Gaussian Processes）** や**潜在変数モデル（Latent Variable Models）** として知られています。

例えば、ポアソン分布を仮定する場合、出力 $y$ は非負の整数値を取ります。この場合、潜在変数 $f(\mathbf{x})$ がガウス過程に従い、$y$ は $f(\mathbf{x})$ を通じてポアソン分布のパラメータと関連付けられます。

これらの一般化については、改めて別の機会に詳しく説明する予定です。

## References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
