Research Proposal: 几何位置编码与流形可解释性的耦合
For: Aris (Auto Claude Code Research in Sleep)
From: 柠檬酸
Date: May 2026
背景：两篇论文，
SAE-Manifold: https://arxiv.org/abs/2604.28119
GRAPE: https://arxiv.org/abs/2512.07805
一个隐藏的对偶
最近两篇 2026 年的论文从相反方向触及了同一个对象——Transformer 表示空间里的李群轨道——但都没有意识到对方的存在。
Bhalla et al. (2026), "Do Sparse Autoencoders Capture Concept Manifolds?" 论证了 LLM 内部的概念不是独立线性方向（线性表示假设 LRH），而是低维流形：周是圆、颜色是抛物面（色相 + 明度）、年份是螺旋、温度/年龄是连续直线。他们把表示模型推广为"流形加性混合"（Additive Mixture of Manifolds），其中 LRH 是流形维度退化为 1 的特例。然后实证发现所有当前主流 SAE 架构（标准 ℓ1、JumpReLU、TopK、BatchTopK、Matryoshka）都处于"稀释"相态——流形被碎裂式地分散到大量冗余的局部检测器上，单个特征看不到几何结构，必须事后用 Ising 模型恢复。他们在论文末尾承认这是 SAE 设计本身的缺陷：原子是一维的，loss 只奖励重构、不奖励几何相干性。
Zhang et al. (ICLR 2026), "GRAPE: Group Representational Position Encoding" 把 RoPE、ALiBi、FoX 统一到 G(n)=exp⁡(nωL)G(n) = \exp(n\omega L)
G(n)=exp(nωL) 的李群作用框架。RoPE 是 so(d)\mathfrak{so}(d)
so(d) 中对易、坐标对齐、对数均匀谱的特殊情形；ALiBi 和 FoX 是 GLGL
GL 中幂零生成元的特殊情形。GRAPE-M 的非对易扩展允许跨子空间耦合，路径积分变体（GRAPE-AP）允许端点依赖的因果偏置。
隐藏的连接
这两篇在做结构上完全镜像的事：
旧观点新观点SAE-Manifold概念 = 方向概念 = 流形（方向是退化情况）GRAPE位置编码 = 工程 trick位置编码 = 李群作用（RoPE 是退化情况）
更深一层：旋转结构在两篇里以相反方式出现。

GRAPE 显式构造带李群结构的位置编码注入模型——位置 nn
n 通过 SO(d)SO(d)
SO(d) 中的旋转作用在 (q,k)(q, k)
(q,k) 上
SAE-Manifold 事后发现模型表示空间里到处是李群轨道（圆、螺旋、周期循环），但 SAE 用方向字典看不见

也就是说，模型表示里的李群轨道是个反复出现的基本结构，无论我们注入它还是发现它。
核心猜想
用 GRAPE-style 的几何先验训练的模型，应该比标准 RoPE 模型在 SAE-Manifold 测得的"流形捕获"指标上表现更好。
直觉是这样的：SAE 之所以陷入稀释相态，部分原因是模型表示里的流形结构本身就偏弱或偏隐式——RoPE 只在 d/2d/2
d/2 个对易的、坐标对齐的平面里旋转，跨子空间的几何耦合必须靠后续层"挣"出来。如果位置编码本身允许非对易耦合（GRAPE-M 扩展）或路径积分的端点依赖结构（GRAPE-AP），表示空间的流形结构可能更显式、更紧凑，从而 SAE 字典能用接近"紧凑捕获"的方式恢复它，而不是碎成一堆冗余局部检测器。
为什么这个连接值得严肃对待

可解释性研究和架构研究通常各做各的。 SAE 文献几乎不讨论位置编码选择对特征几何的影响；位置编码文献也几乎不用 SAE 评估表示质量。如果这个连接成立，它意味着两个社区一直在错过对方的关键变量。
它给 SAE-Manifold 论文末尾的开放问题提供了一个候选答案。 作者呼吁"开发把几何对象当作基本单元的 featurizer"——但其实问题可能不全在 featurizer，也在被 featurize 的对象。修改位置编码可能是比修改 SAE 更便宜的干预。
它给 GRAPE 论文提供了一个意料之外的评估维度。 GRAPE 现在的卖点是 perplexity 和 length extrapolation，但它的真正价值可能是让模型表示更可解释——这是李群框架的一个隐藏 dividend。
如果猜想错了，反例本身也很有信息量。 如果 GRAPE 训练的模型 SAE 仍然处于稀释相态，那说明稀释问题完全是 SAE 端的设计缺陷，跟模型端的几何先验无关——这会显著缩小后续研究的搜索空间。