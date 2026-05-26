# RAG的整体架构设计



# 1. Overview

![RAGprocess](https://raw.githubusercontent.com/songqi4485/note1/master/picture/RAGprocess.jpg)

# 2. Indexing

## 2.1 Indexing工作流程

Indexing（索引构建）是 RAG 的**第一阶段**，核心作用是：

- 把原始 **Documents（文档）** 转化为**向量形式**，存入 **Vectorstore（向量数据库）**
- 为后续 Retrieval（检索）阶段提供高效、准确的向量索引

Indexing 阶段就是“**先切分 → 再向量化 → 最后用 HNSW 建索引**”的过程，把非结构化文档变成 AI 能快速“查字典”的向量数据库，为 RAG 的检索-生成提供高质量上下文。

<img src="https://raw.githubusercontent.com/songqi4485/note1/master/picture/Indexing.png" alt="Indexing" style="zoom:50%;" />

## 2.2 Skip List

王树森PPT

[AdvancedAlgorithms/Slides/02_Basic_3.pdf at master · wangshusen/AdvancedAlgorithms](https://github.com/wangshusen/AdvancedAlgorithms/blob/master/Slides/02_Basic_3.pdf)

## 2.3 NSW

### 1）NSW在HNSW中的定位

​	NSW（Navigable Small World）可以看作一张**单层的邻近图结构**。图中的每个数据点对应一个节点，节点之间通过边相连，边既承担局部邻近连接的作用，也承担跨区域跳转的作用。它的核心目标是让**搜索过程不必遍历整个数据集，而是能够沿着图中的边逐步接近目标。**

​	如果把HNSW拆开来看，可以理解为**“概率跳表式的层级结构”与“每一层内部的NSW图”**的组合。概率跳表决定节点会出现在哪些层，负责形成由稀到密的层次骨架；NSW则负责每一层内部节点之间如何连边、如何执行搜索。

### 2）NSW的核心思想

NSW（Navigable Small World）把数据点看成图中的节点。边分成两类角色：

1. **短程边**：近邻边，作用类似于局部几何骨架，论文把它理解为对 Delaunay graph 的一种近似；
2. **长程边**：远距离跳转边，作用是让**搜索不必只在局部慢慢爬**，而能更快“跳到正确区域”。

![NSW核心思想](https://raw.githubusercontent.com/songqi4485/note1/master/picture/NSW核心思想.png)

​	如果只保留短边，它更像普通近邻图；如果只有长边，又会失去局部精修能力。**NSW 的关键不在于“有边”，而在于“同时混合了局部连通性和全局跳跃性”。**

---

​	NSW之所以叫“可导航小世界”，关键不只是图中存在较短路径，而是存在一种简单的局部搜索规则：**每一步只考察当前节点的邻居，就能找到更接近查询点的下一跳。**也就是说，搜索不需要知道全局结构，只凭当前局部信息就能有效推进。

### 3）NSW的图结构特征

* **图结构体现了局部连通与全局跳跃的结合**

​	从整体上看，NSW既不是纯粹的近邻图，也不是随机图。它保留了大量局部连接，以维持空间邻域中的可达性；同时又存在少量尺度较大的连接，用于在不同区域之间快速跳跃。这种结构使搜索路径通常表现为“先远跳、再细化”的模式。

* **长边并非额外设计出来的独立机制**

​	在NSW中，许多长边并不是通过单独的远程连接策略专门构造出来的，而是在**增量插入过程中自然形成**的。早期插入阶段建立的连接，在数据规模不断扩大后，可能从“当时的近邻边”演化为“后来视角下的远程边”。因此，NSW的小世界性质在很大程度上来源于其增量构建过程。

* **NSW本质上是一种近邻图的增量逼近**

​	如果从理想图结构的角度看，NSW可以理解为**对某种高质量邻近图的近似构造**。它并不要求精确恢复最优的全局邻接关系，而是通过“插入一个点，就搜索并连接一批有代表性的邻居”的方式，逐步生成一张具备导航性质的近似图。这种思想使其既实用又适合动态插入。

## 2.4 HNSW算法

Hierarchical Navigable Small World

  [HNSW算法的基本原理及使用 - 知乎](https://zhuanlan.zhihu.com/p/673027535)<div class="hnsw-note">

### 1）创建HNSW

* HNSW 是 NSW 的自然演化，它从 Pugh 的概率跳表结构中汲取了灵感，添加了层级的概念。向 NSW 中添加层级会产生一个图，其中的链接在不同的层之间分离。在顶层，我们拥有最长的链接，在底层，我们拥有最短的链接。

![创建HNSW1](https://raw.githubusercontent.com/songqi4485/note1/master/picture/创建HNSW1.png)

---

* HNSW的分层图，**顶层是我们的入口点**，**只包含最长的链接**，随着我们往下移动，链接长度变得越来越短，越来越多。

* 在搜索过程中，我们进入顶层，找到最长的链接。这些顶点往往是高度顶点（具有在多个层之间分离的链接），这意味着我们默认情况下会从 NSW 中的缩小阶段开始。
* 我们沿着每个层的边缘遍历，就像我们在 NSW 中所做的那样，贪婪地移动到最近的顶点，直到找到一个局部最小值。与 NSW 不同的是，此时我们**转移到较低层中的当前顶点，并开始再次搜索**。我们重复这个过程，直到找到**底层（层 0）的局部最小值**。

![创建HNSW2](https://raw.githubusercontent.com/songqi4485/note1/master/picture/创建HNSW2.png)

### 2）图构建

在图构建过程中，向量逐个进行插入。层数由参数 *L* 表示。向量在给定层插入的概率由一个由*‘level multiplier’ m_L*归一化的概率函数给出，其中*m_L = ~0*表示向量仅在第0层插入。

![图构建1](https://raw.githubusercontent.com/songqi4485/note1/master/picture/图构建1.png)

对于每个层（除了层 0），概率函数都会重复一次。将向量添加到其插入层以及下面的每一层。

当我们**最小化跨层共享邻居的重叠**时，可以获得最佳性能。减小 *m_L* 可以帮助减少重叠（将更多向量推到层 0），但这会增加搜索过程中的平均遍历次数。因此，我们使用平衡两者的 *m_L* 值。这个最优值的经验法则是 ***1/ln (M)***。

---

**第一阶段：从顶层开始做粗定位**

* **从当前最高层入口点进入图**

​	插入新向量 `q` 时，算法不是直接在底层找邻居，而是先从图的最高层入口点开始。

* **在当前层执行贪婪搜索**

​	在这一层中，算法沿着图中的边不断移动，寻找**距离 `q` 更近的节点**。
​	这里的搜索宽度设为：`ef = 1`<br>	这意味着此时并不是保留一批候选点，而是基本上只保留当前最优方向，属于一种**单路径的贪婪下降**。

* **找到当前层的局部最小点**

​	当在这一层再也找不到更接近 `q` 的邻居时，就认为已经到达了该层的一个局部最优位置。

* **将该点作为下一层入口，继续向下**

​	然后算法不停止，而是把当前找到的局部最优点作为**下一层的入口点**，继续在更低一层重复同样的过程。

* **重复直到到达目标插入层**

​	每个新节点都会随机分配一个最高层级。
​        在到达这个节点应当出现的**最高插入层**之前，算法一直都在执行上面的“粗定位”过程。

---

**第二阶段：从插入层开始做正式建边**

当下降到新节点 `q` 应当进入的最高层时，真正的建图才开始。

* **将搜索宽度从 `ef = 1` 提升到 `efConstruction`**

​	此时算法不再只保留一个当前最优点，而是扩大搜索范围，使用参数：`ef = efConstruction`。这意味着算法会保留更多候选近邻，从而提高找到优质邻居的概率。

* **在当前层搜索一批候选邻居**

​	在当前层中，算法围绕 `q` 搜索，得到一批与 `q` 接近的候选顶点。
​        这些候选点有两个作用：<br>	作为**新节点 `q` 的潜在连接对象**；作为**继续往下一层搜索时的入口候选**

* **从候选集中选择最终邻居**

​	得到候选邻居后，需要从中挑出一部分真正与 `q` 建立连接。最简单的做法是：直接选出距离 `q` 最近的 `M` 个节点。（**M为新插入节点在该层期望保留的邻居数**）

*  **为新节点建立双向连接**

​	选出这 `M` 个邻居后，就把 `q` 与这些节点连接起来。

* **继续向更低层重复建边**

​	在当前层完成连接之后，算法继续下降到下一层。
​	 在每一层中都重复以下过程：

​		-用 `efConstruction` 搜索候选邻居

​		-从候选中选出邻居

​		-建立连接

​	直到最终到达 **第 0 层**。

---

![HNSW2](https://raw.githubusercontent.com/songqi4485/note1/master/picture/HNSW2.png)

## 2.5 HNSW在 Faiss的实现

### 1）索引初始化

​	作用：使用 `faiss.IndexHNSWFlat(d, M)` 创建 HNSW 索引。

​	**重点**：在 Faiss 中，`M_max` 和 `M_max0` 不是这里手动传入的，而是内部自动设置。

```python
import faiss

d = 128      # 向量维度
M = 32       # 每个顶点添加的邻居数

index = faiss.IndexHNSWFlat(d, M)
print(index.hnsw)
```

### 2）构建前的索引状态

​	刚初始化完成时，HNSW 层级结构还没有真正建立：max_level = -1，`levels` 为空。

​	要点：这说明此时只是“索引对象存在”，但“多层图结构”还没构建出来。

```python
# HNSW 刚创建时还没有层
print(index.hnsw.max_level)   # -1

# levels 也还是空的
levels = faiss.vector_to_array(index.hnsw.levels)
print(levels)
```

### 3）构建前需要设置的参数

​	`efSearch`：搜索阶段使用的候选邻居数，影响查询时的搜索宽度。

​	`efConstruction`：建图阶段使用的候选邻居数。影响建图时选邻居搜索的充分程度。

```python
efSearch = 32
efConstruction = 32

index.hnsw.efSearch = efSearch
index.hnsw.efConstruction = efConstruction
```

### 4）正式构建 HNSW 图

​	通过 `index.add(xb)` 把向量加入索引。图的层级、入口点、节点分布都在这一步之后才真正生成。

### 5）构建后的层级信息

```python
print(index.hnsw.max_level)   # 4

# 查看各层节点分布
import numpy as np
levels = faiss.vector_to_array(index.hnsw.levels)
print(np.bincount(levels))
# array([0, 968746, 30276, 951, 26, 1], dtype=int64)

#入口点查看
print(index.hnsw.entry_point)
# 118295
```

### 6）图结构生成逻辑

**set_default_probas 的作用**

Faiss 初始化 HNSW 时会调用 `set_default_probas(M, 1 / log(M))`。这个过程会生成两类关键数据：

* `assign_probas`：各层的插入概率
* `cum_nneighbor_per_level`：到各层为止的累计邻居预算。

```python
import numpy as np

def set_default_probas(M: int, m_L: float):
    """
    计算 HNSW 算法中各层级的节点分配概率和累积邻居数量
    参数: M: 每层的最大连接数（除了第0层是2M）
        m_L: 层级归一化参数，通常设为 1/ln(M)
    """
    
    # 初始化累积邻居计数器，用于跟踪从第0层到当前层的总邻居数
    nn = 0
    
    # 存储每一层的累积邻居数量列表
    # 例如：[64, 96, 128, ...] 表示第0层64个，前两层共96个，前三层共128个
    cum_nneighbor_per_level = []
    
    # 从第0层开始计算
    level = 0
    
    # 存储每一层的节点分配概率
    # 该概率决定新插入的节点被分配到某一层的可能性
    assign_probas = []
    
    # 循环计算各层的概率，直到概率值变得极小（可忽略）
    while True:
        # 计算当前层级的分配概率
        # 公式: P(l) = e^(-l/m_L) * (1 - e^(-1/m_L))
        # 这是一个指数衰减分布，层级越高，分配到该层的概率越低
        # m_L 是归一化因子，控制概率衰减的速度
        proba = np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))
        
        # 当概率小于 1e-9 时认为已经足够小，可以终止循环
        # 这个阈值确保只保留有实际意义的层级
        if proba < 1e-9:
            break
        
        # 将当前层级的概率添加到概率列表中
        assign_probas.append(proba)
        
        # 更新累积邻居数量
        # 第 0 层（底层）：每个节点最多有 M*2 个邻居（双向连接更密集）
        # 其他层：每个节点最多有 M 个邻居
        # 这样设计是因为底层需要更高的连接度以提供更好的搜索准确性
        nn += M * 2 if level == 0 else M
        
        # 将当前的累积邻居数添加到列表中
        # 这个值表示从第0层到当前层，所有层的邻居数总和
        cum_nneighbor_per_level.append(nn)
        
        # 移动到下一层
        level += 1
    
    # 返回两个列表：
    # 1. assign_probas: 各层的节点分配概率
    # 2. cum_nneighbor_per_level: 各层的累积邻居数
    return assign_probas, cum_nneighbor_per_level


# 调用函数，使用典型的 HNSW 参数
# M = 32: 每层最大连接数为32（第0层为64）
# m_L = 1/ln(32): 归一化参数，约等于 0.288
# 这个参数选择使得平均层数约为 ln(N) * m_L，其中 N 是数据集大小
assign_probas, cum_nneighbor_per_level = set_default_probas(
    32, 1 / np.log(32)
)

# 打印各层的分配概率
# 输出示例：[0.721, 0.200, 0.055, 0.015, ...] （概率递减）
print(assign_probas)

# 打印各层的累积邻居数
# 输出示例：[64, 96, 128, 160, ...] （第0层64个，之后每层增加32个）
print(cum_nneighbor_per_level)

[0.96875, 0.030273437499999986, 0.0009460449218749991,
 2.956390380859371e-05, 9.23871994018553e-07, 2.887099981307982e-08]
[64, 96, 128, 160, 192, 224]
```

第 0 层概率最大，所以绝大多数节点都在底层第 0 层的邻居预算也是最大的，因为它按 `2*M` 计算。

### 7）顶点层级分配机制

`random_level` 的作用

* 每个点插入 HNSW 时，并不是固定出现在所有层
* 而是通过 `random_level` 随机决定其最高层级。

```python
def random_level(assign_probas: list, rng):
    """
    根据给定的概率分布随机选择一个层级
    
    参数:
        assign_probas: 各层级的分配概率列表，例如 [0.721, 0.200, 0.055, ...]
        rng: NumPy 随机数生成器对象
    
    返回:
        选中的层级索引（0 表示底层，数字越大层级越高）
    """
    
    # 生成一个 [0, 1) 区间的均匀分布随机数
    # 这个随机数将用于确定落在哪个概率区间
    f = rng.uniform()
    
    # 遍历所有层级，使用累积概率减法来确定选中的层级
    # 这是一种经典的离散概率分布采样方法
    for level in range(len(assign_probas)):
        # 检查随机数 f 是否落在当前层级的概率区间内
        # 例如：如果 assign_probas[0]=0.721，当 f < 0.721 时选择第0层
        if f < assign_probas[level]:
            return level  # 找到对应层级，立即返回
        
        # 如果没有落在当前层级，从 f 中减去当前层级的概率
        # 继续检查下一个层级
        # 例如：f=0.8，第0层概率0.721，则 f 变为 0.8-0.721=0.079
        #       然后检查是否 0.079 < assign_probas[1] (0.200)
        f -= assign_probas[level]
    
    # 如果遍历完所有层级都没有返回（理论上由于浮点误差可能发生）
    # 返回最高层级作为默认值
    # 这是一个安全措施，确保函数总是返回有效的层级
    return len(assign_probas) - 1


# 创建一个空列表，用于存储100万次随机选择的结果
chosen_levels = []

# 创建一个带固定种子的随机数生成器，确保结果可复现
# 种子 12345 使得每次运行代码都会得到相同的随机序列
rng = np.random.default_rng(12345)

# 进行 1,000,000 次随机层级选择实验
# 这个大样本量可以很好地验证概率分布是否符合预期
for _ in range(1_000_000):
    # 调用 random_level 函数随机选择一个层级
    chosen_levels.append(random_level(assign_probas, rng))

# 使用 np.bincount 统计各层级被选中的次数
# bincount 会计算每个整数值（层级）出现的频率
# 输出是一个数组，索引对应层级，值对应该层级被选中的次数
print(np.bincount(chosen_levels))

# 输出结果: array([968821, 30170, 985, 23, 1], dtype=int64)
# 解读：
# - 第0层（底层）：被选中 968,821 次，占比约 96.88%
# - 第1层：被选中 30,170 次，占比约 3.02%
# - 第2层：被选中 985 次，占比约 0.10%
# - 第3层：被选中 23 次，占比约 0.002%
# - 第4层：被选中 1 次，占比约 0.0001%
#
# 这个分布验证了指数衰减的概率分配：
# - 绝大多数节点在底层（第0层），形成密集的搜索基础
# - 越往高层，节点越少，形成稀疏的"快速通道"
# - 这种金字塔结构是 HNSW 算法高效的关键
```

## 2.6 HNSW 退化为 NSW 

当 `m_L` 非常接近 0 时，HNSW 会退化为单层图，也就是 NSW。

## 💕💕💕**面试八股**

​	HNSW算法（Hierarchical Navigable Small World **分层可导航小世界**）是一种高效的**近似最近邻搜索**（Approximate Nearest Neighbor, ANN）算法，特别适用于**高维向量空间**中的**相似性搜索问题**。它通过**构建分层图结构**来快速搜索高维数据中相似的向量。与传统的暴力搜索（即直接计算所有点之间的距离）相比，HNSW可以大大加快搜索速度，同时保持较高的准确度。

![HNSW核心概念](https://raw.githubusercontent.com/songqi4485/note1/master/picture/HNSW核心概念.png)

![与余弦相似度关系](https://raw.githubusercontent.com/songqi4485/note1/master/picture/与余弦相似度关系.png)

![与余弦相似度相比的优势](https://raw.githubusercontent.com/songqi4485/note1/master/picture/与余弦相似度相比的优势.png)

总结：HNSW算法通过构建层次化的小世界图，能够快速在高维空间中进行近似最近邻搜索。当与**余弦相似**度结合使用时，HNSW能够有效加速相似向量的查找过程，广泛应用于需要高维特征相似性度量的场景中，如推荐系统、文本检索和图像检索等。



# 3. Retrival

![retrival](https://raw.githubusercontent.com/songqi4485/note1/master/picture/retrival.png)

# 4. Generation

![generation](https://raw.githubusercontent.com/songqi4485/note1/master/picture/generation.png)
