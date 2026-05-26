# 1.检索优化策略

## 总览

![RAG-5-重排序上下文](https://raw.githubusercontent.com/songqi4485/note1/master/picture/RAG-5-重排序上下文.png)

这一讲核心问题是：**检索到了上下文，并不等于拿到了最适合给大模型看的上下文**。

在 RAG 里，初次召回常常会遇到两个问题：

- **召回不全**：用户问题只用单条查询去搜，容易漏掉表达方式不同但相关的内容。
- **排序不准**：即使召回到了相关片段，真正最重要的片段也不一定排在前面。

因此，这一讲关注的是 **Re-ranking（重排序）**，也就是在“粗检索”之后，再做一轮更精细的排序或筛选，让送进 LLM 的上下文更相关、更紧凑。

Notebook 里主要展示了两条路线：

- **RAG-Fusion + RRF**：多查询召回，再做融合重排。
- **Cohere Rerank**：先召回候选，再交给专门的重排模型重新打分。

# 💕💕💕 八股



<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
   首先需要先检索，之后再进行后面的Ranking、Refinement和Adaptive retrival操作。<strong>检索是在索引的基础上进行查询的</strong>，所以检索方式和索引结构分不开。<strong>构建检索的目的是为了更快的检索</strong>，检索器可以针对单个索引，也可以组合不同检索技术。
</div>



---

### （1）父文档检索（Parent Document Retrieval）

<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
   当我们对文档进行分块的时候，我们可能希望<strong>每个分块不要太长</strong>，因为只有当<strong>文本长度合适，嵌入才可以最准确地反映它们的含义，</strong>太长的文本嵌入可能会失去意义；但是在将<strong>检索内容</strong>送往大模型时，我们又希望<strong>有足够的长的文本，</strong>，以保留完整的上下文。为了实现二者平衡，有以下三种方式实现父文档检索：
</div>



![5检索形式](https://raw.githubusercontent.com/songqi4485/note1/master/picture/5检索形式.png)



<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #2563eb;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">1.</span>
          可以在检索过程中，首先<strong>获取小的分块，</strong>然后查找这些小分块的父文档，并<strong>返回较大的父文档，</strong>这里的父文档指的是小分块的来源文档，可以是整个原始文档，也可以是一个更大的分块。<br>
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">2.</span>
           使用<strong>大模型对文档进行摘要，</strong>然后对摘要进行嵌入和检索，这种方法对处理包含大量冗余细节的文本非常有效，这里的原始文档就相当于摘要的父文档。
         </span>
       </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">3.</span>
           通过大模型为每个文档生成<strong>假设性问题（Hypothetical Questions)，</strong>然后对问题进行嵌入和检索，也可以结合问题和原文档一起检索，这种方法提高了搜索质量。因为与原始文档相比，用户查询和假设性问题之间的语义相似度更高。
         </span>
       </li>



---

### （2）层级检索（Hierarchical Retrieval）



<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
   有大量的文档需要检索，为了高效地在其中找到相关信息，一种高效的方法是<strong>创建两个索引：</strong>：一个由<strong>摘要</strong>组成，另一个由<strong>文档块</strong>组成。然后分<strong>两步搜索</strong>，首先通过<strong>摘要筛选相关文档</strong>，然后<strong>再在筛选的文档中搜索。</strong>【在part4中提到，PAPTOR是其中一种实现方式】
</div>



![5层级检索](https://raw.githubusercontent.com/songqi4485/note1/master/picture/5层级检索.png)

---

### （3）混合检索(Fusion Retrieval)



<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀【在二的part6】有提到<strong>RAG融合(RAG Fusion)</strong>技术，它根据用户的原始问题生成意思相似但表述不同的子问题并检索。其实，还可以<strong>结合不同的检索策略，</strong>最常见的做法是将<strong>基于关键词</strong>的老式搜索和<strong>基于语义</strong>的现代搜索结合起来。</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">1.</span>
          基于关键词的搜索又被称为<strong>稀疏检索器(sparse retriever)，</strong>通常使用BM25、TF-IDF等传统检索算法
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">2.</span>
           基于语义的搜索又被称为<strong>密集检索器(dense retriever)，</strong>使用的是现在流行的Embedding算法。
         </span>
       </li>



<div style="
  background:#f0fdf4;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀通常结合了<strong>稀疏检索（Sparse Retrieval）</strong>和<strong>稠密检索（Dense Retrieval）</strong>的策略，通常可以兼顾两种检索方式的优势，提高检索的效果和效率，两种方法的详细解释如下：</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">1.</span>
          稀疏检索（Sparse Retrieval）：这种方法通常基于倒排索引(Inverted Index)，对文本进行词袋(Bag-of-Words)、BM25或者TF-IDF表示，然后按照关键词的重要性对文档进行排序，稀疏检索的优点是速度快，可解释性强，但在处理同义词、词语歧义等语义问题时效果有限。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">2.</span>
           稠密检索（Dense Retrieval）：这种方法利用深度神经网络，将查询和文档映射到一个低维的稠密向量空间，然后通过向量相似度（如点积、余弦相似度）来度量查询与文档的相关性。稠密检索能更好地捕捉语义信息，但构建向量索引的成本较高，检索速度也相对较慢。
         </span>
       </li>



![关键词搜索与向量搜索的对比](https://raw.githubusercontent.com/songqi4485/note1/master/picture/关键词搜索与向量搜索的对比.png)



<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀在实际RAG系统开发中，现实通常是各种情况都有，难以使用一种搜索方法解决全部问题。用户的查询可能涵盖广泛的类型，从精确的关键词匹配到抽象的概念探索，再到专业领域的术语搜索。同时，知识库中的数据也可能是多样化的，包含结构化和非结构化信息、数字数据、专有名词等。面对这些复杂的需求，仅依赖向量搜索或全文搜索中的一种往往会导致检索结果的不准确。这就是为什么在现代RAG系统中，<strong>混合搜索方法变得越来越重要</strong>的原因</span>



<div style="
  background:#f0fdf4;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀混合搜索的工作原理：</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">1.并行执行:</span><br>
          · 对于每个查询，系统同时执行向量搜索和全文搜索。<br>
          · 向量搜索捕捉查询的语义内容。<br>
          · 全文搜索处理关键词匹配和精确查找。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">2.结果融合：</span>
           · 使用特定算法将两种搜索的结果合并成一个统一的结果集。<br>
           · 最常用的方法之一是倒数排名融合（Reciprocal Rank Fusion，RRF）算法。
         </span>
       </li>



并行执行的实现并不复杂，它的关键技巧是结果融合，这个问题通常是通过**倒数排名融合（Reciprocal Rank Fusion，RRF）**算法来解决的。RRF算法是**对检索结果重新进行排序从而获得最终的检索结果。**

RRF是滑铁卢大学和谷歌合作开发的一种算法，它可以将具有不同相关性指标的多个结果集组合成单个结果集。RRF的主要机制是**根据每个检索结果的排名位置来分配权重**，权重的计算公式为：

<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">
    RRF(d) = ∑<sub>i=1</sub><sup>n</sup> 1 / (k + r<sub>i</sub>(d))
    <br><br>
    • 其中：
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;○ d 是文档或项。
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;○ n 是检索系统的数量，即有多少个检索器的结果被用来融合。
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;○ r<sub>i</sub>(d) 是文档 d 在第 i 个检索器中的排名（rank），越靠前的排名值越小。
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;○ k 是一个常数，通常取 60，用来防止低排名项的权重过高。
    <br><br>
    <strong>k 是一个常量，默认值为 60。</strong>
    RRF 不依赖于每次检索分配的绝对分数，而是<strong>依赖于相对排名</strong>，
    这使得它非常适合组合来自可能具有不同分数尺度或分布的查询结果。
  </span>
</div>



<div style="
  background:#f0fdf4;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀 实现权重平衡的办法，可以通过直接调整经典RRF公式中的k值来实现。在经典RRF公式中，k为常数，建议设为60，实际这个<strong>K是可调的</strong>。通过调整k值，我们可以有效地<strong>改变关键词搜索和向量搜索的相对重要性权重</strong>，从而是RRF算法获得更好的性能。<br>
    直接修改k值是对RRF公式增加权重平衡的最简单方法，易于实施和调整，适合快速优化和实验。2023年5月，提出了一种新的混合搜索算法，称为TM2C2（Theoretical Min-Max Convex Combination），有如下优势：</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">1.稳定性:</span><br>
        	相较于传统的min-max归一化，TM2C2更稳定。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">2.性能：</span>
           在大多数数据集上，TM2C2由于RRF和其他基线方法。
         </span>
       </li>
        <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">3.可解释性：</span>
           a参数直观表示了语义搜索和关键词搜索的相对重要性。
         </span>
       </li>
        <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">4.样本效率：</span>
           只需要很少的训练样本就能调整到较好的性能。
         </span>
       </li>
    TM2C2算法实际上是RRF引入权重参数和归一化函数后的变体。这一变化为特定场景下，混合搜索的性能提升提供了更多的可能性。

---

### （4）多向量检索(Multi-Vector Retrieval)



<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          对于同一份文档，可以有<strong>多种嵌入方式</strong>，也就是为<strong>同一份文档生成几种不同的嵌入向量</strong>，这在很多情况下可以提高检索效果，这被称为<strong>多向量检索器(Multi-Vector Retriever)</strong>。为同一份文档生成不同的嵌入向量有很多策略可供选择，上面所介绍的父文档检索就是比较典型的方法
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         当我们处理包含文本和表格的半结构化文档时，多向量检索器也能派上用场。在这种情况下，可以提取每个表格，为表格生成合适检索的摘要，但生成答案时将原始表格送给大模型。有些文档不仅包含文本和表格，还可能包含图片，随着多模态大模型的出现，我们可以为图像生成摘要和嵌入。
        </span>
      </li>
      </ul>
</div>



<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 16px; margin-right: 8px;">🎀<strong>注意：父文档检索和层级检索很相似，其区别在于父文档检索只检索一次摘要，</strong>然后由摘要扩展出原始文档，而<strong>层级检索是通过检索</strong>摘要筛选出一批文档，然后在筛选出的文档中执行<strong>二次检索。</strong></span>
</div>



### （5）后处理

RAG系统的最后一个问题，如何将检索出来的信息丢给大模型？

<div style="
  background: #f0fdf4;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 16px; margin-right: 8px;">🎀检索出来的信息可能过长或者冗余（比如从多个来源进行检索），我们可以在后处理步骤中对其进行<strong>上下文压缩、排序、去重</strong>等。LangChain中并没有专门针对后处理的模块，文档也是零散分布在各个地方。</span>
</div>



# 2.重排

## 2.1 检索与重排序区别

* **检索（Retrieval）**：从大语料中快速找出候选文档。

* **重排序（Re-ranking）**：对候选文档重新评估，让更相关的文档排得更靠前。

## 2.2 Reciprocal Rank Fusion, RRF

使用 `RRF` 对多路检索结果做融合重排。

核心公式是：

$$
\mathrm{score}(d) = \sum_{i=1}^{m} \frac{1}{k + \mathrm{rank}_i(d)}
$$

其中：

- $d$ 表示某个文档片段
- $m$ 表示查询条数
- $\mathrm{rank}_i(d)$ 表示文档在第 $i$ 路检索结果中的名次
- $k$ 是平滑参数，Notebook 里取 `60`

直观理解：

- 一个文档如果在多条查询里都被检索到，它的累计分数会更高。
- 一个文档排名越靠前，贡献越大。
- `k` 越大，排名差异带来的影响越平缓。

## 2.3 二阶段重排

（1）先用普通检索器召回 `top-k` 候选

（2）再用专门的 `reranker` 对候选逐条打分

（3）最后只保留最相关的 `top_n` 文档进入上下文

Notebook 里使用的是 `CohereRerank`，并通过 `ContextualCompressionRetriever` 包装成“基础检索 + 重排/压缩”的组合检索器。

## 2.4 Notebook工作流

###  （1）构建索引

Notebook 选取的语料是 Lilian Weng 关于 Agent 的博客文章：

- 数据加载：`WebBaseLoader`
- 解析限制：只抓正文、标题、页眉等关键区域
- 目的：减少导航栏、页脚等无关文本噪声

### （2）文本切块

Notebook 使用：

- `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)`
- `chunk_size=300`
- `chunk_overlap=50`

这意味着每个 chunk 约 300 token，相邻块重叠 50 token。  
这样做是为了兼顾两件事：

- chunk 不要太大，便于检索和重排
- chunk 之间保留上下文连续性，避免语义硬切断

### （3）向量化与向量库

Notebook 自定义了一个 `NVIDIAEmbeddingsCompat` 类，用来把 NVIDIA 的 OpenAI 兼容 Embedding 接口接入 LangChain。

关键信息：

- Embedding 模型：`nvidia/nv-embedqa-e5-v5`
- 接口地址：`https://integrate.api.nvidia.com/v1`
- 向量库：`Chroma`

这个包装类实现了两个标准方法：

- `embed_documents(texts)`：把多条文档变成向量
- `embed_query(text)`：把单条查询变成向量

本质上，它做的是 **接口适配**，让 NVIDIA Embedding 能直接作为 LangChain 向量库的 `embedding_function` 使用。

### （4） 多查询生成

Notebook 定义了一个提示词，让模型围绕用户问题生成 4 条检索查询：

- 使用组件：`ChatPromptTemplate`
- 生成模型：`qwen/qwen2.5-7b-instruct`
- 输出解析：`StrOutputParser`

最终会把模型输出按“每行一条”拆成查询列表。

这一步的作用是 **提升召回覆盖面**，而不是直接生成答案。

### （5） 多路检索 + RRF 融合

链路如下：

`generate_queries | retriever.map() | reciprocal_rank_fusion`

含义是：

1. 先生成多条查询；
2. 每条查询都去向量库检索；
3. 把多路结果交给 `reciprocal_rank_fusion` 做融合重排。

值得注意的实现细节：

- 代码里用 `dumps(doc)` / `loads(doc)` 来序列化文档对象；
- 这样同一个文档若在多路检索中重复出现，就能按“同一文档”累计分数；
- 最终输出是按融合分数从高到低排列的 `(文档, 分数)` 列表。

### （6） 最终生成链

Notebook 并没有停在“检索结果列表”，而是继续把重排后的上下文交给 LLM：

`context <- retrieval_chain_rag_fusion`

`question <- 原始问题`

然后通过提示词让模型“基于上下文回答问题”。

这里的关键认知是：

- **重排不是终点**
- **重排是为了提升生成阶段输入质量**

也就是说，重排序上下文的真正目标，是让最终答案更靠谱。

## 2.5 二级重排路径

### 路线 A：RAG-Fusion + RRF

优点：

- 不依赖专门的重排模型
- 对“一个问题的多种表述”更鲁棒
- 能提升召回覆盖面

局限：

- 本质上仍然依赖初始检索结果质量
- RRF 是排名融合，不是更深层的语义逐对比较
- 查询扩展如果跑偏，可能引入额外噪声

适合场景：

- 召回不足比排序不准更突出时
- 希望用较轻量方式提升检索质量时

### 路线 B：Cohere Rerank

Notebook 中的配置是：

- 基础召回：`k=10`
- 重排模型：`rerank-english-v3.0`
- 最终保留：`top_n=5`

优点：

- 更像真正的“二阶段精排”
- 能直接对候选文档和查询做更细粒度相关性判断
- 最终上下文更紧凑

局限：

- 需要额外的服务和 API Key
- 会增加一次重排开销
- 当前 Notebook 里使用的是英文重排模型，跨语言表现要结合实际语料验证

适合场景：

- 初次召回已经不差，但前排质量还不够好
- 希望把有限上下文窗口留给更关键文档

# 💕💕💕 八股

<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          检索得到的数据直接提交给LLM去生成答案，但这样检索出来的chunks并不一定完全和上下文相关的问题，最后导致大模型生成的结果质量不佳。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         这个问题很大程度上是因为<strong>召回相关性不够</strong>或者<strong>召回数量太少</strong>导致的，从扩大召回这个角度思考，借鉴推荐系统的做法，引入<strong>粗排</strong>或<strong>重排</strong>的步骤改进效果。
        </span>
      </li>
        <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         重排越来越流行，在上面的过滤策略中，经常用到Embedding计算文档的相似性，然后根据相似性对文档进行排序（包括Fusion），这里的排序被称为<strong>粗排</strong>。我们还可以使用一些专门的排序引擎对文档进一步排序和过滤，这被称为<strong>精排</strong>。每个子问题检索到的文档根据设定的权重进行排序。再基于这个权重，选择top-k的文档。
        </span>
      </li>
      </ul>
</div>

![7841f239-f29b-4cc2-8cd7-508a7483cb59](https://raw.githubusercontent.com/songqi4485/note1/master/picture/7841f239-f29b-4cc2-8cd7-508a7483cb59.png)

解决召回数量太少的方法是原有的top-k向量检索召回扩大召回数目。

Rerank主要有两种实现方式：
<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
 		1.使用一些<strong>专门的排序引擎</strong>对文档进一步排序和过滤。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         2.使用<strong>大模型来做重排序</strong>
        </span>
      </li>
      </ul>
</div>

## 1.使用Cohere的Re-Rank方案

![使用Cohere的ReRank方案](https://raw.githubusercontent.com/songqi4485/note1/master/picture/使用Cohere的ReRank方案.png)

除此之外，排序引擎还有JinaRerank、SentenceTransformerRerank、Colbert Reranker：

<div style="
  background: #f0fdf4;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
 		Jina AI总部位于柏林，是一家领先的AI公司，提供一流的嵌入、重排序和提示优化服务，实现先进的多模态人工智能。可以使用Jina提供的Rerank API对文档进行精排。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         除了使用商业服务，我们也可以使用一些本地模型来实现重排序。比如sentence-transformer包中的<strong>交叉编码器(Cross Encoder)</strong>可以用来重新排序节点。<br>LIamaIndex默认使用的是cross-encoder/ms-marco-TinyBERT-L-2-v2模型，这个是速度最快的。为了权衡模型的速度和准确性，请参考sentence-transformer文档。
        </span>
      </li>
        <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         另一种实现本地重排序的是ColBERT模型，它是一种快速准确的检索模型，可以在几十毫秒对大文本集合进行基于BERT的搜索。
        </span>
      </li>
      </ul>
</div>



## 2.大模型做重排序

<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 16px; margin-right: 8px;">🎀
    使用大模型来做重排序，<strong>将文档丢给大模型，然后让大模型对文档的相关性进行评分，从而实现文档的重排序。</strong><br>
    使用LLM来决定哪些文档/文本块与给定查询相关。prompt由一组候选文档组成，这时LLM的任务是选择相关的文档集，并用内部指标对其相关性进行评分。为了避免因为大文档chunk化带来的内容分裂，在建库阶段也可做一定优化，<strong>利用summary index对大文档进行索引。</strong> 
    </span>
</div>

基于LLM召回或重排存在一些缺陷，首先就是**慢**；其次就是**增加了LLM的调用成本**；最后由于打分是分批进行的，存在着无法全局对齐的问题。

除此之外，使用LLM进行排序的相关论文方法还有：
<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          RankGPT 是 Weiwei Sun 等人在论文 Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents 中提出的一种基于大模型的 zero-shot 重排方法，它采用了排列生成方法和滑动窗口策略来高效地对段落进行重排序，具体内容可以参考 RankGPT 的源码。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         RankLLM 和 RankGPT 类似，也是利用大模型来实现重排，只不过它的重点放在与 FastChat 兼容的开源大模型上，比如 Vicuna 和 Zephyr 等，并且对这些开源模型专门为重排任务进行了微调，比如 RankVicuna 和 RankZephyr 等。
        </span>



# 3.Retrieval(CRAG)

# 💕💕💕 八股

其本质上是一种**Adaptive-RAG策略**，实现方式为在循环单元测试中自我纠正检索错误，以确定文档相关性并返回到网格搜索，即对检索文档的自我反思/自我评分，主要步骤如下：

![CRAG](https://raw.githubusercontent.com/songqi4485/note1/master/picture/CRAG.png)

<div style="
  background:#fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          首先需要知道的是CRAG<strong>发生在retrieval之后</strong>，即当我们获得了近似的document之后。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         然后会进入一个额外的环节，叫<strong>Knowledge Correction</strong>。我们会先对retrieval得到的每一个<strong>相关切片snippets进行评估</strong>，评估一下获取到的snippet是不是对问的问题有效？（此处重点：评估也是一个LLM）
     </span>

然后会有三种情况：
<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          Correct：那就直接进行RAG正常流程。（不过图中是加了进一步优化）
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         Incorrect：那就直接丢弃原来的document，直接<strong>去web里搜索</strong>相关信息
     	</span>
       </li>
         <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         Ambiguous：对于模糊不清的，就两种方式都要
     	</span>
       </li>

那么在最后的generation部分，也是根据三种不同的情况分别处理。

<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          之前是correct，那现在就直接拼接问题和相关文档。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         之前是incorrect，那现在就直接拼接问题和web获取的信息
     	</span>
       </li>
         <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         之前是ambiguous，那现在就拼接三个加起来
     	</span>
       </li>

以上是CRAG的原始大概逻辑，但在Langchain中对此进行简化：

![CRAG2](https://raw.githubusercontent.com/songqi4485/note1/master/picture/CRAG2.png)

<div style="
  background:#fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         当Incorrect的时候，直接去web上search了（先经过一个transformer_query对问题进行重写，变成更适合web搜索的形式。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
         当至少有一个文档超过了相关性阈值（Correct），则进入生成阶段。
     </span>