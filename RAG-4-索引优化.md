# 概览

![38cd8d9c-df5d-489f-abba-bf21b609f9fe](https://raw.githubusercontent.com/songqi4485/note1/master/picture/38cd8d9c-df5d-489f-abba-bf21b609f9fe.png)

# 1. 主线

**丰富索引结构**本质上在回答：**为什么 RAG 不能只把原始 chunk 做 embedding 然后 top-k 检索就结束**。

围绕三类索引增强思路展开：

* **Multi-representation Indexing**：同一文档构造多种表示，再用这些表示做检索。
* **RAPTOR**：把文档构造成树状层级摘要索引。
* **ColBERT**：不是“一段文本一个向量”，而是“每个 token 一个上下文化向量”的 late interaction 检索。

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
      color: #2563eb;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">分块策略（Chunking）</span>
          简单、更好的存储数据 split，chunk，overlap（简单而直观的数据分割存储方法）。有多种分块方法，可以分解多种格式的文件，以及Embedding模型。<br>
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">多重表示索引（Multi-representation Indexing）</span>
           <strong>先生成文档摘要（"命题"）。再进行相似性搜索，但将完整文档返回给LLM进行生成。</strong>
         </span>
       </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">专用嵌入（Specialized Embeddings）</span>
           <strong>为文档生成特定的向量嵌入，</strong>便于高校的相似性计算。例如使用Colbert专一领域的生成索引的方式。
         </span>
       </li>
         <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">分层索引（Hierarchical Indexing）</span>
           <strong>构建多层次的摘要索引树，</strong>将文档在不同抽象层次上进行摘要。
         </span>
       </li>



# 2. 多表示索引（Multi-Representation Indexing）

## 2.1 原理

LangChain 官方把多表示索引总结为三类常见做法：

* **更小的块**：将文档拆成更细粒度子块
* **摘要**：为每个文档生成摘要并嵌入
* **假设性问题**：为每个文档生成它能回答的问题并嵌入



## 2.2 💕💕💕八股



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
    除了存储文本的向量，还有图像和表格的索引。会对<strong>图片和表格</strong>的<strong>summary进行向量存储</strong>【可以使用多模态模型进行生成或者将图片/表格原文的相关上下文进行摘要选取】，同时<strong>保留原文(图像和表格）及其对应关系</strong>，例如parent-document-retrival。Multi-Representation Indexing，使用LLM生成针对检索进行优化的文档摘要("命题")。嵌入这些<strong>摘要以进行相似性搜索</strong>，但<strong>将完整文档返回</strong>给LLM进行生成。
</div>



---

FLOW:

![4多表示索引](https://raw.githubusercontent.com/songqi4485/note1/master/picture/4多表示索引.png)

![4多索引表示2](https://raw.githubusercontent.com/songqi4485/note1/master/picture/4多索引表示2.png)

**建库阶段：**

1. 输入原始文档 / 表格 / 图片
2. 用模型生成摘要或描述
3. 对摘要做向量化并写入 Vectorstore
4. 将原始内容写入 Docstore
5. 用 `doc_id` 建立两者关联

 **检索阶段：**

1. 用户输入问题
2. **在 Vectorstore 中查找最相关摘要**
3. **通过 `doc_id` 找到对应原始内容**
4. 从 Docstore 取出完整文档 / 表格 / 图片
5. 将原始内容作为上下文交给大模型回答



# 3. RAPTOR（层级树状索引）

## 3.1 原理

​				**Recursive Abstractive Processing for Tree-Organized Retrieval**

​	长文档RAG根本问题：多问题既需要局部事实，又需要跨章节、跨段落的全局主题整合，而普通 top-k chunk 检索只会拿到几个局部片段，难以覆盖整篇材料的语篇结构。

​	RAPTOR：**把文档从底向上建成一棵“原文块—聚类摘要—更高层摘要”的树**。查询时，不只从叶子节点取上下文，而是能同时命中不同抽象层级的节点。这样，高层问题能命中摘要节点，细节问题能命中叶子节点。



## 3.2 💕💕💕八股



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
    传统的RAG方法通常仅<strong>检索较短的连续文本块</strong>，这<strong>限制</strong>了对<strong>整体文档上下文的全面理解。</strong> 散乱的内容详略不一致的很多文档，如何进行有效分类和整理？PAPTOR通过<strong>递归嵌入、聚类和总结文本块，构建一个自底向上的树形结构</strong>，在推理时从这棵树中检索信息，从而<strong>在不同抽象层次上整合长文档的信息</strong>。<br>
    这是<strong>层级性索引的方案</strong>其思想在于<strong>对文档进行生成聚类摘要</strong>，然后将设计成<strong>层级性</strong>。
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
  <span style="font-size: 20px; margin-right: 8px;">🎀<strong>树形结构构建</strong></span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #2563eb;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">文本分块：</span>
          首先将检索语料库分割成短的、连续的文本块。<br>
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">嵌入和聚类：</span>
           使用SBERT（基于BERT的编码器）将这些文本块嵌入，然后使用高斯混合模型（GMM)进行聚类。
         </span>
       </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">摘要生成：</span>
           对聚类后的文本块使用语言模型生成摘要，这些摘要文本再重新嵌入，并继续聚类和生成摘要，直到无法进一步聚类，最终构建出多层次的树形结构。
         </span>
       </li>



---

FLOW:

![4多索引表示3](https://raw.githubusercontent.com/songqi4485/note1/master/picture/4多索引表示3.png)

![4多索引表示4](https://raw.githubusercontent.com/songqi4485/note1/master/picture/4多索引表示4.png)

工作流程（上图）：

1. 输入原始文档

2. 对原始文档做向量化并聚类

   先把每个文档块转成向量表示，再根据语义相似度把它们分成若干簇（clusters）。

3. 对每个簇生成摘要

   每个簇不再只是一堆原始文本，而是生成了一个更抽象、更浓缩的“父节点摘要”。

4. 对这些摘要再次向量化并聚类

   对这些摘要做 embedding，再按语义相似性 clustering，也就是把“摘要节点”继续往上合并。

5. 再次生成更高层摘要

   对新的摘要簇继续生成更高层 summary，形成更抽象的父节点。

6. 形成一棵自底向上的摘要树

   最终构造成一棵树。**叶子层**：原始文本块；**中间层**：多个 cluster summary；**根层**：最高层的全局摘要。

7. 原始文档和各层摘要都会进入向量库

总结：**原文 → 向量化 → 聚类 → 摘要 → 再聚类 → 再摘要 → 最终形成多层摘要树，并把原文与各层摘要一起用于检索**

---

<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀<strong>查询方法</strong></span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #2563eb;
    ">
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
            <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;"><strong>树遍历</strong>：</span>
          从树的根层开始，<strong>逐层选择与查询向量余弦相似度最高的节点</strong>，直到到达叶节点，将所有选中的节点文本拼接形成检索上下文。<br>
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;"><strong>平铺遍历</strong>：</span>
           将整个树结构平铺成一个单层，将<strong>所有节点同时进行比较</strong>，选出与查询向量余弦相似度最高的节点，直到达到预定义的最大token数。
         </span>
       </li>

![4查询方法](https://raw.githubusercontent.com/songqi4485/note1/master/picture/4查询方法.png)

<div style="
  background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀<strong>实验结果</strong></span>
    <ul style="
      margin: 0;
      padding-left: 22px;
      color: #333;
    ">
      RAPTOR在多个任务上显著优于传统的检索增强方法，特别是在涉及复杂多步推理的问答任务中。RAPTOR与GPT-4结合后，在QuALITY基准上的准确率提高了20%。  
      <li style="margin-bottom: 10px;">
        <span style="color: #333;">
            <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;"><strong>代码</strong>：</span>
          RAPTOR的源代码将在GitHub上公开。<br>
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;"><strong>数据集</strong>：</span>
           实验中使用了NarrativeQA、QASPER和QuALITY等问答数据集。
         </span>
       </li>
        <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;"><strong>视频、论文、代码</strong>：</span>
           https://www.youtube.com/watch?v=jbGchdTL7d0；https://arxiv.org/pdf/2401.18059.pdf；https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb
         </span>
       </li>


# 4. ColBERT（Late Interaction 多向量检索）

## 4.1 原理

* 查询不是一个向量，而是**每个 token 一个向量**；
* 文档也不是一个向量，而是**每个 token 一个向量**；
* 检索时不是 query 向量和 doc 向量做一次整体相似度，而是做 **late interaction** 的细粒度匹配。

相较于单向量检索，late interaction 模型会在 **token 粒度** 产生多向量表示，并把相关性分解为可扩展的 token-level 计算；同时 ColBERTv2 通过压缩机制，把这类模型的空间开销降低了 **6–10 倍**。

## 4.2 💕💕💕八股

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
      ColBERT是高效精准的检索模型。ColBERT<strong>为段落中的每个标记</strong>生成一个<strong>受上下文影响的向量</strong>。ColBERT类似地<strong>为查询中的每个令牌生成向量</strong> 。然后每个文档的得分是每个查询嵌入与任何文档嵌入的最大相似性的综合。<br>
      <strong>特定化的Embedding，之前的方法停留在文本层级，ColBERT做到了token级，</strong>为段落中的每个token生成一个受上下文影响的向量，ColBERT同样为查询中的每个token生成向量。然后，每个文档的得分就是每个查询嵌入与任意文档嵌入的最大相似度之和。

* https://hackernoon.com/how-colbert-helps-developers-overcome-the-limits-of-rag

* RAGatouille |LangChain
* https://til.simonwillison.net/llms/colbert-ragatouille
