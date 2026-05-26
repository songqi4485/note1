Query Translation查询转换侧重于**重写**和（或）**修改问题**，使得问题转换为更好解决的问题，更方便检索。

![edda320d-0b24-4d47-a765-bf98c3633239](https://raw.githubusercontent.com/songqi4485/note1/master/picture/edda320d-0b24-4d47-a765-bf98c3633239.png)

# 5. Multi Query多查询策略

## 5.1 核心思想

LangChain 官方文档指出，`MultiQueryRetriever` 会利用 LLM 为同一个问题生成**多个不同视角的查询**，**对每个查询分别检索**，再把**结果做唯一并集**，从而缓解距离检索对措辞变化过于敏感的问题。

## 5.2 实施步骤

原始问题
 -> 生成多个等价/近义/不同视角的问题
 -> 分别检索
 -> 合并去重
 -> 交给生成模型回答

![multi query](https://raw.githubusercontent.com/songqi4485/note1/master/picture/multi%20query.png)

* 优点

​	实现简单、对措辞差异鲁棒、适合大多数普通问答场景

* 缺点

​	改写质量差时会引入噪声。只是“扩展问法”，不一定能解决复杂多跳问题。

* 适用场景

​	用户问题表达可能不标准。知识库同义表达很多。想低成本提升召回率。

## 💕💕💕八股

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
    在同一维度，根据用户输入的问题<strong>生成多个子问题</strong>，对同一问题生成多个视角的提问，
    然后<strong>依次进行检索</strong>，最后将<strong>检索到的文档合并返回</strong>。
</div>



# 6. RAG-Fusion多查询结果融合策略

## 6.1 核心思想

RAG-Fusion 会先为一个问题生成多条相关查询，然后分别检索，再使用 **Reciprocal Rank Fusion（RRF）** 对多路结果做融合排序。相关论文将其概括为：生成多个查询、按倒数排名分数进行融合，以提高答案的准确性、相关性和全面性。

![rag fusion](https://raw.githubusercontent.com/songqi4485/note1/master/picture/rag%20fusion.png)

* 优点

​	比简单并集更重视“多路共同支持”的文档。对排序更友好。适合希望兼顾召回与精度的场景。

* 缺点

​	如果生成出的查询与原问题不够贴近，结果可能会“偏题”。也就是说，Fusion 不是越多越好，关键在于生成查询的相关性。

* 适用场景

​	同一问题存在多个有效检索视角。知识库较大，希望提高候选排序质量。希望优先返回“多路一致认为重要”的文档。

## 💕💕💕八股

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
    RAG Fusion 和 MultiQueryRetriever 基于同样的思路，在 Multi Query 多查询策略<strong>生成子问题并检索的基础</strong>，它对检索结果执行<strong>倒数排名融合（Reciprocal Rank Fusion，RRF）</strong> 算法，使得检索效果更好。。
</div>



# 7. Decomposition 问题分解策略

## 7.1 核心思想

Least-to-Most Prompting 强调：把复杂问题拆成一系列更简单的子问题，并按顺序求解；Decomposed Prompting 则进一步把复杂任务拆成更小的子任务，用模块化方式**分别处理**。两者共同说明：**复杂问题不必一次回答，可以先拆再答**。

* 优点

​	非常适合多跳问题、组合问题、宽问题。中间步骤清晰，可解释性强。方便定位到底是哪一步出了错。

* 缺点

​	子问题生成质量决定上限。链路更长，成本更高。子问题过多时容易累计误差。

* 适用场景

​	“由哪些部分组成”。“为什么会这样”。“先比较 A，再结合 B，最后判断 C”。多跳检索 / 多阶段推理。

## 7.2 实施步骤

复杂原问题
 -> 生成多个子问题
 -> 对每个子问题检索/回答
 -> 聚合中间结果
 -> 生成最终答案

## 💕💕💕八股

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
    在下一个<strong>更简单维度</strong>，将一个复杂问题分解成多个子问题，将问题分解为一组子问题。之后解决这些子问题再进行合并。有两种类型：Answer recursively和Answer individually
</div>

<div style="
 background: #edf9f1;
  border: 1px solid #f0d98c;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    Answer recursively(递归式回答)：- 先生成多个子问题，逐个回答子问题。<strong>后一个子问题可以使用前面已有的问答对作为背景</strong>，最终逐步积累出更完整的答案。这类方式适合“前面的结果会帮助后面的推理”的场景。
- 
</div>

![Answer recursively](https://raw.githubusercontent.com/songqi4485/note1/master/picture/Answer%20recursively.png)

<div style="
 background: #edf9f1;
  border: 1px solid #f0d98c;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    Answer individually(独立回答)：<strong>独立解决每一个问题</strong>，最后将每个答案合并为最终答案。
![Answer individually](https://raw.githubusercontent.com/songqi4485/note1/master/picture/Answer%20individually.png)

<div style="
 background: #edf9f1;
  border: 1px solid #f0d98c;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    与 Multi Query的区别。Multi Query 针对同一个原始问题，生成多个不同表述、不同角度但语义基本等价的查询，用来提升召回。<strong>本质上还是在回答同一个问题。</strong><br>
 <span style="font-size: 20px; margin-right: 8px;">🎀</span>
    Decomposition 问题分解策略针对一个复杂问题，先拆成若干个相互独立或互补的子问题，然后<strong>每个子问题单独检索、单独回答</strong>，最后再汇总成最终答案。<strong>本质上是在回答多个子问题，再做整合。</strong>



# 8. Step Back 问答回退策略

## 8.1 核心思想

​	在更简单的维度，基于用户的原始问题**生成一个后退问**题，后退问题相比原始问题具有更高级别的概念或原则，从而提高解决复杂问题的效果。

### 优点

- 对细节多、背景要求强的问题很有效
- 有助于补足“具体问句找不到，但相关背景很多”的情况
- 适合需要原则、定义、历史脉络的问题

### 缺点

- 抽象过度会丢失用户真正想问的点
- 更适合“补背景”，不一定适合非常精确的事实检索

### 适用场景

- 原问题特别具体
- 需要调用世界知识或一般原理
- 文档库更偏“背景知识”而不是“直接问答对”

## 8.2 实施步骤

具体问题
 -> 改写成更通用、更抽象的问题
 -> 检索高层背景知识
 -> 将原问题上下文 + 抽象问题上下文一起交给模型
 -> 回答原问题

## 💕💕💕八股

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
    构成上包括<strong>抽象abstraction</strong>和<strong>推理reasoning</strong>两个步骤，比如给定一个问题，需要提示大模型，找到回答该问题的一个前置问题，得到前置问题及其答案后，再将其整体与当前问题进行合并，最后送入大模型进行问答，得到最终答案。例如一个关于物理学的问题可以后退为一个关于该问题背后的物理原理的问题，然后对原始问题和后退问题进行检索。
</div>

![Step Back](https://raw.githubusercontent.com/songqi4485/note1/master/picture/Step%20Back.png)



# 9. HyDE假设性文档嵌入

## 9.1 核心思想

HyDE（Hypothetical Document Embeddings）的思想是：不是直接拿问题去做 embedding，而是**先让 LLM 写出一段“假想的答案文档”**，再对这段文档做 embedding 检索。论文指出，这个假想文档虽然可能包含不真实细节，但它能更好地表达“相关性模式”；随后由真实语料库中的邻近文档把它“落地”，并通过 dense bottleneck 过滤掉错误细节。 

### 优点

- 特别适合零样本检索
- 能把短问题扩展成更丰富的语义表示
- 对“问题很短、文档很长”的语义桥接很有帮助

### 缺点

- 假想文档可能幻觉很强
- 如果生成文本跑偏，检索也会跟着跑偏
- 因此 HyDE 更适合“拿生成文本做检索入口”，而不是直接把生成文本当事实答案

### 适用场景

- 语料和问题表述风格差异大
- 原问题太短、信息太稀
- 缺少标注数据，只能做零样本或弱监督检索

## 9.2 实施步骤

原始问题
 -> 生成一段假想回答/文档
 -> 对假想文档做向量检索
 -> 找到真实语料中的邻近文档
 -> 用真实文档回答原问题

## 💕💕💕八股

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
    使用基于相似性的向量检索时，在原始问题上进行检索可能效果不佳，因为它们的嵌入可能与相关文档的嵌入不太相似，但是，如果让大模型生成一个假设的相关文档，然后使用它来执行相似性检索可能会得到意想不到的结果。这就是 假设性文档嵌入（Hypothetical Document Embeddings，HyDE） 背后的关键思想。
</div>

![HyDE](https://raw.githubusercontent.com/songqi4485/note1/master/picture/HyDE.png)



<div style="
 background: #edf9f1;
  border: 1px solid #f0d98c;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span>
   注意，HyDE 可能出现的两个失败场景：<br>
&nbsp;&nbsp;1.<strong>在没有上下文的情况下</strong>，HyDE 可能会<strong>对原始问题产出误解</strong>，导致检索出误导性的文档；比如用户问题是 “What is Bel?”，由于大模型缺乏上下文，并不知道 Bel 指的是 Paul Graham 论文中提到的一种编程语言，因此<strong>生成的内容和原文完全没有关系</strong>，导致检索出和用户问题没有关系的文档；<br>
&nbsp;&nbsp;2.<strong>对开放式的问题，HyDE 可能产生偏见</strong>；比如用户问题是 “What would the author say about art vs. engineering?”，这时大模型会随意发挥，生成的内容可能带有偏见，从而导致检索的结果也带有偏见；

<div style="
 background: #fff7ed;
  border: 1px solid #f2c38b;
  border-radius: 10px;
  padding: 16px 20px;
  color: #333;
  line-height: 1.8;
  font-size: 16px;
">
  <span style="font-size: 20px; margin-right: 8px;">🎀</span><br>
&nbsp;&nbsp;上面的 <strong>查询重写</strong>（Query Rewriting），都是为了处理表达不清的用户输入，和处理聊天场景中的<strong>后续问题（Follow Up Questions）</strong>。<br>
&nbsp;&nbsp;<strong>查询压缩</strong>（Query Compression），用户可能是以<strong>聊天对话</strong>的形式与系统交互的，为了正确回答用户的问题，我们需要<strong>考虑完整的对话上下文</strong>，为了解决这个问题，可以<strong>将聊天历史压缩成最终问题</strong>以便检索。
</div>






