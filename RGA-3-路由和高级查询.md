# 概览

![3路由概览](https://raw.githubusercontent.com/songqi4485/note1/master/picture/3路由概览.png)

# 💕💕💕八股

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
  本节主要解决的是从<strong>获取query之后</strong>，<strong>所选择问题域的方案</strong>，包括Logical routing and Semantic routing，<strong>LLM会基于用户的问题</strong>，选择合适的 <strong>逻辑路由(数据源选择)和语义路由(Prompt选择)</strong> 进行分发。
</div>

# 1.  主要内容

这一讲的核心不是“怎么再做一个向量检索”，而是回答一个更重要的问题：

<center><strong>用户的问题来了之后，RAG 系统应该去哪里查、用什么方式查。</strong></center>

因此，这一讲的重点可以概括为两部分：

1. **路由（Routing）**
    决定一个问题应该走哪条处理路径、调用哪种检索方式或数据源。

2. **高级查询（Advanced Query）**

   让系统不只会做普通文本检索，还能处理：

   - 结构化数据查询（Text-to-SQL）
   - 图关系查询（Text-to-Cypher）
   - 带条件过滤的检索（Self-query）

# 2. 核心思想

传统基础 RAG 的流程：**用户问题 → 向量检索 → 大模型生成答案**

但在真实业务里，这种方式并不总是够用。因为用户的问题可能并不都适合查文本。

更完整的 RAG的流程：**用户问题 → 意图识别 / 路由 → 选择合适的数据源与查询方式 → 检索 / 查询 → 组织答案**

# 3. 路由（Routing）

## 3.1 含义

**在多个候选检索路径中，为当前问题选择最合适的一条或多条。**

## 3.2 逻辑路由（Logical Routing）

**先识别问题类型，再根据规则或模型判断走哪条通道。**

![3逻辑路由](https://raw.githubusercontent.com/songqi4485/note1/master/picture/3逻辑路由.png)

​	（1）Question

​	（2）左边大脑：用一个 LLM 来做**分类器**，而且这个分类器不是随便输出自然语言，它要输出**结构化结果**。

​	（3）下方的 `Available retrievers`：表示系统里已经有多个可选的检索目标。LLM 的任务不是回答内容本身，而是**判断这个问题最适合交给哪一个 retriever**。

​	（4） 中间结构化输出：`{database: Vectorstore}`：路由结果。

​	（6） 右边 `Database relevant to the question`：系统根据上一步的结构化结果，选择对应数据库。

常实现的方式：

* 规则路由：根据关键词、正则、模板来判断。
* LLM 路由：让大模型根据工具描述自动选择。
* 混合路由：先用规则粗筛，再用模型细分。

---

![3逻辑路由2](https://raw.githubusercontent.com/songqi4485/note1/master/picture/3逻辑路由2.png)

​	（1）左边的 Question：用户问题进来。

​	（2）左下的 `Structured Output / Pydantic Object`：你**提前定义好一个输出结构**。

​	（3）中间的 `Function Schema`：把这个结构转换成 **LLM 能理解的 schema / function definition**。

​	（4）中间大脑图标：`Bind function to LLM`：**把这个结构约束绑定给模型**。

​	（5） `LLM returns a JSON string matching output schema`：模型产出符合**你定义格式的 JSON**。

​	（6） `Apply parser`：程序再把 JSON 解析成真正的对象。

​	（7）最右边的 `Structured Output`：拿到的是一个**程序可直接使用的结构化对象**。

* **用 structured output 把 LLM 从“会聊天的人”变成“会按格式做分类决策的路由器”。**
* **先定义路由结果的格式 → 让 LLM 按这个格式输出 → 解析成对象 → 根据对象字段做分流。**

## 3.3 语义路由（Semantic routing）

**不用大模型做显式推理，而是把用户问题和候选路由描述都转成向量，再比较相似度。**

![3语义路由](https://raw.githubusercontent.com/songqi4485/note1/master/picture/3语义路由.png)	

​	（1）Question

​	（2） `Embed`：把问题转成向量表示。

​	（3）左下和右下的 `Prompt #1` / `Prompt #2`：提前准备好的**候选 prompt 模板**。

​	（4）中间大脑：相似度匹配 / 路由器：语义匹配器

​	（5）Chosen prompt

​	（6） 第二个大脑 → `Answer`：把“用户问题”加上“被选中的 prompt”。一起送给 LLM，生成最终答案。



# 4. 问题构建优化策略（ Query Construction）

## 4.1核心思想

**用户的问题不一定只对应“相似文本检索”，它还可能对应数据库查询、图查询或带过滤条件的检索。**

![4Query Construction](https://raw.githubusercontent.com/songqi4485/note1/master/picture/4Query%20Construction.png)

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
	在上一节的 Logical routing 中，我们知道数据可能存储在关系型数据库或图数据库中，根据数据的类型，我们将其分为<strong>结构化（SQL 或图数据库）</strong>、<strong>半结构化（将结构化元素与非结构化元素）</strong>和<strong>非结构化（向量数据库）</strong>三大类。​将自然语言与各种类型的<strong>数据无缝连接</strong>是一件<strong>极具挑战的事情</strong>。要从这些库中检索数据，必须使用特定的语法，而用户问题通常都是用自然语言提出的，所以我们需要<strong>将自然语言转换为特定的查询语法</strong>。这个过程被称为<strong>查询构造（Query Construction）</strong>。​查询构造主要有上图中的三种：Text-to-SQL（关系型数据库）、Text-to-Cypher（图数据库）、Self-Query rertriver（向量数据库），除此之外还有半结构化数据（Text-to-SQL + Semantic）。
</div>

**其中向量数据库中常用的是基于元数据过滤器。**

## 4.2 Text-to-SQL

把自然语言问题转换成 SQL，再在关系型数据库中执行。

主要解决向量检索不擅长精确统计和结构化计算的问题。应用场景：

* 销售/财务数据分析
* 订单、库存、用户数据查询
* 统计报表自动问答

## 4.3 Text-to-Cypher

把自然语言问题转换成图数据库查询语言（如 Cypher），再在图数据库中执行。

主要解决普通文本检索不擅长“关系结构表达”的问题。应用场景：

* 知识图谱问答
* 推荐系统关系建模
* 实体关联分析
* 推荐系统关系建模

## 4.4 Self-query

从自然语言问题中，自动提取出可以结构化表示的条件，并转成 metadata filter。

主要解决：**语义上相关，但条件上不满足的误召回问题**。

​	例如用户问：“找 2025 年财务部发布的报销制度”这里其实包含两部分：<br>		查询主题-报销制度    过滤条件-年份 = 2025、部门 = 财务部<br>	如果没有 Self-query，系统只会搜“报销制度”； 这样可能找到很多旧制度、其他部门制度。有了 Self-query，系统会先过滤，再检索。

应用场景：

* 企业文档检索
* 法律/医疗/金融文档库
* 多部门、多租户知识系统

## 💕💕💕八股

**Query structuring for metadata filters 基于元数据查询过滤**

![4基于元数据查询过滤](https://raw.githubusercontent.com/songqi4485/note1/master/picture/4基于元数据查询过滤.png)

许多向量库都包含**元数据字段**。这使得**基于元数据过滤**特定块成为可能。

​	（1）左边自然语言问题。混了两类信息：**内容条件**`chat langchain`和**时间条件**`published after 2024`

​	（2）左下的 `VectorDB Metadata / Pydantic Object`

​		先定义一个**目标查询结构**。意思是你希望模型最后产出一个固定格式的查询对象，而不是随便解释一段话。这里叫 `VectorDB Metadata`，是因为这些字段通常会映射到向量库里的文本检索字段和元数据过滤字段。

​	（3）中间的 `Function Schema`

​		这是把上面的结构，转换成 LLM 能遵守的 schema。

​	（4）Bind function to LLM

​	（5）LLM returns a JSON string matching output schema

​		模型已经完成了“语义解析”。它把原问题拆成了机器更好处理的字段。

​	（6）Apply parser

​		把 JSON 解析成真正对象。

​	（7）最右边的 `Query`

​		最后你得到的就是一个**程序能直接执行的查询**。

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
元数据过滤器是<strong>基于某些特定的元数据属性</strong>（如时间、类别、语言、标签等）来<strong>限定查询的范围</strong>，从而缩小搜索空间，提高检索的精度。<br>
  &nbsp;在<strong>向量数据库</strong>中通常包含两部分元数据字段和主体数据：<br>
 &nbsp;&nbsp;&nbsp;​元数据字段：不向量化，以原始形式（文本、数值、标签等）存储，用于精确过滤。<br>
 &nbsp;&nbsp;&nbsp;​主体数据：需向量化（如文本、图像、音频），转为高维向量后用于相似性搜索。
</div>

