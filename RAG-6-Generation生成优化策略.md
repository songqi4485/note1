# 总览

![6generation](https://raw.githubusercontent.com/songqi4485/note1/master/picture/6generation.png)

# 1. Retrieval(Self-RAG)

和CRAG的核心一样，都是self-不是reflective，即当发现结果不那么有效时，要**通过环回溯到之前的步骤去优化**。其基本思想在于：使用**循环单元测试自行纠正RAG错误，以检查文档相关性、答案幻觉和答案质量。**

![6Self-RAG](https://raw.githubusercontent.com/songqi4485/note1/master/picture/6Self-RAG.png)

**和CRAG不一样**的是，Self-ARG的流程是**从检索前就开始**进行的，**评估三次**：

![6Self-RAG的实现](https://raw.githubusercontent.com/songqi4485/note1/master/picture/6Self-RAG的实现.png)



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
          <strong>首先先判断问题是否需要retrieval</strong>：如上图右下角，此处问题是写一篇essay，其实根本没必要去retrieval，直接放入LLM就行。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">2.</span>
           <strong>再判断是否检索到了有关relevant</strong>：当问题需要检索的时候，我们会将<strong>得到的每个document snippet分别判断是否相关（类似CRAG）：</strong><br>
           -如果无关，那就不进行下一步<br>
           -如果有关，对所有snippets都判断后，按照相关性进行排序，然后依次送到LLM中进行最后的步骤。<br>
         </span>
      </li>
         <li style="margin-bottom: 10px;">
      <span style="color: #333;">
      <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">3.</span>
      <strong>最后判断生成的内容是否有效。</strong>
     </span>
  </li>



**总结：**



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
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">检索阶段判断（是否进行检索）：</span>
          如果不需要检索，Self-RAG会像常规语言模型一样预测响应的下一个部分。如果需要检索，它会发出一个检索过程的信号，并要求外部检索模块找到相关文档。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">生成阶段判断（检索到的文本是否相关）：</span>
          如果需要检索，则进一步评估检索到的文档是否支持生成内容（能否基于检索到的文档生成内容），然后根据所发现的内容生成响应的下一个部分。 
         </span>
      </li>
         <li style="margin-bottom: 10px;">
      <span style="color: #333;">
      <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">最终结果判断（生成的结果是否正确）：</span>
      如果检索到的文档能支持生成内容，则生成内容。并且评估生成响应的整体质量。如果整体质量好，则结束本次响应。否则，再次从头开始检索。
     </span>
  </li>



**补充：**

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
   除了上述的CRAG和Self-RAG，还有Adaptive-RAG，<strong>根据查询的复杂度自适应地选择最合适的检索策略</strong>。这包括使用一个小型的<strong>语言模型作为分类器来预测查询的复杂度，并根据复杂度级别选择相应的检索策略（如迭代式、单步式或无检索方法）</strong>。Adaptive-RAG的核心思想在于<strong>通过动态调整</strong>来平衡处理简单查询和复杂查询时的效率和准确性。后文会详细介绍。
</div>



# 2.自我总结

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
          上下文压缩：使用上下文压缩历史对话和问题和结果。
        </span>
      </li>
      <li style="margin-bottom: 10px;">
          <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">2.</span>
           路由配置：路由配置备用的大模型，不同的Prompts和助手（艺术专业助手、计算机专业助手）。两种路由选择方式：一种是根据用户输入返回对应的链（手动，例如根据当前登陆用户判断需要什么助手）；一种是计算的一个提示词与问题的相似度，选择相似度高的路由（自动，LLM判断选择哪个路由）。
         </span>
      </li>
        <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">3.</span>
          元数据：在文件级别、段落级别添加元数据，根据元数据进行过滤，可以加快检索，元数据例如：where={"metadata_field": "is_equal_to_this"},# 元数据过滤条件 where_document={"$contains": "search_string"} # 文档内容过滤条件。
        </span>
      </li>
        <li style="margin-bottom: 10px;">
      <span style="color: #333;">
      <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">4.</span>
      查询sql的问题：大模型调用SQL可能会有错（需要知道表结构，工作量大），而且直接访问数据库，可能影响数据库性能。直接查询接口相对来说会好一点，接口有容灾。5.集成/混合检索器：多种检索器结合，使用RRF对检索的结果进行Fusion，再进行精排
     </span>
  </li>
         <li style="margin-bottom: 10px;">
        <span style="color: #333;">
          <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">5.</span>
          集成/混合检索器：多种检索器结合，使用RRF对检索的结果进行Fusion，再进行精排。
        </span>
      </li>
        <li style="margin-bottom: 10px;">
      <span style="color: #333;">
      <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">6.</span>
      text-embedding-3-small 8000 4000 200和gpt-3.5-turbo-16k 160007。
     </span>
  </li>
        </li>
        <li style="margin-bottom: 10px;">
      <span style="color: #333;">
      <span style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700;">7.</span>
      数据处理阶段：增加元数据、数据增强；文档太大，上下文压缩；文档太小，句子窗口搜索；index时候英语文档翻译成中文，查询时候的中文翻译成英语
     </span>
  </li>

