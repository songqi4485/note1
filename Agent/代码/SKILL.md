---
name: "mathmodel-skills"
description: "数学建模竞赛论文全流程自动化技能：从赛题解析、算法选型、数据清洗、建模代码生成、论文微单元生成、图表制作、学术润色、文献检索、质量审计到答辩预演的完整16步流水线。适用于CUMCM/电工杯/美赛等数学建模竞赛。"
---

# 数学建模竞赛论文全流程自动化（MathModel Skills）

---

## ⛔⛔⛔ AI 执行强制规则（最高优先级，加载后立即生效）

> 以下规则优先于本文档的所有其他内容。违反任何一条视为严重执行错误。

### 启动任务时必须先输出自检清单

接到建模论文任务后，**第一件事**是在对话中逐项输出以下确认，全部为"是"后才能开始：

```
启动前自检：
✅/❌ Step 1 赛题摘要将输出给用户确认（防止幻觉）
✅/❌ Step 4 代码将按用户本地环境编写（xgboost/lightgbm/catboost/sklearn），不按 VM 环境降级
✅/❌ Step 3.6 数据预处理章节将会生成（solve_preprocessing.py + 6张图表 + preprocessing_chapter.md）
✅/❌ Step 4 代码写完后将让用户在本地终端运行，不在 VM 中尝试运行
✅/❌ Step 4.5 结果审查将会执行（不跳过）
✅/❌ 字数门禁将会验证（每题≥1500字，全文≥12000字）
✅/❌ Step 6.5 LaTeX 校验将会执行
✅/❌ Step 12 最终审计将会执行
✅/❌ 不会编造数值，所有数值来自代码实际运行
✅/❌ 所有公式使用 LaTeX 格式（$...$或$$...$$）
```

有任何一项为 ❌，必须说明原因并等待用户确认后才能继续。

---

### 步骤进度追踪清单（每步完成后必须更新）

> **⛔ 强制规则：每个步骤执行完毕后，必须在对话中输出更新后的进度清单，将对应项从 `⬜ 待执行` 改为 `✅ 已完成` 或 `⏭️ 已跳过（原因：___）`。禁止静默完成一个步骤而不更新进度清单。**

每次启动任务时，在启动前自检通过后，输出以下进度清单（初始全部为 `⬜`）：

```
📋 流水线进度追踪：
⬜ Step 0   竞赛规范解析         → contest_config.json
⬜ Step 1   赛题解析与模型选型   → A_题意对齐.md / D_模型路线.json（⚠️ 必须用户确认）
⬜ Step 2   算法文献检索         → algo_recommendation.json
⬜ Step 3   数据清洗与可视化     → data_cleaned/ + EDA图表
⬜ Step 3.5 数据质量前置检查     → data_quality_report.json
⬜ Step 3.6 数据预处理章节生成   → solve_preprocessing.py + 6张图 + preprocessing_chapter.md
⬜ Step 4   建模代码生成与执行   → step2_calc_results.json + figures/（⚠️ 用户本地运行）
⬜ Step 4.5 结果审查与三级修复   → review_log.json（⚠️ 门禁：不可跳过）
⬜ Step 5   质量审计与任务清单   → tasks.json（⚠️ 门禁：Step 4.5 完成才可进入）
⬜ Step 6   论文微单元生成与合并 → final_paper.md + final_paper.docx
⬜ Step 6.5 LaTeX校验与修复      → lint_report.json
⬜ Step 7   全流程编排（如需要）
⬜ Step 8   图表精修             → figures/ 出版级图表
⬜ Step 9   论文学术润色
⬜ Step 10  文献检索与引用管理   → 参考文献列表
⬜ Step 11  评委质疑预演
⬜ Step 12  最终质量审计         → PASS/FAIL 报告（⚠️ 门禁：提交前必须执行）
```

**每步完成时输出格式**：

```
✅ Step X 完成 | 耗时：___
   输出文件：___
   关键结果：___（如：R²=0.92, AUC=0.87, 共生成6张图表）
   下一步：等待用户指令 / 自动进入 Step X+1
```

**跳过步骤时输出格式**：

```
⏭️ Step X 跳过
   原因：___（必须填写，如：用户明确指示跳过 / 该步骤不适用于当前竞赛类型）
   风险提示：___（如跳过有潜在影响，必须说明）
```

**当前进度查询**：用户任何时候可以说"进度"或"现在到哪了"，AI 必须立即输出带有最新状态的进度清单。

---

### 三个硬性门禁（BLOCKING GATE）

#### 门禁1：Step 4.5 结果审查（进入 Step 5 的唯一入口）

Step 4 代码运行完毕后，**必须停下来**，在对话中输出指标评估表格：

| 子问题 | 问题类型 | 核心指标 | 当前值 | 达标阈值 | 是否达标 |
|--------|---------|---------|--------|---------|:-------:|
| 问题一 | 回归/分类/优化 | R²/AUC/收敛性 | （填实际值） | R²≥0.15 / AUC≥0.75 | ✅/❌ |

- 出现 ❌：**必须执行至少1轮修复**，展示修复前后指标对比
- 想跳过此步：**必须先输出** `"我正在跳过Step 4.5，原因是：___"` 并等待用户确认
- **禁止从 Step 4 直接跳到 Step 5**

#### 门禁2：字数验证（Step 6 完成后立即执行）

`final_paper.md` 生成后，**必须**统计并输出：

```
字数验证：
- 问题一章节：___字（要求≥1500字）✅/❌
- 问题二章节：___字（要求≥1500字）✅/❌
- 全文总字数：___字（要求≥12000字）✅/❌
```

全文 < 12000字：**禁止交付**，必须补充内容后重新统计。

#### 门禁3：前置文件检查（Step 5 前执行）

检查以下文件，缺失则停止并告知用户：
- `paper_output/step2_calc_results.json`
- `step3_filled_placeholder.py`（`[待填写]` 超过5处需警告）

---

### 内容生成红线

| 违禁行为 | 正确做法 |
|---------|---------|
| 数值"看起来合理"就写入论文 | 必须来自代码实际运行输出 |
| 公式用 Unicode 写（α、Σ、√） | 必须用 LaTeX（`$\alpha$`、`$\sum$`） |
| "如图所示"一句带过 | 必须描述图中具体趋势和数值 |
| "效果良好"等模糊表述 | 必须给出具体指标数值 |
| 在附录后追加内容 | 字数不足时插入对应章节内部 |
| 不检测字体直接用 SimHei | 必须先检测可用性，不可用则用英文标注 |
| Step 之间自动连续执行 | 每步完成后输出 `✅ Step X 完成`，等待用户指令 |
| 多张图片集中堆放，后面统一用一段文字解释 | 每张图后紧跟该图的专属解读段落（"一图一说明"，至少3句；复杂图至少5句） |
| 根据文件名猜测赛题内容（如看到"国赛C题"就套用历年题目） | 必须实际读取 Word 文件，输出摘要让用户确认后才能继续 |
| 因代码运行超时就减少迭代次数/降低模型复杂度 | 超时时改用后台执行或分段运行，不得牺牲模型质量 |
| 一次性生成超长文件导致截断后反复"修复" | 长文件必须分段写入（每段≤150行），禁止在截断后循环修复 |
| 因 pip install 失败就放弃专业库，用基础包手写替代 | 先排查环境问题（解释器路径/虚拟环境），或让用户手动安装后继续 |
| **在 VM/沙箱中尝试 pip install 失败后，用 numpy 重写全部代码** | **禁止降级。代码按用户本地环境编写，VM 跑不了就让用户在本地终端跑** |

---

### ⛔ 代码执行环境强制约束（最高优先级）

> **核心事实**：Claude Code 的执行环境（VM/沙箱）与用户本地环境不同。VM 中只有基础包，但用户本地有完整的机器学习环境。
>
> **绝对禁止的行为**：
> 1. ❌ 在 VM 中尝试 `pip install` → 失败 → 用 numpy/pandas 重写代码
> 2. ❌ 检测 VM 环境中的包 → 发现只有基础包 → 按基础包写代码
> 3. ❌ 在代码中写 `# 由于环境限制，使用简化版本`
> 4. ❌ 用 numpy 手写 XGBoost/随机森林/PSO 等已有成熟库的算法
>
> **正确的执行流程**：
> ```
> Step 4 代码生成：
> 1. 按用户本地环境的完整包列表编写代码（xgboost/lightgbm/catboost/sklearn/scipy 全部可用）
> 2. 代码写完后，不要在 VM 中尝试运行
> 3. 直接输出让用户在本地运行的指令：
>    "请在 VS Code 终端中执行：
>     cd <项目根目录>
>     python code/run_all_models.py
>     跑完后告诉我'跑完了'"
> 4. 等待用户回复"跑完了"
> 5. 读取结果文件，继续 Step 4.5
> ```
>
> **用户本地环境（Windows 全局 Python，路径 `D:\APP\Python 3.13\python.exe`）已安装**：
> - numpy, pandas, matplotlib, seaborn, openpyxl
> - scikit-learn, xgboost, lightgbm, catboost
> - scipy, python-docx
>
> **如果不确定某个包是否可用**：在让用户运行前提醒"如果报 ImportError，请执行 `pip install <包名>`"

---

## 概述

本技能是一套完整的数学建模竞赛论文生产流水线，整合了16个子技能，覆盖从赛题解析到最终交付的全部环节。适用于CUMCM国赛、电工杯、美赛等数学建模竞赛。

**核心理念**：脚本层提供结构骨架，AI层注入实质内容。两者协同才能产出高质量论文。

---

## ⚠️ 使用前必读

### 三个核心输入文件

| 文件 | 必填 | 说明 |
|------|------|------|
| `problem_files/` | ✅ | 赛题PDF/Word + 附件数据 |
| `paper_output/step2_calc_results.json` | ✅ | 建模计算的真实数值结果 |
| `step3_filled_placeholder.py` | ✅ | 论文内容注入文件（从模板复制并填写） |

如果这三个文件都不存在，流程会生成充满`[待填写]`提示的骨架文档——这是正确行为，不是bug。

### 目录结构

```
项目根目录/
├── problem_files/                        # 赛题PDF/Word + 附件数据（必填）
│   ├── spec/                            # 撰写规范（建议）
│   │   └── template.docx               # Word排版模板（强烈建议）
│   └── exemplars/                       # 历年优秀论文PDF（建议）
├── crawled_data/                         # 补充数据（可选）
├── step3_filled_placeholder_template.py  # 内容注入模板（不要直接修改）
├── step3_filled_placeholder.py           # 内容注入文件（从模板复制并填写）
├── code/                                 # 所有求解代码
│   ├── algo_utils.py                    # 算法工具库
│   ├── solve_problem1.py                # 问题一求解脚本
│   ├── solve_problem2.py                # 问题二求解脚本
│   ├── solve_problem3.py                # 问题三求解脚本
│   ├── solve_problem4.py                # 问题四求解脚本
│   └── run_all_models.py                # 一键运行所有问题
├── figures/                              # 所有图表（与论文分离）
│   ├── problem1/
│   ├── problem2/
│   ├── problem3/
│   └── problem4/
├── paper_output/                         # 论文及相关报告
│   ├── contest_config.json              # 竞赛配置（自动生成）
│   ├── step1/                           # 题意对齐与算法推荐
│   │   ├── A_题意对齐.md
│   │   ├── B_论文大纲.md
│   │   ├── C_评分点对齐表.md
│   │   ├── D_模型路线.json
│   │   └── algo_recommendation.json
│   ├── step2_calc_results.json          # 计算结果（建模脚本输出）
│   ├── data_cleaned/                    # 清洗后数据
│   ├── tasks.json                       # 微单元任务清单（自动生成）
│   ├── micro_units/                     # 微单元文本（自动生成）
│   ├── (图表已移至根目录 figures/ 文件夹)
│   ├── data_quality_report.json         # 数据质量前置检查报告（Step 3.5）
│   ├── review_log.json                  # 结果审查Agent修复日志（Step 4.5）
│   ├── word_count_report.json           # 字数验证报告（Step 6）
│   ├── lint_report.json                 # LaTeX校验报告（Step 6.5）
│   ├── final_paper.md                   # 论文Markdown草稿
│   ├── final_paper.docx                 # 最终Word交付物
│   └── ref_check.md                     # 交叉引用检查报告
└── .trae/skills/                        # 技能定义与脚本
```

---

## 完整流水线（16步）

### 阶段一：赛题解析与规划（Step 0-2）

#### Step 0: 竞赛规范解析（contest-spec-analyzer）

**做什么**：从撰写规范和优秀论文中提取格式要求、评分标准、章节比例。

**输入**：
- `problem_files/` 中的赛题文件
- `problem_files/spec/` 中的撰写规范
- `problem_files/spec/template.docx` 中的Word模板（最高优先级）
- `problem_files/exemplars/` 中的历年优秀论文

**输出**：`paper_output/contest_config.json`

**关键提取项**：
- 页数限制、字数要求、纸张规格
- 字体要求（正文宋体/标题黑体）、行距、页边距
- 图表编号格式、参考文献格式
- 各章节的页数占比（从优秀论文学习）
- 图表计划（代码生成 vs Gemini占位符）
- **图表语言**（`figure_language`）：根据竞赛类型自动设定。CUMCM国赛/电工杯默认`"zh"`，MCM/ICM默认`"en"`。写入`contest_config.json`，后续Step 4和Step 8读取此字段决定图表标注语言

**Word模板优先级最高**：如果提供了`.docx`模板，模板中的样式表覆盖所有其他来源。

**contest_config.json新增字段示例**：
```json
{
  "figure_language": "en",
  "page_limit": 20,
  "font_body": "SimSun",
  ...
}
```

---

#### Step 1: 赛题解析与模型选型（problem-doc-model-selector）

**做什么**：解析赛题PDF/Word，提取每道子问题的任务、数据、约束，生成题意对齐与模型路线。

> **⛔ 赛题解析防幻觉门禁（BLOCKING GATE）：赛题内容必须由用户确认后才能继续**
>
> AI 存在根据文件名"猜测"赛题内容的风险（如看到"国赛C题"就套用记忆中的历年C题），即使成功读取了 Word 文件，解析结果也可能混入训练数据中的错误记忆。为防止此类幻觉，**必须执行以下流程**：
>
> **第一步：读取 Word 文件**
> 读取 `problem_files/` 中的赛题 `.docx` 文件，提取全部文字内容。如果工具无法解析，**必须立即告知用户**：
> ```
> ⚠️ 无法读取 Word 文件。请将赛题核心内容直接粘贴到对话中。
> ```
>
> **第二步：输出赛题摘要供用户确认**
> 无论用何种方式获取到赛题内容，**必须**在对话中输出以下摘要并等待用户确认：
> ```
> 赛题解析摘要（请确认是否正确）：
> - 竞赛名称：___
> - 题目类型：___（如：优化/预测/评价/规划）
> - 研究背景：___（用1-2句话描述问题背景）
> - 子问题数量：___
> - 问题一：___（核心任务是什么）
> - 问题二：___（核心任务是什么）
> - 问题三：___（如有）
> - 问题四：___（如有）
> - 数据文件：___（列出附件名称）
> - 特殊约束：___（如页数限制、格式要求）
>
> 以上内容是否正确？如有偏差请直接指出，我将根据你的更正重新解析。
> ```
>
> **第三步：等待用户回复**
> - 用户回复"正确"或"没问题"→ 继续执行 Step 1 后续内容
> - 用户指出错误 → 根据用户更正重新输出摘要，再次确认，直到用户确认正确
> - **禁止在用户确认前生成任何模型选型、大纲或论文内容**

**三条强制规则**：
1. **至少一道题必须使用启发式算法**（PSO/GA/SA），作为论文亮点
2. **每道题必须有图表计划**，不允许"无图无论"
3. **检查Word模板**，确认格式要求

**输出文件**：
- `paper_output/step1/A_题意对齐.md` — 每道题的任务分解
- `paper_output/step1/B_论文大纲.md` — 章节结构
- `paper_output/step1/C_评分点对齐表.md` — 评分点与论文位置映射
- `paper_output/step1/D_模型路线.json` — 模型选型配置

---

#### Step 2: 算法文献检索（mathmodel-algo-researcher）

**做什么**：通过网络检索2022年后的学术文献，为每道子问题找到有文献支撑的算法方案。

**三条强制规则**：
1. **文献年份下限：2022年及之后**（2022年后文献不足可放宽到2020年）
2. **预测模型必须是近年新颖方法**（禁止将线性回归/ARIMA等作为主模型，仅可作基线）
3. **启发式优化必须存在**（若主模型不含内置启发式，必须附加PSO/GA调参步骤）

**候选算法模型库（按问题类型分类）**：

> 以下所有模型均可用用户本地已安装的包实现（scikit-learn / xgboost / lightgbm / catboost / scipy / numpy / pandas）。
> 标注 `[需PyTorch]` 的模型需要额外安装 PyTorch，使用前需确认用户环境可用。

**一、预测/回归类**

| 子类 | 模型 | 适用场景 | 实现包 |
|------|------|---------|--------|
| 梯度提升 | XGBoost | 结构化数据通用 | xgboost |
| 梯度提升 | LightGBM | 大数据集/高维特征 | lightgbm |
| 梯度提升 | CatBoost | 含类别特征 | catboost |
| 集成学习 | 随机森林（RF） | 通用基线/特征重要性 | sklearn |
| 集成学习 | ExtraTrees | 高维数据/快速训练 | sklearn |
| 集成学习 | AdaBoost | 弱学习器提升 | sklearn |
| 集成学习 | GradientBoostingRegressor | 小数据集精调 | sklearn |
| 集成学习 | Stacking（多模型堆叠） | 综合多模型优势 | sklearn |
| 集成学习 | Voting（投票集成） | 简单集成 | sklearn |
| 线性模型 | Ridge / Lasso / ElasticNet | 多重共线性/特征选择 | sklearn |
| 线性模型 | 多元线性回归 | 基线对比 | sklearn |
| 支持向量机 | SVR（RBF/Linear/Poly核） | 小样本非线性 | sklearn |
| 近邻方法 | KNN回归 | 局部模式 | sklearn |
| 神经网络 | MLPRegressor | 非线性拟合 | sklearn |
| 贝叶斯 | BayesianRidge | 不确定性估计 | sklearn |
| 时序模型 | ARIMA / SARIMA | 时间序列基线 | statsmodels（需确认） |
| 深度学习 | LSTM / GRU | 时序预测 | [需PyTorch] |
| 深度学习 | CNN-LSTM | 时空特征 | [需PyTorch] |
| 深度学习 | Informer / PatchTST | 长序列预测 | [需PyTorch] |

**二、分类类**

| 子类 | 模型 | 适用场景 | 实现包 |
|------|------|---------|--------|
| 梯度提升 | XGBClassifier | 通用分类 | xgboost |
| 梯度提升 | LGBMClassifier | 大规模分类 | lightgbm |
| 梯度提升 | CatBoostClassifier | 含类别特征 | catboost |
| 集成学习 | 随机森林分类器 | 通用/特征重要性 | sklearn |
| 集成学习 | ExtraTreesClassifier | 高维 | sklearn |
| 集成学习 | AdaBoostClassifier | 不平衡数据 | sklearn |
| 集成学习 | StackingClassifier | 多模型融合 | sklearn |
| 支持向量机 | SVC（RBF/Linear/Poly核） | 小样本 | sklearn |
| 近邻方法 | KNN分类 | 局部模式 | sklearn |
| 线性模型 | Logistic回归 | 基线/可解释性 | sklearn |
| 贝叶斯 | 朴素贝叶斯（Gaussian/Multinomial） | 文本/高维稀疏 | sklearn |
| 神经网络 | MLPClassifier | 非线性分类 | sklearn |
| 判别分析 | LDA / QDA | 低维/正态分布 | sklearn |
| 决策树 | CART / ID3 | 可解释性/规则提取 | sklearn |

**三、聚类/无监督类**

| 子类 | 模型 | 适用场景 | 实现包 |
|------|------|---------|--------|
| 划分式 | K-Means | 通用聚类 | sklearn |
| 划分式 | K-Means++ | 改进初始化 | sklearn |
| 划分式 | Mini-Batch K-Means | 大数据集 | sklearn |
| 层次式 | 凝聚层次聚类（Ward/Complete/Average） | 层次结构发现 | sklearn |
| 密度式 | DBSCAN | 任意形状/噪声检测 | sklearn |
| 密度式 | OPTICS | 变密度聚类 | sklearn |
| 密度式 | HDBSCAN | 自动确定簇数 | sklearn（≥1.3） |
| 概率式 | 高斯混合模型（GMM） | 软聚类/密度估计 | sklearn |
| 降维+聚类 | PCA + K-Means | 高维数据 | sklearn |
| 降维+聚类 | t-SNE + DBSCAN | 可视化+聚类 | sklearn |

**四、优化类**

| 子类 | 模型 | 适用场景 | 实现包 |
|------|------|---------|--------|
| 启发式 | PSO（粒子群优化） | 连续参数优化 | scipy + numpy 手写 |
| 启发式 | GA（遗传算法） | 离散/组合优化 | numpy 手写 |
| 启发式 | SA（模拟退火） | 全局搜索/避免局部最优 | scipy.optimize.dual_annealing |
| 启发式 | DE（差分进化） | 连续全局优化 | scipy.optimize.differential_evolution |
| 数学规划 | 线性规划（LP） | 线性约束优化 | scipy.optimize.linprog |
| 数学规划 | 整数规划（ILP） | 0-1决策 | scipy + numpy（分支定界） |
| 数学规划 | 非线性规划（NLP） | 非线性约束 | scipy.optimize.minimize |
| 数学规划 | 二次规划（QP） | 二次目标函数 | scipy.optimize.minimize |
| 多目标 | NSGA-II | 帕累托前沿 | numpy 手写 |
| 多目标 | 加权求和法 | 多目标转单目标 | scipy |
| 动态规划 | 背包问题/路径规划 | 最优子结构 | numpy 手写 |
| 图论 | Dijkstra / Floyd | 最短路径 | scipy.sparse.csgraph |
| 图论 | 最小生成树（Kruskal/Prim） | 网络优化 | scipy.sparse.csgraph |
| 网络流 | 最大流/最小割 | 网络分配 | scipy / networkx（需确认） |

**五、评价/决策类**

| 子类 | 模型 | 适用场景 | 实现包 |
|------|------|---------|--------|
| 综合评价 | TOPSIS（逼近理想解排序） | 多指标排序 | numpy/pandas 手写 |
| 综合评价 | 熵权法 | 客观赋权 | numpy 手写 |
| 综合评价 | AHP（层次分析法） | 主观赋权 | numpy 手写 |
| 综合评价 | 模糊综合评价 | 模糊不确定性 | numpy 手写 |
| 综合评价 | 灰色关联分析 | 小样本/贫信息 | numpy/pandas 手写 |
| 综合评价 | DEA（数据包络分析） | 效率评价 | numpy + scipy.linprog |
| 综合评价 | RSR（秩和比法） | 排序评价 | pandas 手写 |
| 组合赋权 | 熵权法 + AHP（主客观组合） | 综合赋权 | numpy 手写 |
| 组合赋权 | CRITIC法 | 基于对比强度和冲突性 | numpy 手写 |

**六、特征工程/降维类**

| 子类 | 模型 | 适用场景 | 实现包 |
|------|------|---------|--------|
| 降维 | PCA（主成分分析） | 去相关/降维 | sklearn |
| 降维 | LDA（线性判别分析） | 有监督降维 | sklearn |
| 降维 | t-SNE | 高维可视化 | sklearn |
| 降维 | UMAP | 高维可视化（保持全局结构） | 需确认 |
| 特征选择 | 方差阈值法 | 去除低方差特征 | sklearn |
| 特征选择 | 互信息 | 非线性相关性 | sklearn |
| 特征选择 | RFE（递归特征消除） | 逐步剔除 | sklearn |
| 特征选择 | L1正则化（Lasso） | 稀疏特征选择 | sklearn |
| 特征选择 | 基于树模型的特征重要性 | XGBoost/RF feature_importances_ | sklearn/xgboost |
| 特征选择 | SHAP值 | 可解释性+特征重要性 | shap（需确认） |
| 特征选择 | Boruta | 全相关特征选择 | numpy + sklearn 手写 |

**七、统计分析/检验类**

| 子类 | 模型 | 适用场景 | 实现包 |
|------|------|---------|--------|
| 相关分析 | Pearson / Spearman / Kendall | 变量相关性 | scipy.stats / pandas |
| 假设检验 | t检验 / F检验 / 卡方检验 | 显著性检验 | scipy.stats |
| 假设检验 | Mann-Whitney U / Wilcoxon | 非参数检验 | scipy.stats |
| 回归诊断 | VIF（方差膨胀因子） | 多重共线性 | statsmodels（需确认） |
| 分布检验 | Shapiro-Wilk / K-S检验 | 正态性检验 | scipy.stats |
| 方差分析 | 单因素/多因素ANOVA | 组间差异 | scipy.stats |
| 灰色预测 | GM(1,1) | 小样本趋势预测 | numpy 手写 |
| 插值拟合 | 多项式拟合 / 样条插值 | 曲线拟合 | numpy / scipy.interpolate |
| 马尔可夫 | 马尔可夫链 | 状态转移预测 | numpy 手写 |

---

**启发式搭配建议**：

| 主模型 | 推荐启发式 | 优化的超参数 |
|--------|-----------|------------|
| XGBoost/LightGBM | PSO / DE | n_estimators, learning_rate, max_depth, subsample |
| CatBoost | SA | iterations, learning_rate, depth, l2_leaf_reg |
| 随机森林 | GA | n_estimators, max_depth, min_samples_split |
| SVR/SVC | PSO | C, gamma, epsilon |
| MLPRegressor | GA / PSO | hidden_layer_sizes, learning_rate_init, alpha |
| KNN | PSO | n_neighbors, weights, p |
| LSTM/GRU | GA | hidden_size, num_layers, dropout | [需PyTorch] |
| CNN-LSTM | PSO | filters, kernel_size, lstm_units | [需PyTorch] |

**模型选型决策树**（根据问题类型快速定位）：

```
问题类型？
├── 预测/回归
│   ├── 结构化表格数据 → XGBoost/LightGBM + PSO调参
│   ├── 时间序列 → LightGBM滑窗特征 或 ARIMA基线 + XGBoost
│   ├── 小样本（<500条）→ SVR + GA调参 或 BayesianRidge
│   └── 需要可解释性 → 随机森林 + SHAP
├── 分类
│   ├── 二分类 → XGBClassifier + PSO调参
│   ├── 多分类 → LGBMClassifier 或 StackingClassifier
│   ├── 不平衡数据 → CatBoost + SMOTE + 阈值调整
│   └── 小样本 → SVC(RBF) + GA调参
├── 聚类/分群
│   ├── 已知簇数 → K-Means++ 或 GMM
│   ├── 未知簇数 → DBSCAN 或 层次聚类（看树状图截断）
│   └── 高维数据 → PCA降维 + K-Means
├── 优化/规划
│   ├── 连续变量优化 → PSO 或 DE（scipy）
│   ├── 离散/组合优化 → GA
│   ├── 线性约束 → scipy.linprog
│   ├── 非线性约束 → scipy.minimize(SLSQP)
│   ├── 多目标 → NSGA-II 或 加权求和
│   └── 路径/网络 → Dijkstra / 最小生成树
├── 评价/排序
│   ├── 多指标综合排序 → 熵权法 + TOPSIS
│   ├── 主观+客观 → AHP + 熵权法组合赋权
│   ├── 效率评价 → DEA
│   └── 模糊不确定 → 模糊综合评价
└── 统计分析
    ├── 相关性 → Pearson/Spearman + 热力图
    ├── 差异检验 → t检验/ANOVA
    ├── 小样本趋势 → 灰色预测GM(1,1)
    └── 曲线拟合 → 多项式/样条插值
```

**输出**：`paper_output/step1/algo_recommendation.json`

---

### 阶段二：数据处理与建模计算（Step 3-4）

#### Step 3: 数据清洗与可视化（data-cleaning-and-visualization）

**做什么**：自动清洗赛题附件数据，生成EDA探索性分析图表。

**功能**：
- 自动发现`problem_files/`中的数据文件
- 智能清洗：缺失值处理、异常值检测、格式转换
- 自动可视化：直方图、箱线图、相关性热力图、散点矩阵

**输出**：
- `paper_output/data_cleaned/` — 清洗后数据
- `paper_output/figures/` — EDA图表

---

#### Step 3.5: 数据质量前置检查（data-quality-precheck）

**做什么**：在建模代码生成之前，对清洗后的数据执行系统性质量评估，生成诊断报告和处理建议，为Step 4的代码生成和Step 4.5的结果审查提供数据层面的决策依据。

**输入**：`paper_output/data_cleaned/` 中的清洗后数据

**6项检查指标**：

1. **缺失值比例**：每列缺失率 > 5% 时标记，推荐填充策略（均值/中位数/众数/插值）
2. **异常值检测**：基于IQR方法（$Q_1 - 1.5 \times IQR$ 到 $Q_3 + 1.5 \times IQR$ 之外），异常比例 > 5% 时标记，推荐截断或Winsorize处理
3. **类别平衡度**（分类问题）：目标变量少数类比例 < 20% 时标记，推荐SMOTE过采样或随机欠采样
4. **特征相关性**：计算VIF（方差膨胀因子），VIF > 10 的特征对标记，推荐剔除或PCA降维
5. **特征分布偏度**：$|\text{skew}| > 2$ 的特征标记，推荐对数变换或Box-Cox变换
6. **特征尺度差异**：最大特征标准差 / 最小特征标准差 > 100 时标记，推荐标准化（StandardScaler / MinMaxScaler / RobustScaler）

**输出**：`paper_output/data_quality_report.json`，包含每项检查的结果、标记的特征列表和推荐处理策略。该报告自动传递给Step 4的代码生成器和Step 4.5的结果审查Agent。

---

#### Step 3.6: 数据预处理章节生成（data-preprocessing-chapter-generator）

**做什么**：基于Step 3.5的数据质量报告，执行完整的数据预处理流程，生成方法候选对比、可视化图表，并输出论文第X章"数据预处理"的完整文本。该章节插入在论文"符号说明"之后、"模型建立与求解"之前。

**输入**：
- `paper_output/data_quality_report.json` — Step 3.5的质量检查结果
- 原始数据文件（`problem_files/` 中的附件）

**五大处理模块（按顺序执行）**：

**模块1：数据概览与样本分析**
- 统计总样本量、特征数、目标变量分布（分类：各类别比例；回归：分布形态）
- 生成样本分布可视化：目标变量饼图/直方图 + 每位研究对象的重复记录次数分布
- 输出表格：数据维度、变量类型统计（连续/离散/类别）、缺失率概览

生成图表：`figures/preprocessing/0_data_overview.png`

---

**模块2：缺失值处理（方法候选对比）**

> ⛔ **必须在论文中列出候选方法并说明选择理由，不得直接跳到结论**

**候选方法清单**：

| 方法 | 适用条件 | 优点 | 缺点 |
|------|---------|------|------|
| 均值填充 | 缺失率 < 5%，连续变量，无偏斜 | 简单快速 | 缩小方差，可能引入偏差 |
| 中位数填充 | 偏斜分布，连续变量 | 对异常值鲁棒 | 不保留分布形态 |
| 众数填充 | 类别变量 | 语义合理 | 可能导致某类别过多 |
| KNN插值 | 缺失率 < 20%，特征间有相关性 | 保留局部结构 | 计算开销大 |
| 多重插补（MICE） | 缺失率较高，MAR假设成立 | 统计理论严谨 | 实现复杂 |
| 删除行 | 缺失率 < 2% 且随机缺失 | 不引入偏差 | 损失样本量 |
| 删除列 | 缺失率 > 50% | 去除无用特征 | 可能损失信息 |

**生成图表**：`figures/preprocessing/1_missing_values.png`（各列缺失率柱状图 + 关键特征完整率图）

**决策逻辑**：
- 读取 `data_quality_report.json` 中的缺失率
- 缺失率 = 0：记录"无缺失值，跳过此步骤"
- 缺失率 < 5%：推荐均值/中位数填充，在论文中说明
- 缺失率 5%~20%：推荐KNN插值，对比均值填充的效果
- 缺失率 > 20%：讨论是否删除该列

---

**模块3：异常值检测与处理（方法候选对比）**

> ⛔ **必须使用至少2种方法检测，对比结果后说明最终选择**

**候选检测方法**：

| 方法 | 判定标准 | 适用场景 |
|------|---------|---------|
| IQR方法 | $x < Q_1 - 1.5 \times IQR$ 或 $x > Q_3 + 1.5 \times IQR$ | 通用，对偏斜分布稳健 |
| Z-score方法 | $\|z\| > 3$（$z = (x-\mu)/\sigma$） | 正态分布数据 |
| 修正Z-score | $0.6745 \times \|x - \text{median}\| / \text{MAD} > 3.5$ | 偏斜或含极端值的数据 |
| 孤立森林 | 异常得分 > 阈值 | 高维数据，多变量联合异常 |
| DBSCAN噪点 | 不属于任何簇的点 | 非线性分布的异常点 |

**候选处理策略**：

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| 保留 | 异常值是真实有效数据点 | 极端但合理的值（如极高BMI） |
| 截断（Winsorize） | 替换为边界值（$Q_1-1.5IQR$ 或 $Q_3+1.5IQR$） | 少量异常，不希望删除样本 |
| 删除 | 直接删除该样本 | 明显录入错误（如身高=1cm） |
| 对数变换 | 压缩极端值范围 | 右偏分布中的极端高值 |
| 单独建模 | 异常点群体单独处理 | 异常点有特殊研究价值 |

**生成图表**：`figures/preprocessing/2_outlier_detection.png`（关键特征箱线图，标注IQR边界和异常点数量）

---

**模块4：特征分布分析与变换（方法候选对比）**

**分布检验**：
- 对所有连续特征计算偏度（skewness）和峰度（kurtosis）
- 执行Shapiro-Wilk正态性检验（$n < 5000$）或K-S检验（$n \geq 5000$）
- 偏度 $|\text{skew}| > 2$ 的特征标记为"强偏斜"

**候选变换方法**：

| 变换 | 公式 | 适用场景 |
|------|------|---------|
| 对数变换 | $x' = \log(x + 1)$ | 右偏分布，$x > 0$ |
| 平方根变换 | $x' = \sqrt{x}$ | 轻度右偏 |
| Box-Cox变换 | $x' = (x^\lambda - 1)/\lambda$ | 自动寻找最优 $\lambda$ |
| Yeo-Johnson变换 | 扩展Box-Cox至负值 | 含负值的偏斜数据 |
| 保持原始 | 无变换 | 树模型（对分布不敏感） |

**生成图表**：`figures/preprocessing/3_distribution_analysis.png`（直方图+KDE+正态拟合，展示偏度）

---

**模块5：特征标准化方法对比**

**候选标准化方法**：

| 方法 | 公式 | 适用场景 | 不适用场景 |
|------|------|---------|----------|
| Z-score标准化 | $x' = (x-\mu)/\sigma$ | 线性模型、SVM、KNN、神经网络 | 含极端异常值时 |
| Min-Max归一化 | $x' = (x-x_{min})/(x_{max}-x_{min})$ | 需要 $[0,1]$ 范围的模型 | 含极端异常值时 |
| Robust标准化 | $x' = (x-\text{median})/\text{IQR}$ | 含异常值的数据 | 对范围敏感的模型 |
| 不标准化 | $x' = x$ | 树模型（XGBoost/LightGBM/随机森林） | 线性模型、距离模型 |

> ⛔ **模型-标准化搭配强制约束**：
> - 线性回归/逻辑回归/SVR/SVM/KNN/神经网络：**必须**标准化
> - XGBoost/LightGBM/CatBoost/随机森林/决策树：**不需要**标准化（可选）
> - 在论文中**必须**说明所用标准化方法及选择原因

**生成图表**：`figures/preprocessing/4_normalization_comparison.png`（四种标准化方法的箱线图对比，直观展示尺度变化）

---

**模块6（可选）：特征工程与派生特征**

仅当 Step 3.5 的数据质量报告中存在以下情况时执行：
- 发现强非线性关系（需要多项式/交互特征）
- 特征间有已知的物理/业务关联关系

**候选特征工程方法**：

| 方法 | 示例 | 适用场景 |
|------|------|---------|
| 交互特征 | $BMI \times 孕周$ | 两变量联合作用 |
| 多项式特征 | $BMI^2$, $孕周^2$ | 非线性关系 |
| 对数特征 | $\ln(BMI)$ | 指数增长关系 |
| 比值特征 | 体重/身高² = BMI（如未提供） | 衍生物理量 |
| 分箱特征 | BMI分组：偏瘦/正常/超重/肥胖 | 非线性+可解释性 |

**生成图表**：`figures/preprocessing/5_feature_engineering.png`（派生特征与原始特征的相关性对比，用彩色条形图区分）

---

**论文章节输出格式**

数据预处理章节写入论文时，结构如下：

```
## X、数据预处理

### X.1 数据概览
[样本量、特征数、目标变量分布描述]
[图0：数据概览图，含3句以上解读]

### X.2 缺失值分析与处理
[候选方法对比表]
[决策说明：本研究选择___方法，理由是___]
[图1：缺失值分析图，含3句以上解读]

### X.3 异常值检测与处理
[检测结果：各特征异常值数量]
[候选处理策略对比]
[决策说明：本研究选择___策略，理由是___]
[图2：异常值箱线图，含5句以上解读（复杂多子图）]

### X.4 特征分布分析
[偏度/正态性检验结果]
[候选变换方法对比]
[决策说明：___特征采用___变换，___特征保持原始]
[图3：分布分析图，含5句以上解读]

### X.5 特征标准化
[候选方法对比表]
[决策说明：回归/分类模型采用___标准化，树模型不标准化]
[图4：标准化方法对比图，含3句以上解读]

### X.6 特征工程（如有）
[派生特征说明及与目标变量的相关性]
[图5：特征工程效果图，含3句以上解读]

### X.7 预处理小结
[预处理后的数据特征总结]
[本章处理流程的逻辑说明，引出下一章建模]
```

**脚本输出**：
- 生成 `code/solve_preprocessing.py` — 完整的预处理脚本
- 生成 `figures/preprocessing/` 目录下的6张图表
- 在 `paper_output/step2_calc_results.json` 中追加 `"preprocessing"` 字段，记录各步骤的统计数据（缺失率、异常值比例、最终选择的方法）

**输出**：
- `code/solve_preprocessing.py` — 数据预处理分析脚本
- `figures/preprocessing/*.png` — 6张预处理可视化图表（0_data_overview ~ 5_feature_engineering）
- `paper_output/preprocessing_chapter.md` — 预处理章节草稿（供Step 6合并时使用）

**与论文的集成方式**：Step 6生成论文时，在"符号说明"章节之后、"模型建立与求解"章节之前，自动读取 `paper_output/preprocessing_chapter.md` 并插入。

---

**做什么**：根据算法推荐方案生成完整可运行的Python建模代码，执行后输出真实数值结果。

**脚本结构**（所有代码存放在`code/`目录）：
```
code/
├── algo_utils.py        ← 算法工具库（回归/PSO/GA/KMeans/决策树/随机森林等）
├── solve_problem1.py    ← 问题一完整求解脚本
├── solve_problem2.py    ← 问题二完整求解脚本
├── solve_problem3.py    ← 问题三完整求解脚本
├── solve_problem4.py    ← 问题四完整求解脚本
└── run_all_models.py    ← 一键运行所有问题
```

**每个脚本的标准结构**：
1. 导入（numpy/pandas/matplotlib）
2. 路径配置
3. 数据加载
4. 数据预处理
5. 基线模型
6. 主模型（含启发式优化）
7. 评估指标（RMSE/MAE/MAPE/R²）
8. 图表生成
9. 结果输出
10. 主函数

**必须生成的图表**：
- 预测结果对比图（对应预测结果数值表）
- 模型性能对比柱状图（对应模型性能对比表）
- 启发式算法收敛曲线（对应优化迭代过程）
- 误差分布图（对应误差统计表）

**图表中的公式标注**：图表标题、坐标轴标签中的数学符号也应使用LaTeX格式（matplotlib原生支持，如 `ax.set_xlabel(r"$\alpha$ value")`、`ax.set_title(r"RMSE vs $\lambda$")`）

**图表语言与字体验证（强制，必须在生成图表前执行）**：

> **⛔ 字体可用性检测（必须执行，不可跳过）：**
> 在写入任何matplotlib图表代码之前，必须先运行以下检测：
> ```python
> import matplotlib.font_manager as fm
> try:
>     path = fm.findfont('SimHei')
>     if 'DejaVu' in path or 'dejavu' in path.lower():
>         raise ValueError('SimHei not actually available')
>     FONT_AVAILABLE = True
> except:
>     FONT_AVAILABLE = False
>     print('WARNING: SimHei not available, falling back to English labels')
> ```
> - 若 `FONT_AVAILABLE = False`：**强制使用英文标注**（标题、坐标轴、图例全部英文），同时将 `contest_config.json` 中的 `figure_language` 改为 `"en"`
> - 若 `FONT_AVAILABLE = True`：可使用中文标注
> - **禁止硬编码 `plt.rcParams['font.sans-serif'] = ['SimHei']` 而不检测可用性**——这会导致Linux/Docker环境下图表中文乱码

**图表语言配置**：
- 读取`contest_config.json`中的`figure_language`字段（`"zh"`或`"en"`）
- matplotlib字体配置：**禁止硬编码SimHei等中文字体**（Linux/Docker环境下不可用会乱码）。默认使用`rcParams['font.sans-serif'] = ['DejaVu Sans']`
- 中文模式（`"zh"`）下：先检测`SimHei`或`WenQuanYi Micro Hei`是否可用（`matplotlib.font_manager.findfont`），可用则使用，不可用则自动fallback到`"en"`模式并在日志中输出警告
- 英文模式（`"en"`）下：所有标题、坐标轴标签、图例、标注均使用英文
- 无论语言模式，图表中的数学符号（变量名、单位）均使用LaTeX语法（如 `r"$\alpha$"`, `r"$R^2$"`）

**代码质量要求**：
- 可复现：固定随机种子（`np.random.seed(42)`）
- 有注释：每个函数有docstring
- 有进度提示：使用 tqdm 进度条（见下方进度条规范）
- 早停机制：所有支持早停的模型必须启用（见下方早停规范）
- 图表自动保存：不弹窗，直接保存到`paper_output/figures/`

**⛔ 早停强制规范**：

> 所有建模脚本必须为支持早停的模型配置早停参数，防止过拟合并大幅缩短训练时间。
>
> **各模型早停配置速查**：
>
> ```python
> # ── XGBoost ──────────────────────────────────────────────
> from xgboost.callback import EarlyStopping
> model = xgb.XGBRegressor(
>     n_estimators=1000,          # 设大，让早停决定实际轮数
>     early_stopping_rounds=50,   # 连续50轮无提升则停止
>     eval_metric='rmse',         # 监控指标
> )
> model.fit(
>     X_train, y_train,
>     eval_set=[(X_val, y_val)],
>     verbose=False
> )
> print(f"XGBoost 实际使用轮数: {model.best_iteration}")
>
> # ── LightGBM ─────────────────────────────────────────────
> import lightgbm as lgb
> callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
> model = lgb.LGBMRegressor(n_estimators=1000)
> model.fit(
>     X_train, y_train,
>     eval_set=[(X_val, y_val)],
>     callbacks=callbacks
> )
> print(f"LightGBM 实际使用轮数: {model.best_iteration_}")
>
> # ── CatBoost ─────────────────────────────────────────────
> from catboost import CatBoostRegressor, EarlyStoppingCallback
> model = CatBoostRegressor(
>     iterations=1000,
>     early_stopping_rounds=50,
>     eval_metric='RMSE',
>     verbose=False
> )
> model.fit(X_train, y_train, eval_set=(X_val, y_val))
> print(f"CatBoost 实际使用轮数: {model.best_iteration_}")
>
> # ── PSO 早停 ─────────────────────────────────────────────
> best_score = np.inf
> no_improve_count = 0
> PATIENCE = 20        # 连续20代无提升则停止
> MIN_DELTA = 1e-6     # 最小提升阈值
>
> for gen in trange(n_generations, desc='PSO优化'):
>     # ... PSO 更新逻辑 ...
>     current_best = gbest_score
>     if best_score - current_best > MIN_DELTA:
>         best_score = current_best
>         no_improve_count = 0
>     else:
>         no_improve_count += 1
>     if no_improve_count >= PATIENCE:
>         print(f"PSO 早停：第{gen}代，连续{PATIENCE}代无提升")
>         break
>
> # ── GA 早停 ──────────────────────────────────────────────
> best_fitness_history = []
> PATIENCE = 15
>
> for gen in trange(n_generations, desc='GA优化'):
>     # ... GA 更新逻辑 ...
>     best_fitness_history.append(best_fitness)
>     if len(best_fitness_history) > PATIENCE:
>         recent = best_fitness_history[-PATIENCE:]
>         if max(recent) - min(recent) < 1e-6:
>             print(f"GA 早停：第{gen}代，种群收敛")
>             break
> ```
>
> **早停不等于减少迭代上限**：`n_estimators` 或 `n_generations` 仍设置为较大值，由早停机制决定实际停止时机。禁止直接将 `n_estimators=50` 来"模拟"早停效果。

**⛔ 进度条强制规范**：

> 所有建模脚本必须使用 `tqdm` 进度条显示运行进度，让用户清楚知道当前进行到哪一步、还剩多少时间。
>
> **tqdm 可用性检测（写在每个脚本开头）**：
> ```python
> try:
>     from tqdm import tqdm
>     from tqdm.auto import trange
>     USE_TQDM = True
> except ImportError:
>     # tqdm 不可用时的降级方案：用简单打印替代
>     def tqdm(iterable, desc='', total=None, **kwargs):
>         if desc: print(f"[进行中] {desc}...")
>         return iterable
>     def trange(n, desc='', **kwargs):
>         if desc: print(f"[进行中] {desc} (共{n}步)...")
>         return range(n)
>     USE_TQDM = False
> ```
>
> **必须加进度条的场景**：
> | 场景 | 写法 |
> |------|------|
> | PSO/GA 迭代 | `for i in trange(n_iter, desc='PSO优化', unit='代')` |
> | 交叉验证 | `for fold, (tr, va) in tqdm(enumerate(cv.split(...)), desc='交叉验证', total=n_splits)` |
> | 多问题顺序执行 | `for step in tqdm(steps, desc='建模进度')` |
> | 大数据集批处理 | `for batch in tqdm(batches, desc='数据处理', unit='batch')` |
> | 模型训练（树模型） | XGBoost/LightGBM 设置 `verbose_eval=10` 或 `callbacks=[tqdm_callback]` |
>
> **run_all_models.py 进度条格式**：
> ```
> 建模进度: 100%|████████████| 4/4 [05:23<00:00, 80.8s/题]
>   ✅ 问题一完成 (耗时: 78.3s) | 结果: R²=0.92
>   ✅ 问题二完成 (耗时: 45.1s) | 结果: AUC=0.87
>   ⏳ 问题三运行中...
> ```

**⛔ 依赖包安装失败处理规范（禁止因 VM 环境受限降级方案）**：

> **核心原则**：Claude Code 的代码执行环境（VM/沙箱）与用户本地环境不同。VM 中缺少的包 ≠ 用户环境中缺少的包。**代码必须按用户本地环境的可用包来编写，而非按 VM 环境来编写。**
>
> **禁止**以下行为：
> - ❌ 因为 VM 中只有 numpy/pandas/matplotlib 就只用这些包写代码
> - ❌ 放弃 XGBoost/LightGBM/CatBoost 等专业库，改用 numpy 手写模型
> - ❌ 放弃 scikit-learn，用 pandas 手写交叉验证
> - ❌ 在代码中写 `# 由于环境限制，使用简化版本` 之类的注释
> - ❌ 未经用户确认就自行决定降级方案
>
> **正确的处理方式**：
> 1. **明确区分"写代码"和"跑代码"**：
>    - 写代码：按用户本地环境的完整包列表来写（见下方清单）
>    - 跑代码：如果 VM 无法运行，让用户在本地 VS Code 终端中运行
> 2. **用户本地环境已确认可用的包（直接使用，无需检测）**：
>    - numpy, pandas, matplotlib, seaborn, openpyxl（基础包）
>    - scikit-learn（机器学习基础：交叉验证、预处理、评估指标）
>    - xgboost, lightgbm, catboost（梯度提升模型）
>    - scipy（科学计算/优化/统计检验）
>    - python-docx（Word文档生成）
>    - **运行环境**：Windows 全局 Python 环境（`D:\APP\Python 3.13\python.exe`），无需激活虚拟环境，直接 `python xxx.py` 即可
> 3. **代码写完后如果 VM 无法运行**：
>    - 不要修改代码去适配 VM
>    - 直接告诉用户在本地运行：
>      ```
>      ⏱️ 代码已生成完毕，但当前执行环境缺少必要的包。
>      请在 VS Code 终端中执行：
>      
>      cd <项目根目录>
>      python code/run_all_models.py
>      
>      运行完成后告诉我"跑完了"，我将读取结果继续后续步骤。
>      ```
> 4. **如果用户反馈某个包确实不存在**（import 报错），再提供安装命令：
>    ```
>    pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple
>    ```
>
> **总结**：VM 环境受限不是降低代码质量的理由。代码质量由用户本地环境决定，执行方式可以灵活调整。

**⛔ 长文件写入规范（防止截断）**：

> AI 的单次输出有长度限制，超长文件会被截断。为避免"截断→修复→再截断"的死循环，**必须遵守以下规则**：
>
> 1. **代码文件**（`.py`）：单个文件超过 150 行时，必须分段写入（先写前半部分，确认成功后再追加后半部分）
> 2. **论文文件**（`final_paper.md`）：按章节分段写入（先写摘要到问题分析，再写模型建立，再写结果分析...），每段写入后确认文件完整性
> 3. **如果检测到文件被截断**：
>    - **禁止**从头重写整个文件（会再次截断）
>    - **正确做法**：定位截断位置，只追加缺失的部分
> 4. **JSON 结果文件**：如果内容过长，按子问题分别写入（如先写 problem1 的结果，再追加 problem2）
>
> **判断标准**：如果同一个文件连续出现 2 次"截断→修复"循环，必须停下来改用分段写入策略，不得继续循环。

**⛔ 超时处理规范（禁止因超时降低模型质量）**：

> 当代码运行超时时，**禁止**通过以下方式"解决"超时问题：
> - ❌ 减少 PSO/GA 迭代次数（如从200代降到10代）
> - ❌ 减少交叉验证折数（如从5折降到2折）
> - ❌ 大幅缩减训练数据量
> - ❌ 去掉启发式优化步骤
> - ❌ 用简单模型替代原定的复杂模型
>
> **正确的超时处理方式（按优先级）**：
> 1. **增加超时时间**：设置更长的执行超时（如 600秒、1200秒或无限制），优先尝试此方案
> 2. **分段执行**：将代码拆分为多个独立脚本分别运行（如 `step4a_preprocess.py` → `step4b_train.py` → `step4c_evaluate.py`），每段控制在超时范围内
> 3. **后台执行**：使用后台进程运行长时间任务，定期检查结果文件是否生成
> 4. **交给用户手动运行**：当以上方式均不可行时，执行以下流程：
>    - 确保代码已保存到 `code/` 目录且可独立运行
>    - 确保代码运行结束后会自动将结果写入 `paper_output/step2_calc_results.json`，图表写入 `figures/`
>    - 向用户输出明确指令：
>      ```
>      ⏱️ 代码运行时间较长，请在 VS Code 终端或 CMD 中手动执行：
>      
>      cd <项目根目录>
>      python code/solve_problem1.py
>      python code/solve_problem2.py
>      （或一键运行：python code/run_all_models.py）
>      
>      运行完成后回来告诉我"跑完了"，我将读取结果文件继续后续步骤。
>      预计运行时间：约 X 分钟
>      ```
>    - **等待用户回复"跑完了"后**，读取 `paper_output/step2_calc_results.json` 和 `figures/` 目录，验证结果文件存在且非空，然后继续执行 Step 4.5
>
> **允许的合理优化（不损害模型质量）**：
> - ✅ 使用 `n_jobs=-1` 并行加速（所有 sklearn 模型默认加上）
> - ✅ XGBoost/LightGBM/CatBoost 默认启用 GPU 加速（见下方 GPU 检测规范）
> - ✅ 使用 `early_stopping` 提前终止（但不减少最大迭代上限）
> - ✅ 对超大数据集使用分层抽样（保留至少 80% 数据）
> - ✅ 减少网格搜索范围但保留关键超参数的合理搜索空间

**⛔ GPU 加速强制规范**：

> 生成建模代码时，**必须**在每个脚本开头加入以下 GPU 检测代码，并根据检测结果自动配置：
>
> ```python
> import subprocess, sys
>
> def check_gpu():
>     """检测 CUDA GPU 是否可用"""
>     try:
>         result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
>         if result.returncode == 0:
>             try:
>                 import xgboost as xgb
>                 # 测试 GPU 是否真的可用
>                 test = xgb.XGBRegressor(device='cuda', n_estimators=1)
>                 import numpy as np
>                 test.fit(np.array([[1,2],[3,4]]), np.array([1,2]))
>                 print("✅ GPU 可用，启用 CUDA 加速")
>                 return True
>             except Exception:
>                 pass
>     except Exception:
>         pass
>     print("⚠️ GPU 不可用，使用 CPU（可用 n_jobs=-1 并行加速）")
>     return False
>
> USE_GPU = check_gpu()
>
> # XGBoost GPU 配置
> XGB_DEVICE = 'cuda' if USE_GPU else 'cpu'
> # XGBoost < 2.0 兼容写法
> XGB_TREE_METHOD = 'gpu_hist' if USE_GPU else 'hist'
>
> # 使用示例：
> # xgb.XGBRegressor(device=XGB_DEVICE, tree_method=XGB_TREE_METHOD, ...)
> # lgb.LGBMRegressor(device='gpu' if USE_GPU else 'cpu', ...)
> # CatBoostRegressor(task_type='GPU' if USE_GPU else 'CPU', ...)
> ```
>
> **各库 GPU 参数速查**：
> | 库 | GPU 参数 | CPU 参数 |
> |----|---------|---------|
> | XGBoost ≥2.0 | `device='cuda'` | `device='cpu'` |
> | XGBoost <2.0 | `tree_method='gpu_hist', gpu_id=0` | `tree_method='hist'` |
> | LightGBM | `device='gpu', gpu_platform_id=0, gpu_device_id=0` | `device='cpu'` |
> | CatBoost | `task_type='GPU', devices='0'` | `task_type='CPU'` |
>
> **GPU 不影响模型质量，只影响速度**，因此所有情况下都应默认尝试启用。

**输出**：
- `paper_output/step2_calc_results.json` — 真实数值结果
- `figures/problem*/*.png` — 真实图表（存放在根目录`figures/`下，按问题分子目录）

---

#### Step 4.5: 结果审查Agent与三级递进修复（result-review-agent）

> **⛔ 硬性门禁（BLOCKING GATE）：Step 4.5 是进入Step 5的绝对前置条件，禁止跳过或简化。**
>
> **AI必须在对话中逐项执行以下操作：**
> 1. 读取 `step2_calc_results.json`，列出每个子问题的核心指标值
> 2. 对照下方达标阈值表，逐项标注"✅达标"或"❌未达标"
> 3. 对未达标的问题，**必须**执行至少1轮修复（调参→特征工程→换模型），并在对话中展示修复前后的指标对比
> 4. 所有修复日志写入 `paper_output/review_log.json`
> 5. 只有所有子问题均达标或已走完三级修复流程后，才允许进入Step 5
>
> **如果AI跳过Step 4.5直接生成论文，或仅说"结果尚可"就进入Step 5，视为严重违规，必须回退重做。**

**做什么**：在Step 4建模代码执行完成后、Step 5论文生成之前，对每个子问题的建模结果执行质量评估。未达标的子问题自动进入三级递进修复流程，直到结果达标或达到最大迭代次数。

**输入**：`paper_output/step2_calc_results.json` + `paper_output/data_quality_report.json`

**质量评估指标（根据问题类型自动选择）**：

| 问题类型 | 核心指标 | 达标阈值 | 辅助指标 |
|---------|---------|---------|---------|
| 回归类 | $R^2$ | $\geq 0.15$ | RMSE, MAE |
| 分类类 | AUC | $\geq 0.75$ | 异常类F1 $\geq 0.3$，异常类召回率 $\geq 0.2$ |
| 优化类 | 收敛性 | 末代/倒数第10代差异 < 5% | 风险降低幅度 > 20% vs 基准 |

> 注：以上阈值为默认值，用户可在`contest_config.json`中自定义覆盖。

**三级递进修复流程**：

```
┌─────────────────────────────────────────────────────────┐
│                  Step 4 执行完毕                          │
│                         │                                │
│                    ┌────▼────┐                           │
│                    │ 质量评估 │                           │
│                    └────┬────┘                           │
│                    ┌────▼────┐                           │
│              ┌─YES─┤  达标？ ├─NO─┐                      │
│              │     └─────────┘    │                      │
│              ▼                    ▼                      │
│         进入Step 5          ┌──────────┐                 │
│                          ┌──┤ 第1轮:调参 ├──┐             │
│                          │  └──────────┘  │              │
│                          │    ┌────▼────┐ │              │
│                          │    │  达标？  │ │              │
│                          │    └──┬───┬──┘ │              │
│                          │  YES │   │ NO  │              │
│                          │      ▼   ▼     │              │
│                          │  Step5  ┌──────────┐          │
│                          │     ┌───┤第2轮:特征工程├──┐     │
│                          │     │   └──────────┘  │      │
│                          │     │   ┌────▼────┐   │      │
│                          │     │   │  达标？  │   │      │
│                          │     │   └──┬───┬──┘   │      │
│                          │     │ YES │   │ NO    │      │
│                          │     │     ▼   ▼       │      │
│                          │     │ Step5 ┌──────────┐     │
│                          │     │    ┌──┤第3轮:换模型├──┐  │
│                          │     │    │  └──────────┘  │  │
│                          │     │    │  ┌────▼────┐   │  │
│                          │     │    │  │  达标？  │   │  │
│                          │     │    │  └──┬───┬──┘   │  │
│                          │     │    │YES │   │ NO    │  │
│                          │     │    │    ▼   ▼       │  │
│                          │     │    │ Step5 强制进入   │  │
│                          │     │    │    Step5       │  │
│                          ▼     ▼    ▼                ▼  ▼
│                       ┌─────────────────────────────┐   │
│                       │     Step 5: 论文撰写         │   │
│                       └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**第1轮：调参重跑**
- 调整超参数（如随机森林树数量从30增到100、max_depth从8增到15、学习率网格搜索）
- 更换随机种子重跑（`random_seed` ∈ {42, 123, 456, 789, 2024}）
- 调整训练/测试比例（从8:2改为7:3或9:1）
- 执行后重新评估指标，达标则进入Step 5，未达标进入第2轮

**第2轮：特征工程与数据处理**
- 读取`data_quality_report.json`中的推荐策略并执行：
  - 类别不平衡（少数类 < 15%）→ SMOTE过采样或多数类随机欠采样
  - 特征偏度大（$|\text{skew}| > 2$）→ 对数变换或Box-Cox变换
  - 多重共线性（VIF > 10）→ 剔除高相关特征或PCA降维
  - 特征尺度差异大 → 更换标准化方式（StandardScaler → MinMaxScaler → RobustScaler）
- 增加交互特征（如 $Z_{21} \times BMI$）和多项式特征
- 执行后重新评估，达标则进入Step 5，未达标进入第3轮

**第3轮：换模型**
- 通过网络检索2022年后的学术文献，找到针对当前数据特点的更优模型方案
- 优先级：同类问题的经典改进模型（如随机森林 → XGBoost/LightGBM）→ 近年新颖模型（如TabNet、FT-Transformer）→ 集成多个模型的Stacking方案
- 重写代码并执行
- **无论结果如何，本轮结束后强制进入Step 5**，但在论文中如实报告模型的局限性

**退出条件**：
- 最多执行3轮，防止死循环
- 每轮之间检查指标提升幅度：若当前轮次相比上一轮指标提升 < 1%，跳过后续轮次直接进入Step 5
- 每轮修复的日志记录到`paper_output/review_log.json`，包含：轮次、采取的措施、修复前后的指标对比、是否达标

**输出**：
- `paper_output/step2_calc_results.json` — 更新后的计算结果（如果经过修复）
- `paper_output/review_log.json` — 修复过程日志
- `paper_output/figures/problem*/*.png` — 更新后的图表（如果经过修复）

---

### 阶段三：论文生成（Step 5-7）

#### Step 5: 质量审计与任务清单生成（quality-assurance-auditor）

**做什么**：在论文生成前做门禁检查，生成微单元任务清单。

**脚本层检查**：
- `problem_files/` 目录非空
- `step3_filled_placeholder.py` 填写状态
- 生成52个微单元任务清单（`tasks.json`），**每个微单元包含`min_words`字段**：
  - 问题理解与建模思路：200字
  - 数学模型推导：300字
  - 算法实现细节：200字
  - 结果呈现与数据解读：300字
  - 图表分析：200字
  - 物理/工程意义阐释：200字
  - 小结与过渡：100字
  - 每道子问题章节合计：≥1500字

**AI层审计**（在对话中执行）：
- 赛题覆盖度：是否100%识别所有子问题
- 模型-任务匹配度：模型类型与任务目标是否一致
- 逻辑一致性：假设⇄模型推导⇄结果解释是否闭环
- 评分点对齐度：每个高分点是否被定位承诺
- 图表-正文链检测：正文引用的图表是否真实存在

**输出**：`paper_output/tasks.json`

---

#### Step 6: 论文微单元生成与合并（paper-micro-unit-generator）

**做什么**：按微单元模板批量生成论文各章节文本，自动合并成完整论文。

**微单元体系**（2000+微单元，覆盖论文全部章节）：
- 摘要（48句，每句15-20字）
- 问题重述（96句）
- 问题分析（192句）
- 模型假设（48句）
- 符号说明（128句）
- 模型建立与求解（576句，每问192句）
- 结果分析与可视化（192句）
- 模型检验（144句）
- 结论与建议（96句）
- 不足与展望（48句）
- 参考文献、附录

**生成流程**：
1. 从`step3_filled_placeholder.py`读取内容
2. 从`step2_calc_results.json`读取数值
3. 按微单元编号顺序生成`paper_output/micro_units/*.txt`
4. 合并为`paper_output/final_paper.md`
5. 自动生成目录、章节编号、图表编号、交叉引用
6. 同时生成`paper_output/final_paper.docx`（不依赖Pandoc）

> **⛔ docx图片嵌入强制约束：**
> 生成docx时，必须解析Markdown中的 `![alt](path)` 语法，读取对应图片文件并用 `python-docx` 的 `add_picture()` 方法嵌入到Word文档中。
> **禁止将 `![...](...)` 当作纯文本输出到docx**——这是常见错误，会导致论文中没有图片。
> 验证方法：生成后用 `doc.part.rels` 检查image类型的rel数量，必须等于论文中引用的图片总数。

> **⛔ 论文内容结构强制约束：**
> 论文的章节顺序必须严格遵循以下结构，**禁止在附录之后添加任何正文内容**：
> 摘要 → 问题重述 → 问题分析 → 模型假设 → 符号说明 → 模型建立与求解（各子问题） → 模型检验 → 结果分析与讨论 → 结论与建议 → 参考文献 → 附录
>
> 如果字数不足需要扩充内容，**必须将新内容插入到对应章节内部**（如扩充问题一的分析就插入到"5.1"小节内部），**绝不能追加到附录之后**。追加到附录之后的内容会被评委视为格式错误，直接扣分。

> **⛔ 图文布局强制约束（一图一说明）：**
> 论文中每张图/表插入后，**紧接下方必须是该图/表的专属解读段落**，实现"一图一说明"的连贯布局。
>
> **具体规则**：
> 1. **普通图表**：插入后紧跟**至少3句**专属解读（描述趋势、引用数值、给出结论）
> 2. **复杂图表**（含多子图、热力图、雷达图、多维对比图）：插入后紧跟**至少5句**专属解读
> 3. **允许2张关联图连续放置**：当且仅当两张图展示同一现象的不同视角（如"预测散点图 + 残差分布图"、"训练曲线 + 验证曲线"），可连续放置后用一段统一解读（但该段仍需≥5句，且必须分别提及两张图的具体内容）
> 4. **禁止3张及以上图片连续堆放**：无论是否关联，连续3张图中间无文字直接判定为布局不合格
> 5. **禁止"图区+文字区"分离布局**：不得将某一节的所有图集中放在开头或末尾，再用一大段文字统一解释
>
> **解读段落的最低质量要求**：
> - 必须描述图中观察到的**具体趋势或模式**（如"随迭代次数增加，RMSE从0.82单调下降至0.31"）
> - 必须引用图中的**具体数值**（如"在第50代达到收敛，最终适应度值为0.156"）
> - 必须给出该图支撑的**结论或发现**（如"表明PSO优化有效提升了模型精度"）
> - 复杂图还需描述**异常点、拐点或分区特征**
>
> **验证方法**：生成论文后扫描所有`![...](...)` 图片标记，检查其后第一个文字段落是否满足上述句数和内容要求。不满足的标记为`[FIGURE_LAYOUT_ERROR]`。

**LaTeX公式规范**：
- 论文中所有数学公式必须以LaTeX代码形式编写
- 行内公式：`$E = mc^2$`、`$\alpha + \beta$`
- 独立公式块：`$$\min_{\theta} \frac{1}{n}\sum_{i=1}^{n}L(f(x_i;\theta), y_i)$$`
- 变量/向量/矩阵：`$x_i$`、`$\mathbf{X}$`、`$\hat{y}$`、`$\nabla J$`
- Markdown文件中LaTeX公式可被MathJax等渲染器正常显示
- Word文件中LaTeX公式以代码形式写入，用户可在Word中使用公式编辑器（插入→公式）逐个替换渲染

**字数门禁验证（Step 6完成后自动执行）**：
- 用代码逐模块统计`final_paper.md`中每个章节的实际字数
- 对照`tasks.json`中的`min_words`字段逐项核验
- 不达标的模块标记为`[NEEDS_EXPANSION]`并触发该模块的重新生成（最多重试2次）
- 全文总字数低于12000字时，自动返回补充内容
- 验证结果写入`paper_output/word_count_report.json`

**输出**：
- `paper_output/final_paper.md` — Markdown论文草稿（公式为LaTeX代码）
- `paper_output/final_paper.docx` — Word论文（公式为LaTeX代码）
- `paper_output/ref_check.md` — 交叉引用检查报告
- `paper_output/word_count_report.json` — 字数验证报告

---

#### Step 6.5: LaTeX自动校验与修复（latex-linter）

**做什么**：在Step 6论文合并完成后、进入Step 8之前，对`final_paper.md`执行LaTeX格式校验，发现并自动修复问题。

**三项检查与修复**：

**检查1：Unicode数学符号扫描与替换**
- 用正则扫描全文中的Unicode数学字符，自动替换为LaTeX代码
- 替换映射表：

| Unicode字符 | LaTeX替换 | Unicode字符 | LaTeX替换 |
|------------|----------|------------|----------|
| `∑` | `$\sum$` | `α` | `$\alpha$` |
| `β` | `$\beta$` | `γ` | `$\gamma$` |
| `δ` | `$\delta$` | `σ` | `$\sigma$` |
| `μ` | `$\mu$` | `ε` | `$\epsilon$` |
| `θ` | `$\theta$` | `λ` | `$\lambda$` |
| `√` | `$\sqrt{}$` | `≥` | `$\geq$` |
| `≤` | `$\leq$` | `∞` | `$\infty$` |
| `±` | `$\pm$` | `≈` | `$\approx$` |
| `≠` | `$\neq$` | `×` | `$\times$` |
| `÷` | `$\div$` | `→` | `$\rightarrow$` |

**检查2：公式括号匹配**
- 扫描所有`$...$`和`$$...$$`块
- 验证花括号`{}`和圆括号`()`的开闭数量是否一致
- 不一致的公式标记为`[LINT_ERROR]`并报告行号

**检查3：变量格式检查**
- 扫描正文中未被`$...$`包裹的独立数学变量名（如连续出现的`W`、`B`、`Y`等）
- 提示是否需要包裹为`$W$`、`$B$`、`$Y$`
- 排除常见英文单词（如"a", "the", "is", "and"等）

**输出**：
- 覆盖写入修复后的`paper_output/final_paper.md`
- `paper_output/lint_report.json` — 校验报告（替换数量、错误列表、警告列表）

---

#### Step 7: 全流程编排（paper-workflow-orchestrator）

**做什么**：串联Step 0-6，一键从赛题文件夹到完整论文。

**一键运行**：
```bash
python .trae/skills/paper-workflow-orchestrator/scripts/run_all.py
```

**完整性交付标准**：
- 必须存在：`paper_output/final_paper.docx`
- 必须存在：`paper_output/final_paper.md`
- 必须存在：`paper_output/tasks.json`
- `final_paper.md`中`[待填写]`不超过5处
- 每个子问题章节有具体数值结果

---

### 阶段四：论文精修（Step 8-12）

#### Step 8: 图表精修（mathmodel-figure-polish）

**做什么**：生成符合竞赛评审标准的出版级图表。

**核心规则**：
- 图表是视觉论证，不是装饰。每张图必须服务于一个明确的科学结论
- **论文中每张数据表格都必须有对应的可视化图表**（强制）
- 柱状图上方必须标注具体数值
- 折线图关键点必须标注数值

**配色方案**：
- 主色：`["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]`
- 基线用灰色`#6C757D`，改进用彩色

**图表类型速查（强制执行）**：

> **⚠️ 硬性约束：论文中禁止出现超过2张同类型图表。必须按下方速查表选择图表类型，不得用柱状图替代其他类型。违反此约束的论文直接判定为图表不合格。**

| 场景 | 强制使用的图表类型 | 禁止使用 |
|------|---------|---------|
| 模型精度对比（多指标） | 雷达图（多维）或分组柱状图（仅限1处） | 连续多张柱状图 |
| 预测vs真实 | 散点图+对角线+置信带 | 柱状图 |
| 敏感性分析 | 热力图/龙卷风图 | 柱状图 |
| 误差分布 | 直方图+密度曲线 | 柱状图 |
| 改善百分比对比 | 折线图（带标注）或瀑布图 | 柱状图 |
| 多维度场景对比 | 热力图（矩阵） | 多张柱状图 |
| 渐进式改善 | 双轴折线图/瀑布图 | 柱状图 |
| 特征重要性 | 水平条形图 | 垂直柱状图 |
| 时序趋势 | 面积图/折线图+置信带 | 柱状图 |
| 分布对比 | 箱线图/小提琴图 | 柱状图 |
| 相关性分析 | 散点矩阵/热力图 | 柱状图 |
| 收敛曲线 | 折线图 | 柱状图 |
| 模型框架 | 流程图（matplotlib patches） | 柱状图 |

**图表多样性强制检查清单**（生成论文前必须逐项验证）：
- [ ] 柱状图（含分组柱状图）总数 ≤ 2
- [ ] 至少使用4种不同的图表类型
- [ ] 每个子问题的图表类型不完全相同
- [ ] 雷达图/热力图/散点图至少各出现1次
- [ ] 无连续2张同类型图表

**图表类型生成优先级**（当场景可匹配多种类型时，优先选择非柱状图）：
1. 散点图（预测vs真实、相关性）
2. 热力图（多维对比、相关矩阵、场景分析）
3. 雷达图（多指标综合对比）
4. 折线图（时序、趋势、收敛）
5. 箱线图/小提琴图（分布对比）
6. 水平条形图（特征重要性、排名）
7. 分组柱状图（仅在其他类型不适用时使用，且最多1处）

**评委友好度检查**：3秒规则、自解释性、标注语言（读取`contest_config.json`的`figure_language`字段，`"zh"`用中文标注，`"en"`用英文标注；中文模式下若SimHei不可用则自动fallback到英文）、单位明确、配色一致、无冗余

**输出**：`figures/` 目录下的出版级图表（PNG 300dpi + SVG + PDF），按问题分子目录（`figures/problem1/`、`figures/problem2/`等）

---

#### Step 9: 论文学术润色（mathmodel-writing-polish）

**做什么**：借鉴Nature级写作方法论，提升论文的逻辑性、可读性与评分友好度。

**润色工作流程（12步）**：

**诊断阶段**：
1. 识别章节类型
2. 判断主要问题（论证链断裂/逻辑不一致/表达空洞/结构混乱/声明过度）
3. 确定优先级：`论证链 > 逻辑一致 > 结构清晰 > 表达精确 > 措辞优化`

**结构修复**：
4. 段落职责检查（每段只承担一个论证职责）
5. 证据-声明对齐（每个声明有数据/图表/推导支撑）
6. 漏斗结构检查（问题分析：宏观→微观；结论：微观→宏观）
7. 图文布局检查（每张图后是否紧跟专属解读段落，普通图≥3句，复杂图≥5句；是否存在连续3张以上图片堆放）

**表达优化**：
7. 动词校准（数学证明用"证明"，数值实验用"表明"，推测用"可能"）
8. 句长控制（每句≤50字，关键结论句≤30字）
9. 学术化转写（"我们发现"→"实验结果表明"，"效果很好"→"RMSE降低了X%"）
10. 过度声明检测（禁止"首次提出""完美解决""最优"等无证据支撑的强声明）

**验收**：
11. 评分友好度检查
12. 输出润色结果

**输出格式**：润色后文本 + 修改说明（3-5条）+ 声明-证据审计表 + 评分友好度评分（0-100）

---

#### Step 10: 文献检索与引用管理（mathmodel-citation-search）

**做什么**：为论文检索、验证并格式化参考文献（GB/T 7714-2015格式）。

**分层检索策略**：
1. **权威来源**：算法原始论文、教材专著、国家标准、官方统计数据
2. **高质量期刊**：中文核心期刊、SCI/EI检索论文、顶级会议
3. **补充来源**：硕博论文、预印本（arXiv）、政府报告

**禁止引用**：百度百科/维基百科、CSDN/知乎、无法验证的网页

**引用完整性检查**：
- 每个`[数字]`引用在参考文献列表中有对应条目
- 编号连续，无跳号
- 中英文文献混排格式统一

**数学建模常用文献速查**：
| 方法 | 推荐引用 |
|------|---------|
| XGBoost | Chen T, Guestrin C. KDD, 2016 |
| LSTM | Hochreiter S, Schmidhuber J. Neural Computation, 1997 |
| 随机森林 | Breiman L. Machine Learning, 2001 |
| TOPSIS | Hwang C L, Yoon K. 1981 |
| 遗传算法 | Holland J H. 1975 |

**输出**：格式化的参考文献列表

---

#### Step 11: 评委质疑预演（mathmodel-defense-rehearsal）

**做什么**：模拟评委视角对论文进行质疑预演，识别薄弱环节并生成应对策略。

**质疑维度（按扣分风险排序）**：

**致命问题（可能直接降档）**：答非所问、模型偷换、结果造假、逻辑断裂、约束违反

**严重问题（扣5-15分）**：无验证、假设不合理、过度声明、缺少对比、敏感性缺失

**一般问题（扣1-5分）**：图表不清、符号混乱、引用缺失、格式问题、表达含糊

**输出**：
- 整体可辩护性评分（0-100）
- 质疑清单（按严重程度排序）
- 每个质疑的修改建议和答辩话术
- 修复优先级清单

---

#### Step 12: 最终质量审计（quality-assurance-auditor）

**做什么**：在提交前做最终一致性检查。

**检查项**：
- 每一问是否有明确的量化结论
- 关键创新点是否在摘要和结论中被突出
- 图表引用是否完整且编号正确
- 符号使用是否前后一致
- 机械重复检测（连续段落完全重复→FAIL）
- `[待填写]`是否已全部替换
- **图文布局连贯性**：扫描所有图片标记，验证每张图后是否紧跟专属解读段落（普通图≥3句，复杂图≥5句）；是否存在连续3张以上图片堆放；存在`[FIGURE_LAYOUT_ERROR]`标记时审计结果为FAIL
- **字数门禁**：读取`word_count_report.json`，逐模块核验是否达到`min_words`。不达标的模块标记为`[NEEDS_EXPANSION]`并触发该模块的重新生成（最多重试2次）。全文总字数 < 12000字时审计结果为FAIL
- **LaTeX残留检查**：扫描全文是否存在Unicode数学符号残留（`lint_report.json`中`[LINT_ERROR]`数量 > 0 时FAIL）

**输出**：PASS/FAIL报告 + 修改清单 + 字数验证报告 + LaTeX校验报告

---

### 辅助技能

#### 上下文记忆管理（context-memory-keeper）

维护双层记忆结构：
- **长期准则**：用户偏好、全局约束（只读为主）
- **短期工作台**：当前任务状态、外部文献索引（读写频繁）

每个子技能完成后都应调用此技能更新进度。

#### Python可视化图库（python-visualization-gallery）

提供箱线图、小提琴图、热力图、散点图、时序图等离线可运行的matplotlib/seaborn代码模板。

#### 权威数据采集（authoritative-data-harvester）

从权威来源获取补充数据（国家统计局、世界银行等）。

#### 竞赛评分标准与模型选型（modeling-paper-rubric-and-model-selector）

根据竞赛评分标准自动推荐模型组合。

---

## 正确的使用流程

```
Step 0: 放赛题文件到 problem_files/
        放撰写规范到 problem_files/spec/
        放优秀论文到 problem_files/exemplars/

Step 1: 运行建模计算脚本 → 输出 paper_output/step2_calc_results.json

Step 1.5: 结果审查Agent自动评估 → 未达标则自动执行三级递进修复（调参→特征工程→换模型）

Step 2: 复制模板并填写内容：
        copy step3_filled_placeholder_template.py step3_filled_placeholder.py
        （编辑 step3_filled_placeholder.py，填写所有 [待填写] 项）

Step 3: 运行全流程 → 生成论文草稿
        python .trae/skills/paper-workflow-orchestrator/scripts/run_all.py

Step 4: 精修（可选但推荐）
        - 图表精修：说"重新画问题二的对比图"
        - 文字润色：说"帮我润色第三章"
        - 文献补充：说"帮我找参考文献"
        - 质疑预演：说"帮我检查论文有没有漏洞"
        - 最终审计：说"帮我审计这篇论文"
```

---

## 精修流程（论文草稿完成后）

```
1. mathmodel-figure-polish    → 重新生成出版级配图
2. mathmodel-writing-polish   → 润色论文文字，修复论证逻辑
3. mathmodel-citation-search  → 补充和格式化参考文献
4. mathmodel-defense-rehearsal → 模拟评委质疑，找出薄弱点
5. quality-assurance-auditor  → 最终一致性检查
```

---

## 常见问题

**Q: 生成的论文全是`[待填写]`？**
A: 说明`step3_filled_placeholder.py`未填写或不存在。复制模板并填写后重新运行。

**Q: 数值结果没有自动填入？**
A: 需要建模脚本输出`paper_output/step2_calc_results.json`，格式：
```json
{
  "problem1": {"model_name": {"RMSE": 0.49, "MAE": 0.39, "R2": 0.92, "label": "..."}},
  "problem2": {...}
}
```

**Q: Word文档里公式显示为`$E=mc^2$`（LaTeX代码）？**
A: 这是**预期行为**。论文中的所有公式均以LaTeX代码形式写入，确保公式结构完整且可移植。Markdown版本可通过MathJax自动渲染；Word版本中，用户可在Word里使用"插入→公式"功能，将LaTeX代码粘贴进公式编辑器即可自动渲染为专业排版。

**Q: 图表没有嵌入论文？**
A: 在`step3_filled_placeholder.py`对应章节中用Markdown图片语法引用：`![图题](paper_output/figures/problem1/xxx.png)`

---

## 评分标准参考（以RMSE/MAE/MAPE/R²各25分为例）

| 指标 | 含义 | 优化方向 |
|------|------|---------|
| RMSE | 均方根误差 | 越小越好 |
| MAE | 平均绝对误差 | 越小越好 |
| MAPE | 平均绝对百分比误差 | 越小越好 |
| R² | 决定系数 | 越大越好（最大1） |

---

## 约束（必须遵守）

1. **不编造数值**：所有写入`step2_calc_results.json`的数值必须来自真实运行结果
2. **不编造文献**：所有引用必须来自真实搜索结果
3. **启发式算法必须真实实现**：不允许用随机数模拟"收敛曲线"
4. **每张数据表格必须有对应图表**（强制）
5. **论文正文中文优先**：除公式、专有名词外，使用规范中文学术表达。**图表标注语言**由`contest_config.json`的`figure_language`字段决定，中文模式下若字体不可用则自动fallback到英文
6. **Memory Interaction**：每个关键步骤完成后更新上下文记忆
7. **最终交付必须包含`.docx`**，不能只交付Markdown
8. **不发明内容**：润色不创造数据、不编造引用、不虚构实验结果
9. **图表类型多样性强制约束**：论文中柱状图（含分组柱状图）不得超过2张；至少使用4种不同图表类型；每个子问题的图表类型不得完全相同；必须按Step 8的图表类型速查表选择图表类型，禁止用柱状图替代其他类型
10. **LaTeX公式强制约束**：论文中所有数学公式、变量字母、数学符号**必须使用LaTeX代码显示**，禁止使用纯文本或Unicode字符代替。具体要求：行内公式用`$...$`包裹，独立公式块用`$$...$$`包裹；变量名、矩阵、向量等均需LaTeX表示（如 `$\mathbf{x}$`、`$\alpha$`、`$\sum_{i=1}^{n}$`）；Word文档(.docx)中的公式同样以LaTeX代码形式写入，后续可由用户手动在Word公式编辑器中渲染
11. **结果审查强制约束（BLOCKING GATE）**：Step 4执行完毕后**必须**经过Step 4.5结果审查Agent评估。AI必须在对话中逐项展示每个子问题的指标评估结果，未达标的问题**必须**执行至少1轮修复尝试。修复日志必须记录到`review_log.json`。**禁止跳过Step 4.5直接进入Step 5**。
12. **docx图片嵌入强制约束**：生成的`.docx`文件必须包含所有论文中引用的图片。生成代码必须解析`![alt](path)`语法并用`add_picture()`嵌入。生成后必须验证图片数量。**禁止将Markdown图片语法当作纯文本输出到docx**。
13. **论文结构强制约束**：论文内容必须严格按章节顺序排列。**禁止在附录之后添加任何正文内容**。字数不足时必须将扩充内容插入到对应章节内部，不得追加到末尾。
14. **图表字体强制约束**：生成图表前**必须**检测中文字体（SimHei/WenQuanYi）的可用性。若不可用，**必须**使用英文标注，**禁止**使用不可用的中文字体导致乱码。
15. **字数门禁强制约束**：Step 6生成论文后必须通过字数门禁验证，每个子问题章节 ≥ 1500字，全文 ≥ 12000字。不达标时自动触发重新生成
16. **文件分目录强制约束**：生成的代码（`solve_problem*.py`、`algo_utils.py`、`run_all_models.py`）存放在`code/`目录；生成的图表（`.png`/`.svg`/`.pdf`）存放在`figures/`目录（按问题分子目录）；论文及相关报告（`.md`、`.docx`、`.json`）存放在`paper_output/`目录。三个目录互不混放
17. **图文布局强制约束（一图一说明）**：论文中每张图/表插入后，紧接下方必须是该图/表的专属解读段落。普通图表至少3句解读，复杂图表（多子图/热力图/雷达图/多维对比）至少5句解读。允许2张有关联的图连续放置（如同一现象的不同视角），但连续放置后的统一解读段落仍需≥5句且分别提及两张图。**禁止3张及以上图片连续堆放**。禁止"图区+文字区"分离布局（即所有图集中放在一处、文字集中放在另一处）

---

## 内容充实度强制规则（论文文字质量保障）

> **核心问题**：AI倾向于生成"结构完整但文字单薄"的论文。以下规则用于强制提升文字深度。
>
> **LaTeX公式规范**：论文中所有数学内容必须使用LaTeX代码表示。行内公式用 `$...$`（如 `$\alpha$`、`$x_i$`），独立公式用 `$$...$$`（如 `$$\sum_{i=1}^{n} a_i = S$$`）。禁止使用纯文本Unicode数学符号（如 ∑、α、β、√）或直接写无格式变量名。

### 总字数最低要求

| 章节 | 最低字数 | 说明 |
|------|---------|------|
| 摘要 | 400字 | 必须覆盖：背景+方法+结果+结论，每问至少2句结果 |
| 问题重述 | 300字 | 不是复制赛题，要用自己的话重新组织 |
| 问题分析 | 500字 | 每道子问题至少100字的分析 |
| 模型假设 | 200字 | 每条假设必须有"合理性论证" |
| 每道子问题求解 | 1500字 | 见下方"子问题章节内容模板" |
| 模型综合对比 | 800字 | 不能只列表格，必须有逐项分析 |
| 结论 | 400字 | 总结+创新点+不足+展望 |
| **正文总计** | **≥12000字** | 不含附录和参考文献 |

### 子问题章节内容模板（每道题必须包含以下全部7个模块）

**模块1：问题理解与建模思路（200-300字）**
- 明确本题要解决什么问题
- 解释为什么选择这种方法（而不是其他方法）
- 描述方法的核心思想和预期效果

**模块2：数学模型推导（300-500字）**
- 必须写出完整的数学公式（不是一句话带过）
- **所有公式必须使用LaTeX代码编写**：行内公式用 `$...$`，独立公式用 `$$...$$`，例如：`$$L(\theta) = \sum_{i=1}^{n}(y_i - f(x_i; \theta))^2$$`
- 变量、参数、矩阵、向量等一律用LaTeX表示（如 `$\alpha$`、`$\beta$`、`$\mathbf{W}$`、`$\hat{y}$`）
- 公式前后必须有文字解释每个符号的含义
- 必须解释公式的物理/几何意义

**模块3：算法实现细节（200-300字）**
- 描述具体的算法步骤（不是"采用XX算法"一句话）
- 说明超参数的选择依据
- 描述训练/优化过程

**模块4：结果呈现与数据解读（300-400字）**
- 引用具体的数值结果（不能只说"效果很好"）
- 对每个关键指标进行解读
- 将结果与基线或其他方法进行对比

**模块5：图表分析（200-300字）**
- 对每张图表进行详细解读（不能只说"如图所示"）
- 描述图表中观察到的具体趋势、模式、异常点

**模块6：物理/工程意义阐释（200-300字）**
- 解释结果在实际应用中的含义
- 讨论模型的优势和局限性

**模块7：小结与过渡（100-150字）**
- 总结本题的核心发现
- 指出与下一题的逻辑关联

### 写作质量红线（以下写法直接判定为不合格）

| 红线写法 | 正确做法 |
|---------|---------|
| "采用XX方法进行分析" | 必须写出XX方法的具体公式和步骤 |
| "结果表明模型效果良好" | 必须给出具体数值并解读其含义 |
| "如图X所示" | 必须描述图中观察到的具体现象 |
| 一段话只有2-3句 | 每段至少5句，核心段落8-10句 |
| 连续多段无公式 | 每个模型章节至少3个独立公式块 |
| 全篇无比对分析 | 每道题至少与1个基线方法对比 |
| 公式用纯文本或Unicode写（如 Σ、α、β） | 必须使用LaTeX代码（如 `$\sum$`、`$\alpha$`、`$\beta$`） |
| 变量名直接写英文无格式（如 x, y, W） | 变量必须用LaTeX包裹（如 `$x$`、`$y$`、`$\mathbf{W}$`） |
| 多张图集中堆放后用一大段文字统一解释 | 每张图后紧跟专属解读段（普通图≥3句，复杂图≥5句） |
| 图表解读只说"如图所示趋势明显" | 必须引用图中具体数值、描述具体趋势方向和拐点 |

### 论文生成前的自检清单

在生成最终论文前，AI必须对以下问题逐一回答"是"：

- [ ] 每道子问题的章节是否超过1500字？
- [ ] 每个模型是否写了完整的数学公式推导？
- [ ] 所有数学公式是否都使用LaTeX代码编写（`$...$`行内 / `$$...$$`独立块）？
- [ ] 所有变量、参数、矩阵是否都用LaTeX格式表示（而非纯文本）？
- [ ] 每张图表是否都有2-3句的具体解读？
- [ ] 数值结果是否都有物理意义的解释？
- [ ] 模型对比是否有逐指标的分析？
- [ ] 全文是否超过12000字？
- [ ] 图文布局是否满足"一图一说明"（每张图后紧跟专属解读，普通图≥3句，复杂图≥5句）？
- [ ] 是否不存在连续3张以上图片堆放的情况？
- [ ] 是否不存在"图区+文字区"分离布局？

如果有任何一项为"否"，必须返回补充内容后再生成最终文档。

---

## 适用时机

- 用户已放好赛题与附件，希望自动生成论文
- 需要为赛题选择有文献支撑的算法方案
- 需要生成完整可运行的建模代码
- 需要润色论文、优化图表、补充文献
- 需要模拟评委质疑、准备答辩
- 触发词："帮我写建模论文"、"生成论文"、"跑一下模型"、"帮我润色"、"帮我找文献"、"帮我检查论文"

---

## 子技能索引

| 编号 | 子技能 | 一句话描述 |
|------|--------|-----------|
| 0 | contest-spec-analyzer | 从规范和优秀论文提取格式要求 |
| 1 | problem-doc-model-selector | 解析赛题，生成题意对齐与模型路线 |
| 2 | mathmodel-algo-researcher | 检索2022年后文献，推荐算法方案 |
| 3 | data-cleaning-and-visualization | 自动清洗数据，生成EDA图表 |
| 3.5 | data-quality-precheck | 数据质量前置检查（6项指标+处理建议） |
| 3.6 | data-preprocessing-chapter-generator | 数据预处理章节生成（方法候选对比+6张可视化+论文章节） |
| 4 | mathmodel-code-generator | 生成建模代码，输出真实数值 |
| 4.5 | result-review-agent | 结果审查Agent（三级递进修复：调参→特征工程→换模型） |
| 5 | quality-assurance-auditor | 门禁检查（含字数门禁），生成任务清单 |
| 6 | paper-micro-unit-generator | 微单元生成与合并成论文（含字数门禁验证） |
| 6.5 | latex-linter | LaTeX自动校验与修复（Unicode替换+括号匹配+变量格式） |
| 7 | paper-workflow-orchestrator | 一键全流程编排 |
| 8 | mathmodel-figure-polish | 出版级图表制作（图表语言可配置：zh/en） |
| 9 | mathmodel-writing-polish | Nature级学术润色 |
| 10 | mathmodel-citation-search | 文献检索与GB/T 7714格式化 |
| 11 | mathmodel-defense-rehearsal | 评委质疑预演与答辩准备 |
| 12 | context-memory-keeper | 上下文记忆管理 |
| 13 | python-visualization-gallery | Python可视化代码模板库 |
| 14 | authoritative-data-harvester | 权威数据采集 |
| 15 | modeling-paper-rubric-and-model-selector | 评分标准与模型选型 |
