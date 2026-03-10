

# ⭐强化学习完整运行框架图

**环境给状态→智能体按策略选动作→环境返回奖励和新状态→智能体根据回报价值评估更新策略参数，用更新后的策略继续交互，周而复始，直到学到最优策略。**

![Gemini_Generated_Image_pm831dpm831dpm83](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_pm831dpm831dpm83.png)

## 1) 主循环：交互产生数据

**环境 → 智能体：给出状态*s_t***

- 环境在时刻*t* 把当前局部观测 (状态 *s_t* 提供给智能体
- 这一步对应图中上方从右到左的箭头“状态*s_t* (观测/局部”

**智能体 → 环境：输出动作*a_t***

- 智能体内部用策略 *π*(a|s) 根据当前状态*s_t* 选择/采样一个动作*a_t*
- 图里写了“动作随机性来自策略：A_π(⋅|s)”——意思是策略可以是随机策略，不一定每次都选同一个动作
- 这一步对应从左到右的箭头“动作*a_t* (控制指令)”

**环境 → 智能体：反馈奖励 *r_t* 和下一状态 *s_{t+1}***

- 环境接收到动作*a_t* 后，按其动力学规则发生状态转移：

  *s_t*+1p*(⋅∣*s_t*,*a_t*)

- 图中强调“状态随机性来自环境”

- 同时环境返回一个即时反馈：**奖励 *r_t***

- 这两项一起回到智能体：图中下方两条箭头“奖励*r_t*”与“下一状态 *s_{t+1}*”

到这里，一次时间步交互完成。图中中央的大字*交互—学习—再交互**”就是强调：交互产生数据，学习更新策略，然后用新策略继续交互

## 2) 智能体内部：评估与学习更新

交互得到的数据通常会被智能体整理成轨迹/经验
$$
(s_t, a_t, r_t, s_{t+1}, \dots)
$$
**(1) 价值评估$V_\pi(s) / Q_\pi(s, a)$**

- 图里的“价值评估”模块用于衡量“这个状态动作有多好”，本质上是在估计未来长期收益的期望
- 常见形式是状态价值$V_\pi(s)$ 或动作价值$Q_\pi(s, a)$

**(2) 回报 $U_t$ 与折扣因$\gamma$**

- 图右下角给了折扣回报

  $$U_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$$

- $\gamma$ 越接1，越看重“更长远”的未来奖励

**(3) 策略更新/学习 (Policy Update)**

- 智能体利用“奖励+ 轨迹/回报 + 价值评估”来更新策略参数（图中写“更新参数：使长期回报更大”，并提示可能用 TD 误差/梯度更新/参数 $w$）
- 更新后的策略 $\pi$ 会回流到“策略模块”，使下一轮在同样状态下更倾向于选择能带来更高长期回报的动作

# 0.强化学习

## 1.基础知识

## 2.强化学习核心概念

### 2.1 State Action
* 状态𝑠：当前“局部观测帧”（PPT用马里奥画面做直观说明）
* 动作 a∈{left,right,up}：智能体可选的控制指令<br>
📌 记忆：状态是“我在哪/看到了什么”，动作是“我要做什么”<br>

### 2.2 Policy（策略）𝜋

* 定义：策略给出在状态𝑠下选择动作𝑎的概率：π(a∣s)=P(A=a∣S=s)
* 策略既可以随机，也可以确定性<br>
💡 直觉：𝜋就是智能体的“行为规则/控制律”<br>
### 2.3 Reward（奖励）𝑅
⚠️ 强化学习的学习信号来自reward，但 reward 本身可能非常稀疏延迟<br>
### 2.4 State Transition（状态转移）𝑝
* 核心观点：动作会把系统从旧状态推到新状态；但这个转移可能是随机的，随机性来自环境
* 转移概率定义p(s′∣s,a)=P(S_{t+1}=s′∣S_{t}=s,A_{t}=a)$

## 3.两个随机性来源
### 3.1 动作随机性（来自策略
给定状态𝑠，动作是随机变量：A∼π(⋅∣s)<br>
### 3.2 状态随机性（来自环境转移
给定状态下一状态是随机变量
$$S_{t+1}∼p(⋅∣s,a)$$
📌 一句话总结
* 策略𝜋决定“我怎么随机”；
* 环境𝑝决定“世界怎么随机

## 4.轨迹、Episode与回报（Return
### 4.1 交互过程与轨迹（trajectory
PPT 描述一轮游戏：观察 $s_t$，采样并执行 $a_t$，环境给$s_{t+1}$ $r_t$。并把序列写trajectory: $(s_0, a_0, r_0, s_1, a_1, r_1, \dots)$，一局从开始到结束称为一episode
### 4.2 Return与折扣回报
* Return（累计未来奖励）$U_t = R_t + R_{t+1} + R_{t+2} + \dots$$
* 为什么要折扣PPT 用“现在给你钱 vs 一年后给你钱”说明：未来奖励通常被认为更不重要
* Discount factor$\gamma$ 是可调超参数
* Discounted return$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots$$
* 📌 直觉$\gamma$ 越接1 越“长远”，越小越“短视”
### 4.3 回报的随机性：$U_t$ vs $u_t$
* 在时刻$t$ 看未来： 后续奖励未知，因$U_t$ 是随机变量
* episode 结束你会观测到具体奖励序列，从而得到一个具体数字$u_t$<br>
💡 PPT强调“为什么随机”： 奖励 $R_n$ 依赖状态与动作；而状态动作若随机，则奖励也随机
## 5.价值函数(Value Functions)
本章导读价值函数的核心就是：用期望把随机回报“平均化”，从而衡量“在某状态做某动作有多好”
### 5.1 动作价值函数$Q$
定义（给定策$\pi$）：$$Q_{\pi}(s_t, a_t) = \mathbb{E}[U_t \mid S_t = s_t, A_t = a_t]$$
📌 条件期望在“平均掉什么随机性”？PPT 特别说明：把 $s_t, a_t$ 当作已知观测，把未来自$S_{t+1:h}, A_{t+1:h}$ 视为随机变量（由 $p$ $\pi$ 共同决定）

### 5.2 状态价值函数$V$
给出“对动作再取期望”的定义
* 离散动作$V_{\pi}(s_t) = \mathbb{E}_{A \sim \pi(\cdot|s_t)} [Q_{\pi}(s_t, A)] = \sum_{a} \pi(a \mid s_t) Q_{\pi}(s_t, a)$$
* 连续动作把求和换成积分
### 5.3 直觉理解
* $Q_{\pi}(s, a)$：在状态$s$ 下“选动作$a$”有多好
* $V_{\pi}(s)$：在状态$s$ 这个“局面”本身有多好（固定策$\pi$）
* $\mathbb{E}[V_{\pi}(S)]$：用整体平均意义衡量策略 $\pi$ 好不好

# ⭐DQN思想示意

**Q 网络看状态输出各动作 Q ($\arg \max$ / $\varepsilon$-探索) 选动作→环境$r, s'$ 存经采样经验TD target 计算 TD error 最小化损失更新网络 周期更新目标网络 回到交互*

![Gemini_Generated_Image_a5zy3da5zy3da5zy](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_a5zy3da5zy3da5zy.png)

## 上半部分：与环境交互

**环境给状态$s_t$**

环境把当前观局面（状态）传给智能体

**状态输Q 网络，输出各动作Q *

智能体内部的 **Q 网络（动作价值网络）** 接收 $s_t$，输出一组数字

$$Q(s_t, a_1; w), Q(s_t, a_2; w), \dots$$

含义是“在状态$s_t$ 下，做每个动作的好坏评分”

**动作选择（通常$\arg\max$ + 探索*

- **贪心选择**a_t = \arg\max_a Q(s_t, a; w)$
- 图里也标**$\varepsilon$-贪心探索**：以小概$\varepsilon$ 随机选动作，避免只走当前最优导致学不到更好策略

**执行动作 $a_t$ 到环*

智能体把动作发送给环境

**环境返回奖励 $r_t$ 和下一状态$s_{t+1}$**

环境执行动作后反馈：即时奖励 + 新状态

**经验样本入库（可选但 DQN 常用*

把一次交互记为经验：$(s_t, a_t, r_t, s_{t+1})$，存**经验回放*，供训练用

## 下半部分：TD 学习训练流程（让 $Q(s, a; w)$ 更接$Q^*$

**A 经验回放 Replay Buffer：随机采样小批量 (minibatch)**

从经验池随机取一$(s, a, r, s')$，目的：打乱相关性，让训练更稳定

**B 计算 TD 目标 (TD target)**

用“下一状态的最Q 值”构造监督信号：

$$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; w^-)$$

- **$\gamma$**：折扣因
- **$w^-$**：目标网络参数（用来稳定训练；图里单独画Target 网络

**C 计算 TD 误差 (TD error)**

$$\delta_t = y_t - Q(s_t, a_t; w)$$

**D 定义损失函数**

$$L = (\delta_t)^2$$

（也就是让当前网络输出更接近 TD 目标

**E 反向传播 + 梯度下降更新参数 $w$**

$$w \leftarrow w - \alpha \nabla_w L$$

更新后，Q 网络对“好动作”的评分会逐渐变高，对“差动作”的评分变低

**目标网络更新（周期性拷贝）**

每隔一段时间执行：$w^- \leftarrow w$

TD 目标的计算更平稳

# 1.基于价值的学习

## 课程大纲

* 折扣回报 (Discounted Return) 与随机性来源（动作/状态随机）
* 动作价值函数$Q_\pi(s, a)$ 与最优动作价值$Q^\star(s, a)$
* 若已$Q^\star$，如何选最优动作：$a^\star = \arg\max_a Q^\star(s, a)$
* DQN：用神经网络 $Q(s, a; \mathbf{w})$ 近似 $Q^\star$
* Temporal Difference (TD) 学习：从“监督学习式更新”到“自(bootstrap) 更新
* TD 如何用于 DQN：TD target、TD error、训练迭代算
## 1.折扣回报 (Discounted Return)
**本章导读**<br>
回报 $U_t$ 之所以是随机的，是因为后续的动作 $A_{t+1:}$ 与状态$S_{t+1:}$ 都可能随机；折扣因子 $\gamma$ 用于衰减远期奖励

定义与公*<br>
PPT 给出折扣回报公式$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + \dots$$并指出其依赖未来的动作与状态随机序列

## 2.动作价值函数$Q_\pi$ 与最优动作价值$Q^\star$

**本章导读**

价值型方法的核心对象是 $Q(s, a)$：它衡量“在状态$s$ 采取动作 $a$ 后，未来折扣回报的期望”。随后引$Q^\star$ 作为“所有策略里最好”的动作价值

### 2.1 策略的动作价值函数$Q_\pi(s, a)$
$$Q_\pi(s_t, a_t) = \mathbb{E}[U_t \mid S_t = s_t, A_t = a_t]$$
并强调期望是对未来动作与状态（$A_{t+1:}, S_{t+1:}$）取的，把当前观$S_t = s_t, A_t = a_t$ 保留

### 2.2 最优动作价值函数$Q^\star$
$$Q^\star(s_t, a_t) = \max_\pi Q_\pi(s_t, a_t)$$
无论采用什么策略s_t$a_t$的结果都不可能优Q^\star(s_t, a_t)$

## 3.若已$Q^{\star}$，如何行动？

**本章导读**<br>
价值型 RL 的“控制”部分非常直接：如果我们掌握 $Q^{\star}$，就贪心选最大值对应的动作即可<br>
PPT 结论$a^{\star} = \arg\max_{a} Q^{\star}(s, a)$$并解$Q^{\star}$ 可以视为“在状态$s$ 选择动作 $a$ 有多好”的指标<br>
⚠️ 关键挑战：我们通常不知$Q^{\star}$

## 4.DQN：用神经网络近似 $Q^{\star}$
**本章导读**<br>
用深度网络$Q(s, a; \mathbf{w})$ 去近$Q^{\star}(s, a)$，从而实现“看状态$\to$ 输出各动作分$\to$ 选最大动作”

### 4.1近似形式
$$Q(s, a; \mathbf{w}) \approx Q^{\star}(s, a)$$
其中 $\mathbf{w}$ 是神经网络参数
### 4.2 网络输入/输出
* 输入：状态$s$
* 输出：动作空间维度大小的分数（每个动作一$Q$ 值）

# ⭐策略学习示意图

![Gemini_Generated_Image_fid6lsfid6lsfid6](C:\Users\SONGQI\Downloads\Gemini_Generated_Image_fid6lsfid6lsfid6.png)

## 基于策略学习 (Policy-based Learning) 核心流程

该流程的核心在于：用**策略网络 (Actor)** 直接输出动作概率分布，按概率采样动作与环境交互；收集一局轨迹后，用回报价值估计来计算策略梯度，更新策略参数$\theta$。流程分为“上半交互”和“下半学习”两大部分

------

##  上半部分：Actor 与环境交互（生成轨迹数据

**环境 → 智能体：状态$s_t$**

- 环境给出当前观测/情况 (state/observation)

**智能体内部：策略网络 Actor 计算动作概率分布**

- 策略网络 $\pi(a|s;\theta)$ $s_t$ 输入后输出一个概率向量，例如
  - $\pi(\text{left}|s) = 0.2$
  - $\pi(\text{right}|s) = 0.1$
  - $\pi(\text{up}|s) = 0.7$
- 通常使用 **Softmax(logits)** 将网络输出转化为“非负且和为 1”的概率

**动作采样模块：按概率采样动作 $a_t$**

$$a_t \sim \pi(\cdot|s_t;\theta)$$

- **强调**：策略本身具有随机性（非价值法中的 $\arg\max$），这种随机性有利于环境探索

**智能体 → 环境：执行动作$a_t$**

- 将采样得到的动作发送给环境执行

**环境 → 智能体：奖励 $r_t$ 与下一状态$s_{t+1}$**

- 环境返回即时奖励和下一步状态。环境中状态转移也可能具有随机性：

  $$s_{t+1} \sim p(\cdot|s_t, a_t)$$

**轨迹/一局 Episode**

- 将一段完整的交互过程记录为轨迹：

  $$(s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)$$

- 该轨迹是后续学习和更新策略的“训练数据”

------

## 下半部分：用轨迹做策略梯度学习（更新 $\theta$

下半部分通过轨迹数据确定参数的“更新方向”

#### 1）优化目

- **目标函数 $J(\theta)$**：最大化期望回报/期望价值

  $$J(\theta) = \mathbb{E}[V(S;\theta)]$$

- **直觉**：优化参数$\theta$ 使策略表现更好，从而获得更大的长期收益

#### 2）关键分岔：回报/价值估$q_t$ 的来自

根据 $q_t$ 的计算方式，通常分为两条路线

- **路线 A：REINFORCE（蒙特卡洛回报）**

  - 跑完整个 episode 后，计算折扣回报

    $$u_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

  - $q_t = u_t$

  - **特点**：无偏估计但方差较大，且必须等待回合结束

- **路线 B：Actor-Critic（低方差估计*

  - 引入一*价值网络评论(Critic)**V(s;\varphi)$ $Q(s, a;\varphi)$

  - Critic 给出更稳定的估计

    $$q_t \approx Q^\pi(s_t, a_t) \text{ 或优势函} A_t$$

  - **特点**：方差更低，学习更稳定且通常更高效

#### 3）计算策略梯

策略梯度的采样估计形式为

$$g_t = \nabla_\theta \log \pi(a_t|s_t;\theta) \cdot q_t$$

- **$\nabla_\theta \log \pi(a_t|s_t;\theta)$**：指明了如何调整参数以增加该动作发生的概率
- **$q_t$**：衡量该动作带来的长期效果好坏
- **结论**：获得好结果的动作会被增强概率，坏结果的动作则被降低概率

#### 4）参数更新

使用梯度上升更新策略参数

$$\theta_{t+1} = \theta_t + \beta \cdot g_t$$

- 更新完成后，策略回到上半部分，利用新策略继续与环境交互，形成闭环

------

##  总结

**Actor 观察状态并输出动作概率 采样动作与环境交互获取轨利用回报/价值估$q_t$ 计算策略梯度 更新 $\theta$ 以增强“高回报动作”的概率 循环往复*

| **维度**     | **REINFORCE**            | **Actor-Critic**             |
| ------------ | ------------------------ | ---------------------------- |
| **估计方式** | 蒙特卡洛 (MC) 全回合回报| 时序差分 (TD) Critic 估计 |
| **方差**     |                       |                           |
| **更新时机** | 回合结束 (Offline)       | 单步或多步更新(Online)      |
| **稳定*   | 较差                     | 较好                         |

# 2.基于策略的学

## 课程大纲
* 策略函数近似：从表格到函数逼近（神经网络）
* 价值函数回顾：折扣回报Q^\pi(s, a)$V^\pi(s)$
* Policy-based 学习目标：最大化 $J(\theta) = \mathbb{E}[V(S; \theta)]$
* 策略梯度：对 $V(s; \theta)$ 求导并构造可采样的梯度估
* 落地算法：用采样近似梯度；REINFORCE Actor-Critic 的分
## 2.1策略函数近似(Policy Function Approximation)
本章导读<br>
策略方法不直接学 $Q$ 表，而是直接表示并学<mark>在状态$s$ 下选动作$a$ 的概率分</mark>。当状态动作空间很大时，需要用函数逼近（神经网络）来表示策略

### 1）策略函数是什么？

策略函数 $\pi(a|s)$ 是一个概率分布（PDF），输入状态$s$，输出各动作的概率，并据此随机采样动作
$$\pi(\text{left}|s) = 0.2, \pi(\text{right}|s) = 0.1, \pi(\text{up}|s) = 0.7$$
📌 要点：策略方法的“动作选择”天然是随机的：$A \sim \pi(\cdot|s)$<br>
### 2能不能直接学一张策略表

* 若状态动作很少：可以画表（矩阵）学每个表项
* ⚠️ 若状态动作很多甚至连续：表格不可行，需要函数逼近
![619670b9-1f32-486a-a0a2-d713512e3995.png](C:\Users\SONGQI\Desktop\笔记\attachment:619670b9-1f32-486a-a0a2-d713512e3995.png)
## 2.2策略网络 (Policy Network)

**本章导读**<br>
用神经网络$\pi(a|s; \theta)$ 近似策略 $\pi(a|s)$，参数为 $\theta$。讲义强调调softmax 让输出成为合法分布

### 1定义与结果

policy network $\pi(a|s; \theta)$ 近似 $\pi(a|s)$，$\theta$ 是网络可训练参数
![7af727d7-e473-4218-88fe-f53cbacb48bb.png](C:\Users\SONGQI\Desktop\笔记\attachment:7af727d7-e473-4218-88fe-f53cbacb48bb.png)
### 2为什么用 Softmax?

给出约束 $\sum_{a \in \mathcal{A}} \pi(a|s; \theta) = 1$，因此使用 softmax<br>
💡 理解：softmax 把任意实logits 映射为非负且和为 1 的概率向量，方便“采样动作”

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.output_layer = nn.Linear(128, action_dim)

    def forward(self, s):
        x = F.relu(self.fc(s))
        logits = self.output_layer(x)
        probs = F.softmax(logits, dim=-1) # 确保概率之和1
        return probs

## 模拟输入状态

state = torch.randn(1, 10) # 假设状态维度为 10
net = PolicyNetwork(10, 3) # 假设3 个可选动作
print("动作概率分布:", net(state))
```

## 2.3价值函数回报

### 1折扣回报 (Discounted return)
定义折扣回报 $U_t$ 为从时刻 $t$ 开始所有未来奖励的加权总和$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots$$
回报的具体数值依赖于后续的随机动作与随机状态转移。这里存在两类随机性来源：
* 动作随机性：由策略函数决定，$\mathbb{P}(A = a | S = s) = \pi(a|s)$
* 状态转移随机性：由环境动力学决定\mathbb{P}(S' = s' | S = s, A = a) = p(s'|s, a)$
### 2动作价值函数$Q^\pi(s, a)$
定义
$$Q^\pi(s_t, a_t) = \mathbb{E}[U_t \mid S_t = s_t, A_t = a_t]$$
即：**在给定当前状态$s_t$ 并采取特定动作$a_t$ 后，未来折扣回报的条件期望*

### 3状态价值函数$V^\pi(s)$
$$V^\pi(s_t) = \mathbb{E}_{A \sim \pi(\cdot|s_t)}[Q^\pi(s_t, A)] = \sum_{a \in \mathcal{A}} \pi(a|s_t) Q^\pi(s_t, a)$$
## 2.4Policy-based 学习目标：最大化 $J(\theta)$
**本章导读**<br>
讲义把“策略网络”与“价值”连接起来：$\pi(a|s; \theta)$ 表示策略，并通过<mark>梯度上升</mark>更新 $\theta$，使期望价值变大<br>
近似状态价值（用策略网络加$Q$）：
$$
V(s; \theta) = \sum_{a} \pi(a|s; \theta) Q^\pi(s, a)
$$
并定义目标：学习 $\theta$ 最大化
$$
J(\theta) = \mathbb{E}[V(S; \theta)]
$$
更新思想：用<mark>策略梯度</mark>上升
$$
\theta \leftarrow \theta + \beta \nabla_\theta J(\theta)
$$


## 2.5策略梯度推导 (Policy Gradient)
### 1从定义出发求
从：
$$V(s; \theta) = \sum_{a} \pi(a \mid s; \theta) Q^\pi(s, a)$$
$\theta$ 求导，把导数推进求和号：
$$\nabla_\theta V(s; \theta) = \sum_{a} \nabla_\theta \pi(a \mid s; \theta) Q^\pi(s, a)$$
⚠️ 讲义特别提醒：这里“假$Q^\pi$ $\theta$ 无关”，推导并不严格
### 2改写为期望形式（便于采样
把求和式改写为：$$\nabla_\theta V(s; \theta) = \mathbb{E}_{A \sim \pi(\cdot \mid s; \theta)} [\nabla_\theta \log \pi(A \mid s; \theta) \cdot Q^\pi(s, A)]$$
💡 补充理解：常用恒等式$$\nabla_\theta \pi(a \mid s; \theta) = \pi(a \mid s; \theta) \nabla_\theta \log \pi(a \mid s; \theta)$$
## 2.6 如何计算策略梯度：采样估计与更新步骤
本章导读<br>
理论期望不可直接计算时，用采样近似：先从策略采样动作，再用某种方式估$Q^\pi(s, a)$，从而构造无偏梯度估计并更新网络

### 1单步梯度估计
讲义步骤
* 采样动作 $a_g \sim \pi(\cdot|s; \theta)$
* 计算 $g(a_g, \theta) = \nabla_\theta \log \pi(a_g|s; \theta) \cdot Q^\pi(s, a_g)$
并说$\mathbb{E}[g(A, \theta)] = \nabla_\theta V(s; \theta)$，因$g$ 是无偏估计
### 2训练循环 
把“训练一次更新”写成：
* 观察 $s_t$
* 采样 $a_t \sim \pi(\cdot|s_t; \theta_t)$
* 计算 $q_t \approx Q^\pi(s_t, a_t)$<mark>某种估计</mark>
* 反传得到 $d\ell_t = \nabla_\theta \log \pi(a_t|s_t; \theta)$ 
* 近似梯度g_t = q_t \cdot d\ell_t$
* 更新\theta_{t+1} = \theta_t + \beta g_t$<br>
📌 一句话总结这一段：策略梯度更新 =（动作价值估计）$\times$（该动作 log 概率对参数的梯度）
## 2.7 关键Q^\pi$ 怎么估？
### 1Option 1: REINFORCE (蒙特卡洛回报)
* 方法：运行到回合结束，计算实际的折扣回报 $u_t = \sum_{k=t}^T \gamma^{k-t} r_k$
* 估算：令 $q_t = u_t$
* 代价：必须等待回合结束，且方差通常较大
### 2Option 2: Actor-Critic (用网络逼近 $Q$)
* 方法：引入第二个神经网络来估$Q^\pi$
* 结构：Actor：策略网络$\pi(a \mid s; \theta)$。Critic：价值网络，Actor 提供低方差的 $q_t$ 估计

# ⭐Actor-Critic 示意

**Actor 按概率采样动作与环境交互得到 $(s, a, r, s')$ $\rightarrow$ Critic 利用 TD 目标计算误差并更新自身网络$\rightarrow$ Actor 根据 Critic 的价值评估进行策略梯度更新$\rightarrow$ 使用 新策开启下一轮交互*

![Gemini_Generated_Image_wb2jzkwb2jzkwb2j](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_wb2jzkwb2jzkwb2j.png)

## 1.上半部分：与环境交互（产生一步经验）

- **环境 $\rightarrow$ 智能体：给出状态$s_t$**

  环境提供当前观测/局部$s_t$

- **Actor 输出动作概率分布**

  Actor（策略网络）$\pi(a|s;\theta)$ 接收 $s_t$，输出各动作概率（经Softmax 处理后的概率向量）

- **动作采样a_t \sim \pi(\cdot|s_t;\theta)$**

  按概率分布“采样”一个动作$a_t$

  > **注意* 动作随机性来自策略本身（这与 Value-based 方法使用 argmax 选择动作不同）

- **智能$\rightarrow$ 环境：执行动作$a_t$**

  将采样得到的动作送到环境执行

- **环境 $\rightarrow$ 智能体：返回奖励 $r_t$ 和下一状态$s_{t+1}$**

  环境反馈即时奖励 $r_t$ 与下一状态$s_{t+1}$（环境转移可能具有随机$p(s'|s,a)$）

**打包为“一步经验”：**$$(s_t, a_t, r_t, s_{t+1})$$  送往学习部分

## 2. 下半部分 A→E：学习更新流

- **A. 【输入经验】并$s_{t+1}$ 采样“下一动作”（不执行）**

  输入(s_t, a_t, r_t, s_{t+1})$

  进行采样\tilde a_{t+1} \sim \pi(\cdot|s_{t+1};\theta)$

  > **标注* 不执$\tilde a_{t+1}$，仅用于估计价值。此过程类似SARSA / On-policy TD

- **B. 【Critic 评估】计算当前与下一步的 Q *

  Critic（价值网络）$q(s,a;w)$ 近似 $Q^\pi(s,a)$

  - $q_t = q(s_t, a_t; w)$
  - $q_{t+1} = q(s_{t+1}, \tilde a_{t+1}; w)$

- **C. 【TD 目标TD 误差】构造学习信*

  - **TD 目标 (Target)* $y_t = r_t + \gamma \cdot q_{t+1}$
  - **TD 误差 (Error)* $\delta_t = q_t - y_t$
  - 其中 $\gamma$ 是折扣因子，$\delta_t$ 衡量 Critic 预测与实际目标的偏差

- **D. 【更新Critic】（梯度下降，拟TD 目标*

  TD 误差作为监督信号，更新参数$w$

  $$w \leftarrow w - \alpha \cdot \delta_t \cdot \nabla_w q(s_t, a_t; w)$$

  *Critic 的监督信号源自奖励$r_t$（通过 TD 学习获得）

- **E. 【更新Actor】（策略梯度/梯度上升*

  Actor 利用 Critic 提供的价值评估调整策略参数$\theta$

  - **计算梯度* $g_t = \nabla_\theta \log \pi(a_t|s_t;\theta) \cdot q_t$
  - **更新参数* $\theta \leftarrow \theta + \beta \cdot g_t$
  - **直觉解释* $q_t$ 较大（动作好），则提高该状态下该动作的概率；反之则降低
  - **优化建议* 可使Advantage $A_t$ 替代 $q_t$ 以降低方差，提高稳定性

## 3. 最后回到闭环：更新后再交互

完成步骤 E 后，流程沿箭头回到上半部分：

**更新后的 Actor 继续与环境交$\rightarrow$ 产生新经$\rightarrow$ 再次更新*

# 3.Actor-Critic 方法

## 3.1概念

强化学习常见的三类思路

- **价值型 (Value-Based)**：学$Q$ $V$，通过“价值最大化”来选择动作
- **策略(Policy-Based)**：直接学习策$\pi(a|s)$，按策略概率分布采样动作
- <mark>**Actor-Critic**：将二者结合*Actor** 学习策略*Critic** 学习价值并指导 Actor</mark>

> **直觉理解**：Actor 负责“怎么做”（动作生成），Critic 负责“做得好不好”（评分）。训练时，Critic Actor 提供更稳定的学习信号，解决纯策略梯度方差大的问题

![b9f250f3-f660-4e62-b4da-9eead1d55bf7](https://raw.githubusercontent.com/songqi4485/RL_Git/main/b9f250f3-f660-4e62-b4da-9eead1d55bf7.png)

## 3.2Value Network and Policy Network

### 1）Policy Network（Actor

用神经网络近似策$\pi(a|s)$

- $\pi(a|s;\theta)$近似真实策略 $\pi(a|s)$
- θ是策略网络可训练参数

* 输入：状态s$
* 输出*对所有动作的概率分布**
* 动作集合 $\mathcal A$上概率和1\sum_{a\in\mathcal A}\pi(a|s;\theta)=1$，所以使**Softmax** 把网络输出变成合法分布

![da7ee69c-dd16-4533-9246-4bb6cfe31680](https://raw.githubusercontent.com/songqi4485/RL_Git/main/da7ee69c-dd16-4533-9246-4bb6cfe31680.png)

### 2Value Network（Critic

用神经网络近似动作价值Q^\pi(s,a)$

* $q(s,a;w$) 近似 $Q^\pi(s,a)$
* w 是价值网络可训练参数
* 输入：状态s$
* 输出：该状态下**所有动作的 action-value（每个动作一q值）**

![beed1eea-1327-4784-aacf-be80c9d9b7c0](https://raw.githubusercontent.com/songqi4485/RL_Git/main/beed1eea-1327-4784-aacf-be80c9d9b7c0.png)

### 3）耦合网络

$$
V^\pi(s) = \sum_a \pi(a|s) \cdot Q^\pi(s, a)\approx \sum_a \pi(a|s;\theta) \cdot q(s,a;w)
$$

**Actor 产出“权重”（动作概率），Critic 产出“评分”（动作价值），二者合成对状态的评价**

## 3.3训练神经网络

### 1）使用TD算法更新价值网络

* **Compute** $q(s_t, a_t; \mathbf{w}_t)$ and $q(s_{t+1}, a_{t+1}; \mathbf{w}_t)$.
* **TD target**: $y_t = r_t + \gamma \cdot q(s_{t+1}, a_{t+1}; \mathbf{w}_t)$.
* **Loss**: $L(\mathbf{w}) = \frac{1}{2} [q(s_t, a_t; \mathbf{w}) - y_t]^2$.
* **Gradient descent**: $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \left. \frac{\partial L(\mathbf{w})}{\partial \mathbf{w}} \right|_{\mathbf{w}=\mathbf{w}_t}$

<mark>这里采用梯度下降是为了让预测与TD target更接</mark>

### 2）使用策略梯度更新策略网络

* 定义单步梯度估计$g(a,\theta)\propto \nabla_\theta \log \pi(a|s;\theta)\cdot q(s,a;w)$
* 用随机采样保证无偏，并用 **随机梯度上升**更新

<mark>这里采用梯度上升是为了让平均分更新/mark>

## 3.4算法总结

1）观测状态$s_t$ 、随机抽$a_t \sim \pi(\cdot | s_t; \boldsymbol{\theta}_t)$.

2）执$a_t$; 然后新的环境生成 $s_{t+1}$ 奖励 $r_t$.

3）随机抽$\tilde{a}_{t+1} \sim \pi(\cdot | s_{t+1}; \boldsymbol{\theta}_t)$. (Do not perform $\tilde{a}_{t+1}$!)

4）评估价值网络$q_t = q(s_t, a_t; \mathbf{w}_t)$ and $q_{t+1} = q(s_{t+1}, \tilde{a}_{t+1}; \mathbf{w}_t)$.

5）计算TD误差: $\delta_t = q_t - (r_t + \gamma \cdot q_{t+1})$.

6）对价值网络求 $\mathbf{d}_{\mathbf{w},t} = \left. \frac{\partial q(s_t, a_t; \mathbf{w})}{\partial \mathbf{w}} \right|_{\mathbf{w}=\mathbf{w}_t}$.

7*更新价值网络** $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \delta_t \cdot \mathbf{d}_{\mathbf{w},t}$.

8）对策略网络求导: $\mathbf{d}_{\boldsymbol{\theta},t} = \left. \frac{\partial \log \pi(a_t | s_t, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \right|_{\boldsymbol{\theta}=\boldsymbol{\theta}_t}$.

9*更新策略网络:** $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \beta \cdot q_t \cdot \mathbf{d}_{\boldsymbol{\theta},t}$.

$\tilde{a}_{t+1}$是一个假想动作，只是用来计算$q$。算法每一次循环只执行$a_t$这一个动作。因为TD算法需\tilde{a}_{t+1}$，从而更新q的梯度下降法也需要

# ⭐AlphaGo工作示意

![Gemini_Generated_Image_wkqib6wkqib6wkqi](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_wkqib6wkqib6wkqi.png)

# 4.**AlphaGo**

## 4.1核心主题

- **核心范式* AlphaGo 的“学+ 搜索”范式
- **深度神经网络学习*
  - **策略网络 (Policy Network)* 给出“下一步走哪儿”的概率分布
  - **价值网络(Value Network)* 评估局面好坏（输的期望）
- **蒙特卡洛树搜(MCTS) 决策* 在真实落子前，进行大量“假想推演”，综合策略先验与价值评估选出最终动作

## 4.2问题设定与数据细

**动作空间* $\mathcal{A} \subset \{1, 2, \dots, 361\}$（基19×19 棋盘）

**状态表示：** 布局可简化为 $19 \times 19 \times 2$ 0/1 张量；实际使用更丰富的特征（$19 \times 19 \times 48$）

**网络输入维度*

- **AlphaGo 策略网络* 输入可达 $19 \times 19 \times 48$
- **AlphaGo Zero* 输入示例$19 \times 19 \times 17$，且采用“策略头 + 价值头”的共享主干结构

## 4.3AlphaGo 整体流程

**训练阶段 (3 **

1. **行为克隆 (Behavior Cloning, BC)* 利用人类棋谱监督学习初始化策略网络
2. **策略梯度 (Policy Gradient, PG)* 通过自我对弈继续强化策略网络
3. **训练价值网络(Value Network)* 使用强化后的策略生成数据来学习局面价值

**执行阶段 (真正下棋)**

- 利用策略网络和价值网络辅**MCTS**，经过大量模拟后选择最终动作

## 4.4. Step 1. Behavior Cloning

### 1）必要

若策略网络一开始随机：

- 两个随机策略互弈基本等于乱下
- 361 分支的超大空间里，靠纯随机探索学到“像样的围棋”会非常慃69

人类棋谱提供*强先*：先学会“像人一样走”，再用 RL 去超越

### 2）数字

 KGS 数据集（160K 局人类对局）

### 3）训练步

* 给定状态$s_t$，策略网络输出：
  $$
  \mathbf{p}_t = (\pi(a=1|s_t;\theta), \dots, \pi(a=361|s_t;\theta))
  $$
  
* 标签：人类专家动作$a_t^*$ One-hot 编码 $\mathbf{y}_t$

* 损失：交叉熵

$$
L = \text{CrossEntropy}(\mathbf{y}_t, \mathbf{p}_t)
$$

* 用梯度下降更新策略网络参数$\theta$

<mark>注意：BC 属于**模仿学习**（分回归问题），不依赖环境奖励</mark>

## 4.5. Step 2. Policy Gradient

让策略“超过人类模仿

### 1）RL的必要

BC 的致命问题：**分布外状态（OOD*

- 如果当前状态 s_t$在训练集中出现过，模仿专家动作大概率不错
- 但围棋状态空间巨大，很多状态根本没在棋谱里出现过，此时 BC 可能走出很差的动作

### 2）自我博弈环

Player（当前要更新的策略）：用最新参数的策略网络

Opponent（环对手）：从历史迭代版本里随机抽取旧参数策略

- 这相当于“对手池”，能避免只针对某一个固定对手过拟合

![3ac573a5-8256-4c76-9b63-af38d83975aa](https://raw.githubusercontent.com/songqi4485/RL_Git/main/3ac573a5-8256-4c76-9b63-af38d83975aa.png)

### 3）奖励与回报（回合结束才给奖励

**终局部T$*

- 终局前：$r_0=r_1=\dots=r_{T-1}=0$
- 终局：赢$r_T=+1$，输$r_T=-1$

**回报（不折扣）：**
$$
u_t=\sum_{k=t}^{T}r_k
$$
因此赢家所u_t=+1$，输家所$u_t=-1$

### 4）策略梯度更新

* 用回报u_t$替代动作价值估计，得到近似梯度项：

$$
\nabla_\theta \log \pi(a_t|s_t;\theta)\cdot u_t
$$

* 对整局轨迹的梯度求和得 *g*，更新：

$$
\theta \leftarrow \theta + \beta \cdot g
$$

## 4.6. Step 3. Value Network

### 1价值函数定

- 价值函数（状态价值）
  $$
  \hat V(s)=\mathbb{E}[U_t\mid S_t=s],\quad U_t\in\{+1,-1\}
  $$
  期望对未来动作与未来状态取

- <mark>使用神经网络$v(s;w)$估计$\hat V(s)$</mark>

![cb079794-853b-4507-814a-b63352179cd7](https://raw.githubusercontent.com/songqi4485/RL_git/main/cb079794-853b-4507-814a-b63352179cd7.png)

### 2）训练步

<mark>**不是 actor-critic**，因为不是同时训练两者，而是“策略先训好，再单独训价值”</mark>

* 用已训练好的策略网络自我博弈到终局，得到整局所 u_t\in\{+1,-1\}$

* 回归损失（平方误差）

$$
L=\sum_{t=0}^{T}(v(s_t;w)-u_t)^2
$$

* 梯度下降w \leftarrow w-\alpha \nabla_w L$



回顾训练

1.behavior cloning。根据人的棋谱初步训练策略网络

2.策略梯度算法。进一步训练策略网络

3.结束训练策略网络之后，单独训练价值网络V

下棋的时候策略网络和价值网络都不做决策，蒙特卡洛树搜素做决策。该方法不需要训练可以直接用来下棋。之前训练两个网络是为了帮助蒙特卡洛树搜素

## 4.6. Step 4 . 执行阶段:MCTS

### 1）必要

* 人类下棋靠“向前看很多步”；如果能穷举未来所有分支就能赢，但分支巨大做不到
* **策略网络的价值*：给出强先验，能“排除大多数不好的动作”，把搜索重点放在更可能的分支上

### 2）MCTS流程

* **Selection**：玩家选择一个动作（注意：这是*假想动作**”，不是实际落子
* **Expansion**：对手选择动作、状态转移（同样是假想动作，由策略网络驱动）
* **Evaluation**：评估状态价值并获得评分 $v$。将游戏玩至结束以获得奖励$r$。将评分 $\frac{v+r}{2}$ 分配给动作$a$
* **Backup**：使用评估$\frac{v+r}{2}$ 来更新动作价值

### 3）Selection

**搜索树每条边 $(s, a)$ 存储的信息：**

- **$Q(s, a)$**：由 MCTS 计算的动作价值（待定义）
- **$N(s, a)$**：在给定 $s_t$ 的情况下，目前选择 $a$ 的次数字
- **$\pi(a \mid s_t; \boldsymbol{\theta})$**：已学习的策略网络

**选择规则*
$$
a_t = \arg\max_a \big( Q(s_t, a) + u(s_t, a) \big), \quad u(s, a) \propto \frac{\pi(a \mid s_t; \boldsymbol{\theta})}{1 + N(s, a)}
$$

- **探索与利用：** 先验概率 $\pi(a \mid s_t; \boldsymbol{\theta})$ 越大越倾向于探索，但随着访问次数 $N$ 的增加，探索奖励 $u$ 会逐渐衰减
- 选择评分最高的动作进行执行

### 4）Expansion

用策略网络“模拟对手”，把策略当成转移模型

对对手动作从策略分布采样a_t^{(opp)} \sim \pi(\cdot | s_t^{(opp)}; \theta)$

**关键解释*

- 真实转移概率 $p(s_{t+1} | s_t, a_t)$ 不好显式写出（因为对手是智能体）
- 用策$\pi$ 近似/充当转移函数 $p$ 来做“向前推演”

### 5）Evaluation

价值网络+ 快Rollout (两条路评估叶

- **Rollout* 用较快策略把局面走到终局拿到 $z_L$ 或终局奖励（赢 $+1$ / $-1$）
- **价值网络：** 直接给叶子局面一个评估$v_\theta(s_L)$
- **两者融合：** $V(s_L) = (1 - \lambda) v_\theta(s_L) + \lambda z_L$

### 6）Backup

更新 $Q$ $N$ (用平均值累

- **访问次数累积* 
  $$
  N(s, a) = \sum_{i=1}^{n} \mathbf{1}(s, a, i)
  $$
  
- **动作价值：** 用“过该边的模拟评估均值”计算：
  $$
  Q(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^{n} \mathbf{1}(s, a, i) V(s_L^{(i)})
  $$

### 7最终落子规

- 做完很多MCTS 模拟后，实际落子选：
  $$
  a_t = \arg\max_a N(s_t, a)
  $$
  $N(s_t, a)$:在状态s_t$下动作a被选择的次数字

- **下一手再来一次完整搜*

  - 重新初始$Q, N$ $0$，再做成千上万次模拟
  - 示例：李世石走完一步，再次轮到 AlphaGo，会重新MCTS 并重$Q/N$

## 4.7 Alpha Go Zero vs Alpha Go

- 不用人类经验*没有 BC**
- 训练时用 MCTS*AlphaGo Zero MCTS 训练策略网络，AlphaGo 训练策略时不MCTS**

## 4.8 主要结论

- **结论 1：AlphaGo 的本质是“学一个强先验 + 学一个快评估 + 用搜索做最终决策*
  - 策略网络提供先验 $P$（减少分支）
  - 价值网络提$v(s)$（减rollout 计算）；
  - MCTS 把两者整合起来做 look-ahead，最终以访问次数 $N$ 决策
- **结论 2：BC 是“加速器”，PG 自我博弈是“超越器*
  - **行为克隆 (BC)** 让策略快速达到“像人类高手”的起点
  - 自我博弈 + 策略梯度让策略突破模仿分布，解决“训练集没见过的局面”
- **结论 3：价值网络训练要格外注意样本相关性导致的过拟*
  - 用整局棋谱直接监督会因强相关性而“记忆胜负”；
  - 用大量自我博弈、抽取更独立的局面（3000 万个不同位置）能显著缓解

# 5. Monte Carlo算法

## 5.0统一思想:把目标量写成期望

1.设随机变X*，构造函h(\cdot)$，使得目标量
$$
Q = C \,\mathbb{E}[h(X)]
$$
2.采样$X_1,\dots,X_N$（独立同分布 i.i.d.），用样本均值估计期望：
$$
\widehat{\mathbb{E}[h(X)]}=\frac{1}{N}\sum_{i=1}^N h(X_i)
\quad\Rightarrow\quad
\widehat{Q}=C\cdot \frac{1}{N}\sum_{i=1}^N h(X_i)
$$
3.误差规模
$$
\widehat{Q}\xrightarrow[N\to\infty]{} Q,\qquad
\mathrm{Var}(\widehat{Q})=\frac{C^2\,\mathrm{Var}(h(X))}{N},\qquad
\text{典型误差}\sim O(N^{-1/2})
$$

## 5.1案例1:随机抽样计算$\pi$

![image-20260308161140942](https://raw.githubusercontent.com/songqi4485/RL_Git/main/image-20260308161140942.png)

在单位正方形$[0,1]\times[0,1]$内均匀随机撒点 $(X,Y)\sim \mathrm{Unif}([0,1]^2)$)
$$
S=\{(x,y)\in[0,1]^2:\ x^2+y^2\le 1\}
$$
定义$n$为落在圆里的个数，定义P为落在圆里的概率，则期望
$$
Pn=\frac{\pi*n}{4}
$$
定义$m$为真实落在圆里的个数，则
$$
m≈\frac{\pi n}{4},\pi≈\frac{4m}{n}
$$

## 5.2案例2:Buffon投针计算$\pi$

![e70e8eed-83c2-4189-9523-f12e594b892a](https://raw.githubusercontent.com/songqi4485/RL_Git/main/e70e8eed-83c2-4189-9523-f12e594b892a.png)

平行线之间的距离$d$，针的长度为 $l$。随机投掷一枚针；针可能会也可能不会跨越（其中一条）线。针与线相交的概率为 $P = \frac{2l}{\pi d}$可用积分证明)

* 针与平行线夹角：$\Theta\sim \mathrm{Unif}(0,\pi/2)$
* 针中心到最近直线的距离D\sim \mathrm{Unif}(0,t/2)$

命中条件
$$
D\le \frac{l}{2}\sin\Theta
$$
联合密度
$$
f_{D,\Theta}(d,\theta)=\frac{2}{t}\cdot\frac{2}{\pi}
$$
因此
$$
P=\int_0^{\pi/2}\int_0^{(l/2)\sin\theta}\frac{4}{\pi t}\,\mathrm{d}d\,\mathrm{d}\theta
=\frac{2l}{\pi t}\int_0^{\pi/2}\sin\theta\,\mathrm{d}\theta
=\frac{2l}{\pi t}
$$
所以：
$$
\pi=\frac{2l}{tP},\qquad \widehat{\pi}=\frac{2l}{t\widehat{P}},\ \widehat{P}=\frac{K}{N}
$$

## 5.3案例3:估计阴影面积

![4591f38c-5e4d-41de-bf64-9a2267061495](https://raw.githubusercontent.com/songqi4485/RL_Git/main/4591f38c-5e4d-41de-bf64-9a2267061495.png)

定义$A_1$为圆的面积，$A_2$为阴影部分的面积

则点落在阴影区域的概P$为：
$$
P=\frac{A_2}{A_1}
$$
则：
$$
nP=\frac{nA_2}{A_1}
$$
$m$是观测到在阴影区域点的个数，则：
$$
m≈\frac{nA_2}{A_1},A_2≈\frac{mA_1}{n}
$$

## 5.4案例4:近似求积

给定一个一元函$f(x)$，计算定积分
$$
I = \int_{a}^{b} f(x) dx
$$

* 从区$[a, b]$ 中随机均匀地抽$n$ 个样本；将其记为 $x_1, \dots, x_n$
*  计算样本均值并缩放，得$Q_n$

$$
Q_n = (b - a) \cdot \frac{1}{n} \sum_{i=1}^{n} f(x_i)
$$

* 返回 $Q_n$ 作为积分 $I = \int_{a}^{b} f(x) dx$ 的近似值

理论依据 :大数定律 保证了当样本$n \to \infty$ 时，$Q_n \to I$



给定一个多元函$f(\mathbf{x})$，计算在集合 $\Omega$ 上的积分
$$
I = \int_{\Omega} f(\mathbf{x}) d\mathbf{x}
$$

* 从集$\Omega$ *均匀随机*抽取 $n$ 个样本；将其记为 $\mathbf{x}_1, \dots, \mathbf{x}_n$
* 计算集合 $\Omega$ *体积（或测度* $V$

$$
V = \int_{\Omega} d\mathbf{x}
$$

* 计算 $Q_n$          

$$
Q_n = V \cdot \frac{1}{n} \sum_{i=1}^{n} f(\mathbf{x}_i)
$$

* 返回 $Q_n$ 作为积分 $I = \int_{\Omega} f(\mathbf{x}) d\mathbf{x}$ 的近似值

<mark>近似求积分的思想可以联想到定积分求定</mark>

## 5.5案例5:计算期望

估计数学期望 $\mathbb{E}_{X \sim p}[f(X)] = \int_{\mathbb{R}^d} f(\mathbf{x}) \cdot p(\mathbf{x}) d\mathbf{x}$

* 从概率分$p(\mathbf{x})$ 中抽$n$ 个随机样本，记为 $\mathbf{x}_1, \dots, \mathbf{x}_n$
* 计算函数值的算术平均Q_n = \frac{1}{n} \sum_{i=1}^{n} f(\mathbf{x}_i)$
* 返回 $Q_n$ 作为期望$\mathbb{E}_{X \sim p}[f(X)]$ 的近似值

# ⭐SARSA示意

SARSA 每一步都是：先按当前策略采样$(s_t,a_t,r_t,s_{t+1},a_{t+1})$ $y_t$ 让当前的 $Q(s_t,a_t)$（表格格子或网络输出）往 $y_t$ 靠近一点

![Gemini_Generated_Image_dr3ctndr3ctndr3c](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_dr3ctndr3ctndr3c.png)

# 6. SARSA算法

## 6.1关键推导

### 1）折扣回报$U_t$ 与关键恒等式

定义折扣回报 $U_t$
$$
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + \cdots
$$
先写$U_{t+1}$
$$
U_{t+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$
两边同乘 $\gamma$
$$
\gamma U_{t+1} = \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + \cdots
$$
$U_t$的定义相加对齐可得：
$$
U_t = R_t + \gamma U_{t+1}
$$

### 2）动作价值函数$Q^\pi(s_t,a_t)$ 定义

给定当前状态动作后，未来折扣回报的条件期望
$$
Q^\pi(s_t,a_t)=\mathbb{E}\!\left[U_t \mid S_t=s_t,\;A_t=a_t\right]
$$

### 3）推导贝尔曼期望方程

定义
$$
Q^\pi(s_t,a_t)=\mathbb{E}\!\left[U_t \mid S_t=s_t,\;A_t=a_t\right]
$$
代入恒等$U_t=R_t+\gamma U_{t+1}$
$$
Q^\pi(s_t,a_t)=\mathbb{E}\!\left[R_t+\gamma U_{t+1}\mid S_t=s_t,\;A_t=a_t\right]
$$
利用期望线性性展开
$$
Q^\pi(s_t,a_t)=
\mathbb{E}\!\left[R_t\mid S_t=s_t,\;A_t=a_t\right]
+
\gamma\mathbb{E}\!\left[U_{t+1}\mid S_t=s_t,\;A_t=a_t\right]
$$

---

先引入一个中间条(S_{t+1},A_{t+1})$，用全期望公式：
$$
\mathbb{E}[X \mid Z]
=
\mathbb{E}\!\left[\ \mathbb{E}[X \mid Y, Z]\ \middle|\ Z\right]
$$

$$
\mathbb{E}\!\left[U_{t+1}\mid S_t=s_t,\;A_t=a_t\right]
=
\mathbb{E}\!\Big[
\mathbb{E}\!\left[U_{t+1}\mid S_{t+1},A_{t+1}\right]
\;\Big|\; S_t=s_t,\;A_t=a_t
\Big]
$$

注意到按定义
$$
\mathbb{E}\!\left[U_{t+1}\mid S_{t+1},A_{t+1}\right]
=
Q^\pi(S_{t+1},A_{t+1})
$$
代回去得到：
$$
\mathbb{E}\!\left[U_{t+1}\mid S_t=s_t,\;A_t=a_t\right]
=
\mathbb{E}\!\left[Q^\pi(S_{t+1},A_{t+1})\mid S_t=s_t,\;A_t=a_t\right]
$$

---

将上面结果代回到 $Q^\pi$ 的等式链
$$
Q^\pi(s_t,a_t)
=
\mathbb{E}\!\left[R_t\mid S_t=s_t,\;A_t=a_t\right]
+
\gamma\mathbb{E}\!\left[Q^\pi(S_{t+1},A_{t+1})\mid S_t=s_t,\;A_t=a_t\right]
$$

$$
Q^\pi(s_t,a_t)
=
\mathbb{E}\!\left[
R_t+\gamma Q^\pi(S_{t+1},A_{t+1})
\mid S_t=s_t,\;A_t=a_t
\right]
$$

### 3）TD target $y_t$ 的由

不知道期望，因此用采样（MC 思想）近似

一次采样会给出转移
$$
(s_t,a_t,r_t,s_{t+1})
$$
并在 $s_{t+1}$ 处按当前策略采样下一动作
$$
a_{t+1}\sim \pi(\cdot\mid s_{t+1})
$$
用该样本把期望替换为一次观测，得到 **一TD 目标**
$$
Q^\pi(s_t,a_t)
=\mathbb{E}\!\left[r_t+\gamma Q^\pi(s_{t+1},a_{t+1})\mid S_t=s_t,\;A_t=a_t\right]
$$

$$
y_t = r_t + \gamma Q(s_{t+1},a_{t+1})
$$

<mark>TD learning：让$Q^\pi(s_t,a_t)$接近$y_t$</mark>

这是因为$Q^\pi(s_t,a_t)$完全是估计，而TD target部分基于真实的奖励更可靠）。把$y_t$当作观测值固定住，改变动作价值Q^\pi(s_t,a_t)$接近$y_t$

## 6.2 SARSA（表格版

**TD target**
$$
y_t = r_t + \gamma Q(s_{t+1},a_{t+1})
$$
**TD error**:
$$
\delta_t = Q(s_t,a_t)-y_t
$$
**表格更新**
$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)-\alpha\delta_t
$$

## 6.2 SARSA（神经网络版

![4b2b8aa7-8df2-4399-afad-633ff1d5a22d](https://raw.githubusercontent.com/songqi4485/RL_Git/main/4b2b8aa7-8df2-4399-afad-633ff1d5a22d.png)

### 1）TD target TD error

$$
y_t = r_t + \gamma\, q(s_{t+1},a_{t+1};\mathbf{w})
$$

$$
\delta_t = q(s_t,a_t;\mathbf{w})-y_t
$$

### 2）定义损失函$L(\mathbf{w})$

$$
L(\mathbf{w})=\frac{1}{2}\delta_t^2
$$

### 3）对 $\mathbf{w}$ 求导

$$
\nabla_{\mathbf{w}}L(\mathbf{w})
=
\nabla_{\mathbf{w}}\left(\frac{1}{2}\delta_t^2\right)
=
\delta_t\cdot \nabla_{\mathbf{w}}\delta_t
$$

代入 $\delta_t=q(s_t,a_t;\mathbf{w})-y_t$
$$
\nabla_{\mathbf{w}}\delta_t
=
\nabla_{\mathbf{w}}q(s_t,a_t;\mathbf{w})-\nabla_{\mathbf{w}}y_t
$$
由于$y_t$ 当作常数
$$
\nabla_{\mathbf{w}}y_t \approx 0
\quad\Rightarrow\quad
\nabla_{\mathbf{w}}\delta_t \approx \nabla_{\mathbf{w}}q(s_t,a_t;\mathbf{w})
$$

$$
\nabla_{\mathbf{w}}L(\mathbf{w})
=
\delta_t\cdot\nabla_{\mathbf{w}}q(s_t,a_t;\mathbf{w})
$$

### 3）梯度下降更新

$$
\mathbf{w}\leftarrow \mathbf{w}-\alpha\nabla_{\mathbf{w}}L(\mathbf{w})
=
\mathbf{w}-\alpha\delta_t\nabla_{\mathbf{w}}q(s_t,a_t;\mathbf{w})
$$

# ⭐Q-Learning算法示意

![Gemini_Generated_Image_wtttfewtttfewttt](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_wtttfewtttfewttt.png)

# 7. Q-Learning算法

##  7.1关键知识与推

### 1SARSA vs Q-learning

* **SARSA**：学习策$\pi$ 下的动作价值$Q^\pi(s,a)$，其 TD 目标

$$
y_t = r_t + \gamma Q^\pi(s_{t+1}, a_{t+1})
$$

并用于更新价值网络（critic）

* **Q-learning**：学习最优动作价值$Q^\star(s,a)$，其 TD 目标

$$
y_t = r_t + \gamma \max_{a} Q^\star(s_{t+1}, a)
$$

并用于更新DQN

### 2）TD target推导

$$
Q^\pi(s_t,a_t)=\mathbb{E}\big[R_t+\gamma Q^\pi(S_{t+1},A_{t+1})\big]
$$

$\pi=\pi^\star$，则有：
$$
Q^{\pi^\star}(s_t,a_t)=\mathbb{E}\big[R_t+\gamma Q^{\pi^\star}(S_{t+1},A_{t+1})\big]
$$
$Q^{\pi^\star}$ $Q^\star$ 都表示最优动作价值函数字
$$
Q^\star(s_t,a_t)=\mathbb{E}\big[R_t+\gamma Q^\star(S_{t+1},A_{t+1})\big]
$$

---

最优下一动作定义
$$
A_{t+1}=\arg\max_{a}Q^\star(S_{t+1},a)
$$
因此
$$
Q^\star(S_{t+1},A_{t+1})=\max_{a}Q^\star(S_{t+1},a)
$$
代回恒等式得到：
$$
Q^\star(s_t,a_t)=\mathbb{E}\big[R_t+\gamma \max_{a}Q^\star(S_{t+1},a)\big]
$$

---

观测$R_t=r_t$ $S_{t+1}=s_{t+1}$，用当前估计值近$Q^\star$
$$
Q^\star(s_t,a_t)=\mathbb{E}\big[r_t+\gamma \max_{a}Q^\star(s_{t+1},a)\big]
$$
得到 Q-learning 的一TD target
$$
y_t=r_t+\gamma \max_{a}Q^\star(s_{t+1},a)
$$

## 7.2 Q-Learning(表格

**适用条件**：状态与动作有限，可直接画一$Q$ 表逐格更新

![7b9b90fb-5a36-4ccb-8445-0573b5a728e4](https://raw.githubusercontent.com/songqi4485/RL_Git/main/7b9b90fb-5a36-4ccb-8445-0573b5a728e4.png)

**一次更新用到的观测转移**
$$
(s_t,a_t,r_t,s_{t+1})
$$


**TD target**
$$
y_t=r_t+\gamma\max_{a}Q(s_{t+1},a)
$$
**TD error**
$$
\delta_t = Q(s_t,a_t)-y_t
$$
**更新*
$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)-\alpha\delta_t
$$

## 7.3 Q-Learning(DQN

**目标**：用 DQN 近似 $Q^\star$
$$
Q^\star(s,a)\approx Q(s,a;\mathbf{w})
$$
**控制（选动作）方式**
$$
a_t=\arg\max_{a}Q(s_t,a;\mathbf{w})
$$
**TD target**:
$$
y_t=r_t+\gamma\max_{a}Q(s_{t+1},a;\mathbf{w})
$$
**TD error**:
$$
\delta_t=Q(s_t,a_t;\mathbf{w})-y_t
$$
**参数更新**:
$$
\mathbf{w}\leftarrow \mathbf{w}-\alpha\cdot\delta_t\cdot\nabla_{\mathbf{w}}Q(s_t,a_t;\mathbf{w})
$$

## 7.4总结

**结论 1**：Q-learning 的关键在TD target 使用 $\max$（而不是采样到$a_{t+1}$），因此它学习的$Q^\star$，可用于“更新DQN 并按 $\arg\max$ 控制智能体”

**结论 2**：表格版DQN 版在数学上是一致的：都是让 $Q(s_t,a_t)$（表项或网络输出）向 $y_t$ 靠拢，只是“可学习参数”从表格单元变成了网络参数$\mathbf{w}$



# 8. Multi-Step TD Target

## 8.1多步TD好处

一TD 目标只用到了 **一*奖励 $r_t$。多TD 的想法是：既然我们在接下来的几步里还能继续拿到奖励，那就**接下$m$ *的奖励也纳入监督

* **励传播更新*：一TD 只把 $r_t$ 这点信息往回“传”一格；$m$ TD 能把接下来几步的奖励打包，更快影响更早的状态动作估计
* **信息更充*：多看几步奖励，目标 $y_t^{(m)}$ 往往更接近真实“这一步到底值多少”
* $m$ 也不是越大越好：$m$ 太大时，目标依赖更长的未来轨迹，可能更“抖”（更受随机性影响），所以需要“调得合适”

## 8.2推导

$$
U_t = R_t + \gamma U_{t+1}
$$

$$
U_{t+1} = R_{t+1} + \gamma U_{t+2}
$$

展开$2$ 步：
$$
U_t = R_t + \gamma R_{t+1} + \gamma^2 U_{t+2}
$$
展开$3$步：
$$
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 U_{t+3}
$$
推广到一般的 $m$ 步：
$$
U_t = \sum_{k=0}^{m-1} \gamma^k R_{t+k} + \gamma^m U_{t+m}
$$
前半段是 “未$m$ 步能拿到的折扣奖励”，最后一$\gamma^m U_{t+m}$ “从$t+m$ 步往后的剩余长期回报”



---

上面得到的是 **回报** $U_t$ 的展开式，但算法训练要的是“可用于监督的目$y_t$”。关键一步就是：把“未知的 $U_{t+m}$”用“当前的价值估计”来 **自举（bootstrap*

SARSA $m$ TD 目标
$$
y_t^{(m)} = \sum_{k=0}^{m-1} \gamma^k r_{t+k} + \gamma^m Q^\pi(s_{t+m}, a_{t+m})
$$
其中 $a_{t+m}$ 来自同一策略 $\pi$

Q-learning $m$ TD 目标
$$
y_t^{(m)} = \sum_{k=0}^{m-1} \gamma^k r_{t+k} + \gamma^m \max_a Q^\star(s_{t+m}, a)
$$

# ⭐Experience Replay示意

## ER核心概念

![Gemini_Generated_Image_i2yidmi2yidmi2yi](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_i2yidmi2yidmi2yi.png)

## ER训练流程(DQN的一次更新

![Gemini_Generated_Image_3z7v2x3z7v2x3z7v](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_3z7v2x3z7v2x3z7v.png)

## 相关性对比（顺序更新 vs 随机采样

![Gemini_Generated_Image_7jewae7jewae7jew](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_7jewae7jewae7jew.png)

## PER（Prioritized ER）机制流程图

![Gemini_Generated_Image_ywtsqsywtsqsywts](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_ywtsqsywtsqsywts.png)

## 重要性采样修正推导步骤图

![Gemini_Generated_Image_9bqrij9bqrij9bqr](https://raw.githubusercontent.com/songqi4485/RL_Git/main/Gemini_Generated_Image_9bqrij9bqrij9bqr.png)

# 9. Experience Replay

## 9.1经验回放的必要

### 1）缺1：经验浪

* transition $(s_t,a_t,r_t,s_{t+1})$，而“经验”是所$t=1,2,\dots$ transition 集合
* 传统在线 TD：用一次就丢，**浪费数据**

### 2）缺2：更新相

* 在线按时间顺序更新：$(s_t,a_t,r_t,s_{t+1})$、再$(s_{t+1},a_{t+1},r_{t+1},s_{t+2})$…
* 但连续状态$s_t$ $s_{t+1}$ **强相*，导致梯度更新也强相关，训练不稳定

## 9.2关键推导与适用边界

### 1）推

把每transition 存入回放缓冲区（replay buffer），只保留最$n$ 条：

* 存储(s_t,a_t,r_t,s_{t+1})$
* 超出容量则移除最旧样
* 容量 $n$ 是待调超参数，通常很大，如 $10^5\sim 10^6$

---

<mark>ER 训练的核心变化：**不按时间顺序用样*，而是buffer **随机采样**SGD</mark>

单条随机样本（索引为 $i$）：

* 抽样得到 $(s_i,a_i,r_i,s_{i+1})$
* 计算

$$
y_i=r_i+\gamma\max_a Q(s_{i+1},a;w),\quad
\delta_i=Q(s_i,a_i;w)-y_i
$$

* 梯度:

$$
g_i=\delta_i\cdot \nabla_w Q(s_i,a_i;w)
$$

* SGD 更新

$$
w\leftarrow w-\alpha g_i
$$

---

实践中每次会随机抽取多条 transition 组成 batch，算出多条随机梯度再求平均更新$w$

batch 大小$B$，则常用更新式为
$$
w\leftarrow w-\alpha\cdot \frac{1}{B}\sum_{i=1}^{B}\Big(\delta_i\nabla_w Q(s_i,a_i;w)\Big)
$$

### 2）优

* **让更新尽量不相关**（random sampling 打破序列相关性）
* **重复利用经验多次**（提升样本效率）

* 经验不足时立刻训练容易不稳定，因此实践中常先收集足够多的四元组再开始更新
* DQN 约收集到 $2\times10^5$ 条经验后再更新更好；Rainbow DQN 可在$8\times10^4$ 条时开始更新

### 3）适用边界

<mark>ER 能反复使用旧经验的前提：算法允许 **行为策略**（behavior policy）与 **目标策略**（target policy）不同</mark>

**异策略（off-policy）**：允许复用旧数据。例Q-learning、确定性策略梯度（DPG）等

**同策略（on-policy）**：要求数据来自当前策略，旧数据过时不能直接复用。如 SARSA、REINFORCE、A2C 等

<mark>**ER / PER DQN（off-policy）体系的关键组件**；对on-policy 算法通常不直接使用</mark>

## 9.3 Prioritized ER

### 1）必要

并非所transition 同等重要。如果某条样本的 TD 误差 $|\delta_t|$ 大，说明当前网络对它学得不准，应给予更高优先级

### 2）非均匀抽样(核心)

**ER（普通经验回放）**：从 replay buffer *均匀随机**抽样 transition

**PER（优先经验回放）**：用**非均匀抽样**代替均匀抽样

---

**TD 误差 $|\delta_t|$**定义重要样本

$|\delta_t|$ $\Rightarrow$ 当前网络对这条样本“预测错得多”，多抽它几$\Rightarrow$ 更快把“学得不准的地方”修正回报

---

**抽样概率 $p_t$ 的两种设*

* TD 误差幅度

$$
p_t \propto \delta_t + \epsilon
$$

其中 $\epsilon$ 用来避免概率$0$

* 按排rank

先把 transition $\delta_t$ 降序排序，令 $\mathrm{rank}(t)$ 是第 $t$ 条样本的名次
$$
p_t \propto \frac{1}{\mathrm{rank}(t)}
$$
PER 做的事就是：**$\delta_t$ 大的样本 $\Rightarrow$ $p_t$ $\Rightarrow$ 更常被抽*

---

### 3）重要性采样修

消除非均匀抽样带来的偏差<br>如果 buffer 里共$n$ transition*理想的“均匀抽样*相当于每条样本概率都$\frac{1}{n}$。PER 实际用的$p_t$（不相等）去抽样。结果就是，<mark>计算到的梯度期望会偏向“被高频抽到的样本”，从而引入偏差</mark>

---

**推导：把“均匀期望”改写成“按 $p(i)$ 抽样 + 权重*

令第 $i$ 条样本的“单样本梯度”记$g_i$:

* **均匀目标*的期望更新方

$$
\mathbb{E}_{\text{uniform}}[g]
=\sum_{i=1}^{n}\frac{1}{n}g_i
$$

* 乘除同一$p(i)$（只$p(i)>0$）：

$$
\sum_{i=1}^{n}\frac{1}{n}g_i
=\sum_{i=1}^{n}p(i)\cdot\frac{1}{n\,p(i)}g_i
=\mathbb{E}_{p}\!\left[\frac{1}{n\,p(i)}g_i\right]
$$

因此可以$p(i)$ 抽样，但每次更新要乘上权$\frac{1}{np(i)}$，这样才对应“均匀目标”的无偏估计

---

**实现方式：用 $(n,p_t)^{-\beta}$ 缩放学习*

* SGD 基本更新w \leftarrow w-\alpha\cdot g$
* 若采用重要性采样修正，则把学习率缩放为

$$
\alpha_t=\alpha\cdot (n\,p_t)^{-\beta},\quad \beta\in(0,1)
$$

开始时 $\beta$ 小，逐渐增加$1$

* 于是单样本更新可写为
  $$
  w \leftarrow w-\alpha\cdot (n\,p_t)^{-\beta}\, g_t
  $$

## 9.4 总结

* **每条 transition 都要绑定一TD 误差 $\delta_t$**
* **新样本刚buffer 时不知道 $\delta_t$**：直接把它的 $\delta_t$ 设为当前最大值，让它拥有最高优先级（保证新经验能尽快被学到）
* **每次抽到某条样本训练*：用最新网络重新计算它$\delta_t$，并更新该样本的优先级（$p_t \propto \delta_t+\epsilon$）



# 10. Dueling Network

## 10.1预备知识

### 1）折扣回报$U_t$

$$
U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + \cdots
$$

### 2）动作价值函数$Q^\pi(s,a)$

$$
Q^\pi(s_t,a_t)=\mathbb{E}\!\left[U_t\mid S_t=s_t,\ A_t=a_t\right]
$$

### 3）状态价值函数$V^\pi(s)$

$$
V^\pi(s_t)=\mathbb{E}_{A\sim \pi(\cdot\mid s_t)}\!\left[Q^\pi(s_t,A)\right]
=\sum_a \pi(a\mid s_t)\,Q^\pi(s_t,a)
$$

$$
V^\star(s)=\max_\pi V^\pi(s)
$$

### 4）最优价值函数

$$
Q^\star(s,a)=\max_\pi Q^\pi(s,a)
$$

### 5）最优优势函

（Optimal advantage function
$$
A^\star(s,a)=Q^\star(s,a)-V^\star
$$

## 10.2优势函数性质与关键定

### 1）定:$V^\star(s)=\max_a Q^\star(s,a)$

### 2）推$\max_a A^\star(s,a)=0$

由优势函数定义：$A^\star(s,a)=Q^\star(s,a)-V^\star(s)$

从定义出发取最大值：
$$
\max_a A^\star(s,a)=\max_a\Big(Q^\star(s,a)-V^\star(s)\Big)
$$
因为 $V^\star(s)$ 与动作$a$ 无关，可提出
$$
\max_a A^\star(s,a)=\max_a Q^\star(s,a)-V^\star(s)
$$
代入定理 1
$$
\max_a A^\star(s,a)=V^\star(s)-V^\star(s)=0
$$

### 3）定:用零项重$Q^\star(s,a)$

$$
Q^\star(s,a)=V^\star(s)+A^\star(s,a)-\max_{a'}A^\star(s,a')
$$

推导
$A^\star(s,a)=Q^\star(s,a)-V^\star(s)$ 得：
$$
Q^\star(s,a)=V^\star(s)+A^\star(s,a)
$$
上式右侧加上一个“恒$0$ 的项”不会改变等式：
$$
Q^\star(s,a)=V^\star(s)+A^\star(s,a)-0
$$

## 10.3DQN Dueling Network

### 1）DQN 的逼近目标

$$
Q(s,a;\mathbf{w})\approx Q^\star(s,a)
$$

### 2分解式逼近

* 用网络近$A^\star(s,a)$A(s,a;\mathbf{w}_A)\approx A^\star(s,a)$
* 用网络近$V^\star(s)$V(s;\mathbf{w}_V)\approx V^\star(s)$，且 $V(s;\mathbf{w}_V)$ 输出*实数标量**

* 实现时两支路可共享卷积特征（共享 $conv$ 参数）

![1fd8bc2e-5a41-4dce-8901-ae22aa895663](https://raw.githubusercontent.com/songqi4485/RL_Git/main/1fd8bc2e-5a41-4dce-8901-ae22aa895663.png)

### 3）Dueling Network 的组合公

$$
Q(s,a;\mathbf{w})=V(s;\mathbf{w}_V)+A(s,a;\mathbf{w}_A)-\max_{a'}A(s,a';\mathbf{w}_A),
\quad \mathbf{w}=(\mathbf{w}_A,\mathbf{w}_V)
$$

 **Alternative**
$$
Q(s,a;\mathbf{w})=V(s;\mathbf{w}_V)+A(s,a;\mathbf{w}_A)-\mathrm{mean}_{a'}A(s,a';\mathbf{w}_A)
$$
![95b704d3-0057-46bf-b530-c5d2921e5dea](https://raw.githubusercontent.com/songqi4485/RL_Git/main/95b704d3-0057-46bf-b530-c5d2921e5dea.png)

### 4）训练

Dueling network $Q(s,a;\mathbf{w})$ 仍是在逼近 $Q^\star(s,a)$，参数$\mathbf{w}=(\mathbf{w}_A,\mathbf{w}_V)$ 的学习方式与其他 DQN 相同。同样可以使用常DQN trick*Prioritized Experience Replay、Double DQN、Multi-step TD target**

<mark>**不要分别训练** $V$ $A$，而是训练整网络$Q(s,a;\mathbf{w})$ 拟合目标</mark>

## 10.4添加零项的必要

### 1）不可辨

以下等式（记Equation 1）存**non-identifiability**
$$
Q^\star(s,a)=V^\star(s)+A^\star(s,a)
$$

---

设存在一组真实分$(V^\star,A^\star)$ 满足 Equation 1

令：
$$
V_R(s)=V^\star(s)+10，A_R(s,a)=A^\star(s,a)-10
$$
则仍有：$Q^\star(s,a)=V_R(s)+A_R(s,a)$

结论：仅$Q^\star(s,a)$ 无法唯一确定 $V^\star(s)$ $A^\star(s,a)$，因此训练时两支路的“分工”可能漂移

### 2）固定范

给出 Equation 2并指出它**没有**不可辨识性问题：
$$
Q^\star(s,a)=V^\star(s)+A^\star(s,a)-\max_{a'}A^\star(s,a')
$$
直观上，它强制了
$$
\max_{a}A^\star(s,a)=0
$$
从而把“常数平移自由度”锁死

## 10.5总结

* Dueling Network 的本质是让网络显式输$V(s)$（标量）$A(s,a)$（向量），再
  $$
  Q(s,a)=V(s)+A(s,a)-\max_{a'}A(s,a')
  $$
  
  $$
  Q(s,a)=V(s)+A(s,a)-\mathrm{mean}_{a'}A(s,a')
  $$
  组合得到 $Q$

*  “减$\max$ / $\mathrm{mean}$”并不是为了改变 $Q^\star$ 的数值（因为该项在最优意义下可视$0$），而是为了**消除不可辨识*，使 $V$ $A$ 的分解有唯一的规范