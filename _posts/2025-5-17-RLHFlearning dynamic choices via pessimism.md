---
layout: post
title: "RLHF: reward learning:dynamic choices via pessimism"
date:   2025-5-17
tags: [强化学习, RLHF,Reward learning]
comments: true
author: Jincheng Ying

---







# RLHF:learning dynamic choices via pessimism

### 背景

离线RLHF从数据集中的轨迹中学习人类表现的策略,但面临挑战,一方面,人类反馈数据有限但状态空间非常大,人类决策的有限理性,异策略的分布偏移

**motivation**:计量经济学中的动态离散选择模型能够建模人类的行为选择,有四种估计方法,Nested fixed poing; conditional choice probability;MPEC;approximation.

为什么使用的是条件选择概率进行估计?

### 思想方法与结论

#### **核心思想**:

在离线RL中,agent无法直接在数据集中观测到环境中的奖励,但 数据集的(s,a,s')反映了人类在环境奖励影响下的状态与偏好动作,因此本文通过这种内涵的信息,借助Dynamic Discrete Choice (DDC)建模人类被环境驯化后的行为策略,从该策略中恢复了环境的奖励函数,再代入RL中去学习在奖励下的最优策略,因此最终次优性结论的一个重要条件是数据集中的轨迹能够覆盖最优策略.

#### **核心方法(DCPPO)**:

1. 首先根据离散动态选择模型

   ![image-20250523010336103](https://JinChengYing.github.io/images/image-20250523010336103.png)

   假设要求了![image-20250523010630267](https://JinChengYing.github.io/images/image-20250523010630267.png)

   model class足够大能够得到最优的r和Q,包含真实模型,且基于model class定义了惩罚函数$\rho$,model class可以是一族神经网络(ReLU)或核函数.

2. 通过极大似然估计(MLE)估计人类行为策略和Q函数,有对数似然:

   ![image-20250523010938474](https://JinChengYing.github.io/images/image-20250523010938474.png)

   可以估计策略和Q估计的泛化误差界(在数据集上求期望?)

   ![image-20250523011439075](https://JinChengYing.github.io/images/image-20250523011439075.png)

3. 为了恢复reward函数,我们为了简化表示??将model class都写成特征函数(独热向量表示,表征提取,表示学习)的线性组合的形式

   ![image-20250523012127799](https://JinChengYing.github.io/images/image-20250523012127799.png)

   线性表示下,MLE可以写成logistic回归

   ![image-20250523012625228](https://JinChengYing.github.io/images/image-20250523012625228.png)

    

   

4. 通过最小化Bellman均方误差(上一步学到的价值函数),恢复环境Reward函数,在线性假设下变成岭回归

   ![image-20250523012935709](https://JinChengYing.github.io/images/image-20250523012935709.png)

   ![image-20250523012942969](https://JinChengYing.github.io/images/image-20250523012942969.png)

   有闭形式解.

   ![image-20250516202622193](https://JinChengYing.github.io/images/image-20250516202622193.png)

5. 在正则性条件下估计对奖励函数估计的误差bound

   ![image-20250523013159418](https://JinChengYing.github.io/images/image-20250523013159418.png)

   一些研究证明了在可以探索到奖励的线性回归上,上述估计成立,另一些研究证明了在人类行为策略有着足够的覆盖(即包含了真实奖励函数)时,可以达到$O(n^{-1/2})$的收敛率,本文则证明了在没有充分覆盖的强假设下,仍然达到了次优的收敛率

6. 将上一步学到的Reward代入RL.通过悲观价值迭代得到近乎最优的策略,

   悲观惩罚通过迭代中更新的V(不是第一个算法的$\hat{V}$)得到,是V的不确定性量化器(如何计算:基于数据集D)

   ![image-20250523014141494](https://JinChengYing.github.io/images/image-20250523014141494.png)

   通过如下迭代,得到最优的策略(即人类更喜欢的策略,输出回复)

![image-20250516202641246](https://JinChengYing.github.io/images/image-20250516202641246.png)

V和Q的更新都进行悲观惩罚

策略使得可能的V最大

#### 次优Gap

1. 先进行线性MDP假设,

![image-20250523020058085](https://JinChengYing.github.io/images/image-20250523020058085.png)
存在参数集合对减法封闭,

建立了悲观惩罚(th3.5的结果提供保证):

![image-20250523020350108](https://JinChengYing.github.io/images/image-20250523020350108.png)

并且假设最优策略诱导的轨迹被数据集D覆盖

2. 单策略覆盖假设:特征的平方和大于平方的期望*n

   ![image-20250523020644809](https://JinChengYing.github.io/images/image-20250523020644809.png)

假设了算法1得到的人类行为策略覆盖

了最优策略(该假设弱于人类行为策略充分覆盖的强假设)

![image-20250523020839465](https://JinChengYing.github.io/images/image-20250523020839465.png)

jin等人对次优性gap的定义:到最优策略的距离

![image-20250523023348845](https://JinChengYing.github.io/images/image-20250523023348845.png)

#### 结论

1. 在线性MDP和再生核希尔伯特空间(RKHS)两种模型类中分别讨论了次最优性gap,估计了reward 函数的估计误差界限,次优性几乎可与之前的工作匹配.

2. 次最优性的前提是数据集能够有效覆盖最优策略诱导的轨迹.

   

### 细节

#### 线性MDP model假设:

1. 单策略覆盖,

   ![image-20250523015720052](https://JinChengYing.github.io/images/image-20250523015720052.png)

2. 避免恢复Q时的同一策略对应多种Q,规定了Q的唯一性.

   ![image-20250523011005318](https://JinChengYing.github.io/images/image-20250523011005318.png)

3. 正则性条件:$\theta$的界,$r$的线性结构,权重的范围,特征函数的界,覆盖数的界.



#### 再生核希尔伯特空间假设:

1. RKHS下的ridge regression ,由表示定理推出r的封闭形式解,进而建立不确定性量化器;
2. 对Bellman算子的封闭性假设,



1. 放在HOlder空间上呢?

2. 为什么讨论了RKHS上的模型

   从线性mdp到RKHS



