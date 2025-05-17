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



### 思想方法与结论

#### **核心思想**:

在离线RL中,agent无法直接在数据集中观测到环境中的奖励,但 数据集的(s,a,s')反映了人类在环境奖励影响下的状态与偏好动作,因此本文通过这种内涵的信息,借助Dynamic Discrete Choice (DDC)建模人类被环境驯化后的行为策略,从该策略中恢复了环境的奖励函数,再套到RL中去学习在奖励下的最优策略,因此最终次优性结论的一个重要条件是数据集中的轨迹能够覆盖最优策略.

#### **核心方法(DCPPO)**:

1. 通过极大似然估计(MLE)估计人类行为策略和Q函数

2. 通过最小化Bellman均方误差(上一步学到的价值函数),恢复环境Reward函数

   ![image-20250516202622193](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250516202622193.png)

3. 将上一步学到的Reward代入RL.通过悲观价值迭代得到近乎最优的策略

![image-20250516202641246](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250516202641246.png)

#### 结论

1. 在线性MDP和再生核希尔伯特空间(RKHS)两种模型类中分别讨论了次最优性gap,估计了reward 函数的估计误差界限.
2. 次最优性的前提是数据集能够有效覆盖最优策略诱导的轨迹.

### 细节

#### 线性MDP model假设:

1. 单策略覆盖,
2. 避免恢复Q时的同一策略对应多种Q,规定了Q的唯一性.
3. 正则性条件:$\theta$的界,$r$的线性结构,权重的范围,特征函数的界,覆盖数的界.



#### 再生核希尔伯特空间假设:

1. RKHS下的ridge regression ,由表示定理推出r的封闭形式解,进而建立不确定性量化器;
2. 对Bellman算子的封闭性假设,