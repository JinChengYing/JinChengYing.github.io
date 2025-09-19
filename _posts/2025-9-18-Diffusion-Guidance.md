---
layout: post
title: "Diffusion Guidance"
date:   2025-9-18
tags: [扩散模型,深度学习]
comments: true
author: Jincheng Ying
---

### **Diffusion Guidance Is a Controllable Policy Improvement Operator**

##### 背景:

 利用生成学习＋强化学习的方法去学习超出数据的performance的策略,得到可扩展的RL算法框架 .

##### 关注的问题:  

1. scaling up RL algorithms 扩展应用范围,问题规模与维度

2.  训练出的性能超出样本数据所能给出性能的极限

   

##### 局限性: 

1. 在扩展RL算法方面, 生成建模被证明是可扩展的, 能否迁移到RL上?
2. 传统的从数据集学习策略的behavioral cloning 方法尽管简单, 并且可以结合扩散与flow-matching 生成model ,但只能达到样本数据中的最优,无法更进一步
3. 运用迭代的RL算法可以取得更优的策略提升, 但受到超参数的灵敏度和不稳定性的影响, 难以扩展到更大的任务.

##### Motivation:

1. 结合迭代RL和behavioral cloning 的优势开发框架.



##### 工作:

1. 推导了策略提升与扩散模型引导之间的关系,

2. 提出了CFGRL框架,可以用简单的监督学习训练,无需显式学习价值函数, 提升了数据上训练出的策略性能

3. 增加guidance的权重导致性能提升

   

##### 优势:

1. 无需显式学习价值函数
2. CFGRL框架,可以用简单的监督学习训练(goal-conditioned behavioral cloning)
3. 策略的改进程度可以在test时控制.



#### 方法:

##### outline:

先从数据中通过behavioral cloning得到初始策略, 并假设为reference policy, 并且将后续的策略假设为两个因子的内积,即reference policy  和一个最优性的分布, 当这个最优性的分布与优势函数成正比,说明该内积生成的策略是改进的.



##### 详细分步:

1. 先参数化策略为内积

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918123914715.png" alt="image-20250918123914715" style="zoom:50%;" />

​    一方面 当f对于A来说非负且单调增的,则上述内积对于初始的reference policy是提升的, 另一方面,可以对f加幂次实现策略的提升, 但是会导致过度调整,偏离reference policy

![image-20250918125320507](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918125320507.png)

![image-20250918125327304](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918125327304.png)

上面说明存在一种权衡:即最大化回报和不偏离reference policy, 并且可以用KL的正则化目标理解这种权衡.

![image-20250918125602625](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918125602625.png)

2. diffusion guidance通过得分函数学习内积 策略

   在二分类<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918133837781.png" alt="image-20250918133837781" style="zoom: 33%;" />的情况下考虑策略, 其似然可以通过f写为<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918133859606.png" alt="image-20250918133859606" style="zoom:50%;" />

   Z是所有动作的积分,是正则化项(概率化处理后,得分函数可以考虑成两个分布的得分函数相加),因此(5)的内积策略等价于<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918134015815.png" alt="image-20250918134015815" style="zoom:50%;" />

   扩散模型 通过学习score function 学到上面内积策略的分布, 注意这里存在的可加性

   <img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918134147859.png" alt="image-20250918134147859" style="zoom:50%;" />

   利用贝叶斯公式,则 optimality-conditioned policy的得分函数为

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918171459253.png" alt="image-20250918171459253" style="zoom:50%;" />

3. 在策略提升方面, diffusion guidance 控制最优性的衰减,即可以控制f的幂次,去权衡偏离度和策略更新速度.

   得到的新得分函数为:<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918134604676.png" alt="image-20250918134604676" style="zoom:67%;" />

只需要在test时用一个神经网络控制w即可权衡.

也就是在训练时只需要训练reference 策略, 在采样时控制w 参与采样,得到提升的策略





3. 用CFGRL训练与采样(采用flow matching)

![image-20250918123259845](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918123259845.png)



<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918172604925.png" alt="image-20250918172604925" style="zoom:50%;" />

对状态动作对数据集加噪, 学习基于reference的score function的速度场函数,再逆向从正态分布结合权重w采样出提升后策略的分布,o=1保证了策略的提升

训练目标如下

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918162448281.png" alt="image-20250918162448281" style="zoom:80%;" />





#### CFGRL改进了离线RL中对加权策略的提取

传统的advantage-weighted regression (AWR)方法<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918170717680.png" alt="image-20250918170717680" style="zoom:50%;" />

损失容易集中在少数异常的状态动作对,导致权重集中异常,实际利用数据很少.

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918170829874.png" alt="image-20250918170829874" style="zoom:50%;" />

CFGRL解决了这个问题. 因为AWR的温度参数与CFGRL的reference的权重相似,AWR的温度参数必须提前制定,用在整体训练目标里. 但CFGRL中,注意到reference 策略和optimality-conditioned policy是分离的,只在采样时结合用于更新速度场进行生成. 更方便进行权衡



此外,CFGRL可以改进goal-conditioned behavioral cloning(GCBC)得到的策略,无需改动GCBC的假设





### Policy-Guided Diffusion

#### 背景: 从behavioral policy 收集的离线数据集里进行 学习,降低成本

#### 局限性:

1. distribution shift导致了out-of-sample,即数据没有覆盖足够多的状态动作对,coverage不够充分,从而导致了对未见状态动作对的高估,过度乐观(这也是为啥悲观估计也是离线RLHF研究的一个路子)

   

#### **已有方法**：bias 和coverage难以权衡 

1. model free 方法对目标策略进行正则化(penalizing value estimates in uncertain states) 或者朝着behavior 策略正则化,但会牺牲目标政策性能以达到稳定
2.  本文focus on 的生成模型以增强数据集并且减少Out of sample issue 的方法, 先前该路线的研究使用的是model based 的方法,单步的world model与目标策略 进行rollout 生成合成的 on-policy的训练experience.但是, 这样会导致compound error, 截断会导致交互链条不足,不足以完成覆盖.

针对难以权衡的问题, 一个思路是生成未见过的经验,实现在分布外样本的优化, 提高了样本覆盖率,但不能保证合成经验轨迹的最优性,因此需要保守的off policy技术去学习(依赖于已有的行为策略去做价值评估). 可以想象成盲人摸象, 只是在摸索, 通过探索摸索出分布外的情况, 而直接学习到目标分布,可以保证知道分布(大象)的轮廓,再由分布生成高质量的轨迹,并且保证了轨迹质量不断提升(如何量化轨迹质量:在测试集上得到了更高的return,并且return的方差更低)

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250919115235398.png" alt="image-20250919115235398" style="zoom:67%;" />

#### 工作:

1. 提出了policy-guided diffusion (PGD),通过生成整条轨迹避免 compounding error.

2. 步骤:

    (1)  在离线数据集上训练扩散模型

    (2)  基于behavior policy 采样合成轨迹 ,解决了稀疏数据的难题

    (3)  启发于分类器引导的扩散模型. 使用目标策略作为guidance ,逐步更新轨迹, 生成一个正规的目标分布 称为behavior-regularized target distribution,这确保动作不会偏离行为策略太远,限制了泛化误差.避免了compounding error.

   

   

#### 详细内容:

1. ##### 自回归生成 与直接生成 实现离线数据的onpolicy sampling

   方法的有效性取决于参数化轨迹分布的方式

   **对自回归生成:<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918215809726.png" alt="image-20250918215809726" style="zoom: 50%;" />**

   训练一个单步转移model<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918215041569.png" alt="image-20250918215041569" style="zoom: 50%;" />

    从离线数据集初始化一个状态$s_0$,从目标策略迭代采样动作，并使用所学的动力学模型近似环境转移

由于compounding error ,所以限制智能体的步数小于等于k.

而从在数据集中任意时间t采样获得k个步长的轨迹片段,都可以去近似如下的子轨迹分布

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918215229689.png" alt="image-20250918215229689" style="zoom:50%;" />

使用如下函数族去近似上面的分布(离线分布的p替代了目标分布p)

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918215526853.png" alt="image-20250918215526853" style="zoom:50%;" />

条件子轨迹分布可以写成

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918215447822.png" alt="image-20250918215447822" style="zoom:50%;" />



![image-20250918215927708](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918215927708.png)

当从该分布生成轨迹时，\(p_{\text{target}}(s_t)\)与\(p_{\text{off}}(s_t)\)之间的差异会使推演的起始偏向行为策略所访问的状态。此外，我们仍需要k较小以避免复合误差。综合来看，从离线数据集采样会将合成推演 “锚定” 到离线数据集中的状态，而截断推演又阻止合成轨迹远离这一锚点

**我的理解是,状态转移和p_off(s在数据集中的频率)是从离线数据集中学出来的,p_target是状态转移和p_off生成的,由于k较小,最终还会倾向于在有限步之内,围绕着p_off**

这种方法导致了对于behavior的偏差以及 无法解决out of sample problem



**直接生成:$p_{off}(\tau)$** 



<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918215835736.png" alt="image-20250918215835736" style="zoom:50%;" />

使用了重要性采样:<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918221101883.png" alt="image-20250918221101883" style="zoom:50%;" />



直接对行为分布$p_{off}(τ ; θ)$来参数化实现对目标轨迹分布的参数化,可直接生成完整轨迹(**利用transition和p(s_0)乘积为离线数据集轨迹概率,消去了对于transition 的学习需要,避免了compounding error,transition 隐含在了离线的轨迹里面了**),也存在计算重要性采样权重需访问行为策略\(\pi_{\text{off}}(a|s)\)，且多个重要性权重的乘积易导致高方差问题等不足。







2. #####  Policy-Guided Diffusion

基于直接生成的思路, 训练学习轨迹分布的扩散模型.

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918212500811.png" alt="image-20250918212500811" style="zoom:50%;" />

目标分布怎么来的?

**能直接学习行为分布下的噪声条件得分函数** \(\nabla_{\hat{\tau}} \log p_{\text{off}}(\hat{\tau}; \sigma)\), 所以是先建模行为分布下的score function 再通过下面的公式近似得到噪声条件下的得分函数\(\nabla_{\hat{\tau}} \log p_{\text{target}}(\hat{\tau}; \sigma)\)

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918223032296.png" alt="image-20250918223032296" style="zoom:50%;" />

噪声趋于0时,可以用<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918223413119.png" alt="image-20250918223413119" style="zoom: 50%;" />替代<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918223434054.png" alt="image-20250918223434054" style="zoom:50%;" />上式可重新逼近为:

![image-20250918223446789](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918223446789.png)

形式如下:

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250918223653527.png" alt="image-20250918223653527" style="zoom: 67%;" />





在guidance term里加入guidance coefficients是分类器guidance diffusion的标准技术, 应用在PGD里增强guidance有如下形式: 

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250919110616448.png" alt="image-20250919110616448" style="zoom: 67%;" />

实际是加上了guidance coefficients $\lambda$作为控制目标轨迹分布的幂次, 取对数后作为乘积系数,起到插值的作用, 控制了朝向目标策略的引导强度.

当噪声趋近于0, 即接近目标分布时,采样的分布可以变换为:
<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250919110843356.png" alt="image-20250919110843356" style="zoom:67%;" />

![image-20250919111057115](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20250919111057115.png)



如上图, 左侧是目标分布, 右侧是采样分布, 随着控制$\lambda$增大,采样分布朝着目标策略似然高的区域转变,即从红星到蓝星区域的分布变化.



 