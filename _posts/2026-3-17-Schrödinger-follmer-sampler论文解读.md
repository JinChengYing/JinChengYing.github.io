---
​---
layout: post
title: "Schrodinger follmer sampler论文解读"
date: 2026-3-17
tags: [生成模型,learning theory]
comments: true
author: Jincheng Ying
​---

---



## Schrödinger follmer sampler

### Schrödinger Bridge问题（SBP）

考虑在 $t=0$ 时刻观察到粒子分布为 $\tilde{\nu}$， 在 $t=1$ 时刻观察到粒子分布为 $\tilde{\mu}$

考虑标准布朗运动作为先验与参照， 有先验测度 $\mathbf{P} = \int \mathbf{W}_x dx$，$\mathbf{W}_x$ 是从 $x$ 点出发的维纳测度, 在任何时刻 $t$，它的边际分布都是勒贝格测度 $\mathscr{L}$.

考虑在路径空间 $\Omega = C([0, 1], \mathbb{R}^p)$ 中，每一个元素 $\omega$ 则是一条连续的路径，有投影算子$Z_t$满足

$$Z_t(\omega) = \omega_t$$

即给出路径$\omega$在t时刻的位置。 



如果 $\tilde{\nu}$ 到 $\tilde{\mu}$ 的演化不符合单纯的布朗运动规律（非自然扩散），那么在这两个分布之间，粒子最可能的路径概率分布 $\mathbf{Q}^*$ 是什么样的？

约束条件 $\mathbf{Q}_0 = \tilde{\nu}, \mathbf{Q}_1 = \tilde{\mu}$下， 有路径的概率测度 $\mathbf{Q}^*\in \mathcal{P}(\Omega)$, 并且t时刻的边际分布满足$Q^*_t=(Z_t)_{\#}\mathbf{Q}^*=\mathbf{Q}^* \circ Z_t^{-1}， t \in [0,1]$(其实是用原像定义t时刻粒子落在某个区域的概率：$\mathbf{Q}^*_t(A) = \mathbf{Q}^*(Z_t^{-1}(A))$)
$$
Q^* \in \arg\min \mathbb{D}_{\text{KL}}(\mathbf{Q} \| \mathbf{P}),
$$
其中 
$$
\mathbb{D}_{\text{KL}}(\mathbf{Q} \| \mathbf{P}) = \int \log\left(\frac{d\mathbf{Q}}{d\mathbf{P}}\right) d\mathbf{Q}
$$
根据Radon-Nikodym 定理，只有当 $\mathbf{Q} \ll \mathbf{P}$ 时，dQ/dP 才在 $\mathbf{P}$-几乎处处存在且是可测的。



SBP在上述边界约束条件下， 寻找一个与布朗运动在KL散度上最接近的路径概率测度， 目标是$\mathbf{Q}^*$ 和 $\mathbf{P}$ 必须在同一个路径空间簇内，相当于在先验的基础上做修正，尽可能的模仿标准布朗运动

### schrodinger follmer diffusion SDE

 

schrodinger follmer diffusion将t=0时的初始分布$\delta_0$传输到t=1时的目标分布$\mu$。
$$
f(x) = \frac{d\mu}{dN(0, \mathbf{I}_p)}(x), x \in \mathbb{R}^p.
$$
设热半群$Q_t$:
$$
Q_t f(x) = \mathbb{E}_{Z \sim N(0, \mathbf{I}_p)} \left[ f(x + \sqrt{t}Z) \right], \quad t \in [0, 1].
$$
Schr¨odinger-F¨ollmer diffusion process 定义为：
$$
dX_t = -\nabla_x U(X_t, t) dt + dB_t, X_0 = 0, t \in [0, 1]
$$
U是势能函数：
$$
U(x, t) = - \log \mathcal{Q}_{1-t} f(x).
$$
随机过程$\{X_t\}$是上面SDE的解，满足初始测度$\tilde{\nu}=\delta_0$(dirac测度)，$\tilde{\mu}=\mu$

SDE的漂移项可以写为：
$$
b(x, t) \equiv -\nabla_x U(x, t) = \frac{\mathbb{E}_Z[\nabla f(x + \sqrt{1 - tZ})]}{\mathbb{E}_Z[f(x + \sqrt{1 - tZ})]}, \quad x \in \mathbb{R}^p, t \in [0, 1],
$$
**C1 ：**

drift term b满足线性增长条件
$$
\Vert b(x,t) \Vert_2^2 \le C_0 (1 + \Vert x \Vert_2^2), x \in \mathbb{R}^p, t \in [0, 1]
$$
**C2：**

drift term b满足lipschitz条件：
$$
\Vert b(x, t) - b(y, t) \Vert_2 \leq C_1 \Vert x - y \Vert_2, \quad x, y \in \mathbb{R}^p, t \in [0, 1],
$$


 ![image-20260317200637442](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317200637442.png)

Proposition 2.1 说明当  C1和C2成立， 上述边值SDE有唯一strong solution。



 接下来只需要对C1和C2满足的sde采样就能得到想要的目标分布：

考虑Euler-Maruyama离散化

![image-20260317201104783](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317201104783.png)

其中，drift  term定义为
$$
b(Y_{t_k}, t_k) = \frac{\mathbb{E}_Z[\nabla f(Y_{t_k} + \sqrt{1 - t_k} Z)]}{\mathbb{E}_Z[f(Y_{t_k} + \sqrt{1 - t_k} Z)]}, Z \sim N(0, \mathbf{I}_p).
$$
实际运算中分子分母中的期望 需要对其进行蒙特卡洛估计， 这需要$f, \nabla f$是可以计算的：



目标分布可以写成 归一化的形式：
$$
\mu(x) = \frac{1}{C} \exp(-V(x)), x \in \mathbb{R}^p
$$
Radon-Nikodym导数 $ f$ 可以写为 

![image-20260317203845720](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317203845720.png)

于是漂移项可以改写为显式进行蒙特卡洛模拟计算的形式：

![image-20260317202754281](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317202754281.png)
$$
\tilde{b}_m(Y_{t_k}, t_k) = \frac{\frac{1}{m} \sum_{j=1}^{m} \left[ \nabla g\left(Y_{t_k} + \sqrt{1 - t_k} Z_j\right) \right]}{\frac{1}{m} \sum_{j=1}^{m} \left[ g\left(Y_{t_k} + \sqrt{1 - t_k} Z_j\right) \right]}, \quad k = 0, \dots, K - 1,
$$
或
$$
\tilde{b}_m(Y_{t_k}, t_k) = \frac{\frac{1}{m}\sum_{j=1}^m \left[Z_j g\left(Y_{t_k} + \sqrt{1 - t_k}Z_j\right)\right]}{\frac{1}{m}\sum_{j=1}^m \left[g\left(Y_{t_k} + \sqrt{1 - t_k}Z_j\right)\right] \cdot \sqrt{1 - t_k}}, \quad k = 0, \dots, K-1.
$$
于是就可以用如下方法采样：
$$
\tilde{Y}_{t_{k+1}} = \tilde{Y}_{t_k} + s\tilde{b}_m(\tilde{Y}_{t_k}, t_k) + \sqrt{s}\epsilon_{k+1}, k = 0, 1, \dots, K-1,
$$
伪代码如下：

![image-20260317203009432](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317203009432.png)

> [!TIP]
>
> **热半群：**是热方程的解算子：$u(x, t) = (P_t f)(x)$
>
> 可以表示为卷积， 满足半群的性质；扩散模型中，前向扩散过程（Forward Process）本质上就是通过热半群（或其变体，如 Ornstein-Uhlenbeck 半群）将复杂的原始数据分布逐渐转化为简单的正态分布。





### 理论性能保证

**C3假设：关于x lipschitz连续， 关于t holder连续**
$$
\Vert b(x,t) - b(y,s) \Vert_2 \le C_1 \left( \Vert x-y \Vert_2 + |t-s|^{1/2} \right), x, y \in \mathbb{R}^p \text{ and } t, s \in [0, 1],
$$
**定义衡量分布间距的W_d度量：**
$$
W_d(\nu_1, \nu_2) = \inf_{\nu \in \mathcal{D}(\nu_1, \nu_2)} \left( \iint_{\mathbb{R}^p \mathbb{R}^p} \|\theta_1 - \theta_2\|_2^d \,d\nu(\theta_1, \theta_2) \right)^{1/d}.
$$
这里只介绍不能直接计算采样只能monte carlo的算法2的性能保证

**C4假设： 势能U的强凸性假设**
$$
U(x,t) - U(y,t) - \nabla U(y,t)^{\mathrm{T}} (x-y) \geq (M/2) \left\| x - y \right\|_{2}^{2}
$$
并且$M< C_1  $ given in C3

下面给出性能保证：

![image-20260317203717591](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317203717591.png)

进一步可以有：

![image-20260317210504191](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317210504191.png)

#### 证明细节（th4.2-4.3）：

##### 需要的引理：

![image-20260317211452671](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317211452671.png)

![image-20260317211500701](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317211500701.png)

![image-20260317211514250](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317211514250.png)

![image-20260317211520598](C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317211520598.png)

##### th4.2的证明：

目标是bound住误差在K时的L2范数的上界

<img src="C:\Users\33702\AppData\Roaming\Typora\typora-user-images\image-20260317214205397.png" alt="image-20260317214205397" style="zoom: 50%;" />

先拆开成误差传播的形式
$$
\begin{align*}
\Delta_{k+1} &= \Delta_k + (X_{t_{k+1}} - X_{t_k}) - (\tilde{Y}_{t_{k+1}} - \tilde{Y}_{t_k}) \\
&= \Delta_k - s \left[ \tilde{b}_m (\tilde{Y}_{t_k}, t_k) - \tilde{b}_m (\tilde{Y}_{t_k} + \Delta_k, t_k) \right] + \int_{t_k}^{t_{k+1}} \left[ b(X_t, t) - \tilde{b}_m (X_{t_k}, t_k) \right] dt.
\end{align*}
$$
借助lemma A.3放缩第一项
$$
\begin{align*}
\left\Vert \Delta_k - s \left[ \tilde{b}_m(\tilde{Y}_{t_k}, t_k) - \tilde{b}_m(\tilde{Y}_{t_k} + \Delta_k, t_k) \right]\right\Vert_{L_2} \\
\leq \left\Vert \Delta_k - s \left[ b(\tilde{Y}_{t_k}, t_k) - b(\tilde{Y}_{t_k} + \Delta_k, t_k) \right]\right\Vert_{L_2} + s \left\Vert b(\tilde{Y}_{t_k}, t_k) - \tilde{b}_m(\tilde{Y}_{t_k}, t_k) \right\Vert_{L_2} \\
+ s \left\Vert \tilde{b}_m(\tilde{Y}_{t_k} + \Delta_k, t_k) - b(\tilde{Y}_{t_k} + \Delta_k, t_k) \right\Vert_{L_2} \\
\leq \rho \left\Vert \Delta_k \right\Vert_{L_2} + s \left\Vert b(\tilde{Y}_{t_k}, t_k) - \tilde{b}_m(\tilde{Y}_{t_k}, t_k) \right\Vert_{L_2} + s \left\Vert b(\tilde{Y}_{t_k} + \Delta_k, t_k) - \tilde{b}_m(\tilde{Y}_{t_k} + \Delta_k, t_k) \right\Vert_{L_2} .
\end{align*}
$$
借助 lemma A.6 得到上界
$$
\left\| \Delta_k - s \left[ \tilde{b}_m(\tilde{\mathbf{Y}}_{t_k}, t_k) - \tilde{b}_m(\tilde{\mathbf{Y}}_{t_k + \Delta_k}, t_k) \right] \right\|_{L_2} \leq \rho\|\Delta_k\|_{L_2} + s \cdot \mathcal{O}\left(\sqrt{\frac{p}{\log(m)}}\right).
$$
借助C3假设和 lemma A.2和A.6得到第二项的上界
$$
\left\| \int_{t_k}^{t_{k+1}} (b(X_t, t) - \tilde{b}_m(X_{t_k}, t_k)) dt \right\|_{L_2} \le \mathcal{O}(\sqrt{ps^{3/2}}) + \mathcal{O}\left(s\sqrt{\frac{p}{\log(m)}}\right).
$$
进而有误差传播不等式
$$
\Vert \Delta_{k+1} \Vert_{L_2} \leq \rho \Vert \Delta_k \Vert_{L_2} + \mathcal{O}(\sqrt{p s^{3/2}}) + \mathcal{O}\left(s \sqrt{\frac{p}{\log(m)}}\right).
$$
递推并假设t0时误差为0， 得到上界估计：
$$
\left\| \Delta_{k+1} \right\|_{L_2} \leq \rho^{k+1} \left\| \Delta_0 \right\|_{L_2} + \mathcal{O}(\sqrt{sp}) + \mathcal{O}\left(\sqrt{\frac{p}{\log(m)}}\right).
$$

$$
\begin{align*}
W_2(\operatorname{Law}(\tilde{Y}_{t_K}), \mu) &\leq \rho^K \|\Delta_0\|_{L_2} + \mathcal{O}(\sqrt{sp}) + \mathcal{O}\left(\sqrt{\frac{p}{\log(m)}}\right) \\
&\leq \mathcal{O}(\sqrt{sp}) + \mathcal{O}\left(\sqrt{\frac{p}{\log(m)}}\right).
\end{align*}
$$

可见证明需要引理中对于漂移项的误差进行估计，

