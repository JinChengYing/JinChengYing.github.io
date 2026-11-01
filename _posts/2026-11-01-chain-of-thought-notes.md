---
layout: post
title: "思维链（CoT）研究笔记"
description: "梳理零样本思维链、Auto-CoT 及相关提示方法。"
date: 2026-11-01
tags: [大语言模型, 思维链, 研究笔记]
post_workflow: v1
---

可以通过提示LLM生成推理步骤或从prompt提供的推理示例中学习推理过程

_提示：_

```
这组数中的奇数加起来是偶数：4、8、9、15、12、2、1。A：将所有奇数相加（9、15、1）得到25。答案为False。这组数中的奇数加起来是偶数：15、32、5、13、82、7、1。A：
```

_输出：_

```
将所有奇数相加（15、5、13、7、1）得到41。答案为False。
```

## zero shot COT

直接在prompt提示LLM逐步思考：
最近提出的一个新想法是 [零样本CoT](https://arxiv.org/abs/2205.11916)（ Kojima 等人，2022年），它基本上涉及将“让我们逐步思考”添加到原始提示中。让我们尝试一个简单的问题，看看模型的表现如何：

_提示：_

```
我去市场买了10个苹果。我给了邻居2个苹果和修理工2个苹果。然后我去买了5个苹果并吃了1个。我还剩下多少苹果？
```

_输出：_

```
11个苹果
```

答案是不正确的！现在让我们尝试使用特殊提示。

_提示：_

```
我去市场买了10个苹果。我给了邻居2个苹果和修理工2个苹果。然后我去买了5个苹果并吃了1个。我还剩下多少苹果？让我们逐步思考。
```

_输出：_

```
首先，您从10个苹果开始。您给了邻居和修理工各2个苹果，所以您还剩下6个苹果。然后您买了5个苹果，所以现在您有11个苹果。最后，您吃了1个苹果，所以您还剩下10个苹果。
```

## Auto-CoT自动思维链
目标: 消除人工次优提示 

<table>
<thead><tr><th>方法</th><th>基本做法</th><th>示例从哪里来</th></tr></thead>
<tbody>
<tr><td><strong>Zero-Shot-CoT</strong>，零样本思维链</td><td>在问题后加入“让我们一步一步思考”等提示</td><td>不提供例题</td></tr>
<tr><td><strong>Manual-CoT</strong>，人工思维链提示</td><td>先展示几道带解题过程的例题，再提出新问题</td><td>人工编写</td></tr>
<tr><td><strong>Auto-CoT</strong>，自动思维链</td><td>自动选取例题、生成解题过程，再组成提示</td><td>程序选题，模型生成解答</td></tr>
</tbody>
</table>
**环节：**
1. 准备目标任务相关的候选问题作为候选示例
2. 将问题编码为向量，聚类相似问题为一组
3. 为候选问题生成解题过程和答案，并且按照既定规则筛选
4. 将示例（自动构建的例题和解答+新问题）放入prompt

类似于上下文学习


## 推理即搜索
推理过程可以考虑为在诸多路径中进行搜索的问题，可能会有多个中间步骤，用什么结构表示推理过程可以提高推理精度和效率？  



## 局限性：
依赖详细多步推理示范，难以获得；没有标准的拆解复杂问题为多个子问题的办法； 中间推理步的误差会累积，容易影响结论。


## 相关工作梳理
<table>
<thead><tr><th>改动对象</th><th>方法</th><th>更准确的定位</th><th>参考文献</th></tr></thead>
<tbody>
<tr><td><strong>示例构建</strong></td><td><strong>Auto-CoT</strong></td><td>对候选问题聚类，选取具有多样性的代表问题，自动生成并筛选 CoT 示例。</td><td><strong>[1] Zhang et al., 2023，ICLR</strong>。(<a href="https://arxiv.org/pdf/2210.03493" title="https://arxiv.org/pdf/2210.03493">arxiv.org</a>)</td></tr>
<tr><td><strong>示例构建</strong></td><td><strong>Automate-CoT</strong></td><td>从带答案标签的数据生成推理链，依据标签筛选，再优化提示中的示例组合。</td><td><strong>[2] Shum et al., 2023，Findings of EMNLP</strong>。(<a href="https://aclanthology.org/2023.findings-emnlp.811/" title="Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data">ACL Anthology</a>)</td></tr>
<tr><td><strong>示例构建</strong></td><td><strong>Synthetic Prompting</strong></td><td>从少量人工示例出发，交替生成问题与解答，扩充并筛选 CoT 示例。</td><td><strong>[3] Shao et al., 2023，ICML</strong>。(<a href="https://proceedings.mlr.press/v202/shao23a.html" title="Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models">proceedings.mlr.press</a>)</td></tr>
<tr><td><strong>示例选择</strong></td><td><strong>Active-Prompt</strong></td><td>根据模型预测的不确定性选择值得标注的问题，再由人工编写其 CoT 示例。</td><td><strong>[4] Diao et al., 2024，ACL</strong>。(<a href="https://aclanthology.org/2024.acl-long.73/" title="Active Prompting with Chain-of-Thought for Large Language Models">ACL Anthology</a>)</td></tr>
<tr><td><strong>示例选择</strong></td><td><strong>Complexity-Based Prompting</strong></td><td>优先选择推理步骤较多的复杂示例；还将复杂度用于输出推理链的筛选。</td><td><strong>[5] Fu et al., 2023，ICLR</strong>。(<a href="https://openreview.net/pdf?id=yf1icZHC-l9" title="Published as a conference paper at ICLR 2023">openreview.net</a>)</td></tr>
<tr><td><strong>示例内容</strong></td><td><strong>Contrastive CoT</strong></td><td>同时提供明确区分的有效与无效推理示例，引导模型学习正确推导并避免错误模式。</td><td><strong>[6] Chia et al., 2023，arXiv</strong>。(<a href="https://arxiv.org/abs/2311.09277" title="Contrastive Chain-of-Thought Prompting">arxiv.org</a>)</td></tr>
<tr><td><strong>推导表达</strong></td><td><strong>Tab-CoT</strong></td><td>用表格组织中间推导，利用行与列的结构表达推理步骤及其关联。</td><td><strong>[7] Jin and Lu, 2023，Findings of ACL</strong>。(<a href="https://aclanthology.org/2023.findings-acl.651.pdf" title="https://aclanthology.org/2023.findings-acl.651.pdf">aclanthology.org</a>)</td></tr>
<tr><td><strong>输入模态</strong></td><td><strong>Multimodal-CoT</strong></td><td>融合图像与文本信息，采用“解释生成—答案预测”的两阶段模型框架。</td><td><strong>[8] Zhang et al., 2024，TMLR</strong>。(<a href="https://arxiv.org/pdf/2302.00923" title="https://arxiv.org/pdf/2302.00923">arxiv.org</a>)</td></tr>
</tbody>
</table>。
## 对应参考文献

**[1]** Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2023. **[Automatic Chain of Thought Prompting in Large Language Models](https://openreview.net/forum?id=5NTt8GFjUHkr)**. _The Eleventh International Conference on Learning Representations（ICLR 2023）_.

**[2]** Kashun Shum, Shizhe Diao, and Tong Zhang. 2023. **[Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data](https://aclanthology.org/2023.findings-emnlp.811/)**. _Findings of the Association for Computational Linguistics: EMNLP 2023_, pp. 12113–12139.

**[3]** Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. **[Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models](https://proceedings.mlr.press/v202/shao23a.html)**. _Proceedings of the 40th International Conference on Machine Learning（ICML 2023）_, PMLR 202, pp. 30706–30775.

**[4]** Shizhe Diao, Pengcheng Wang, Yong Lin, Rui Pan, Xiang Liu, and Tong Zhang. 2024. **[Active Prompting with Chain-of-Thought for Large Language Models](https://aclanthology.org/2024.acl-long.73/)**. _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics（Volume 1: Long Papers）_, pp. 1330–1350.

**[5]** Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. 2023. **[Complexity-Based Prompting for Multi-Step Reasoning](https://openreview.net/forum?id=yf1icZHC-l9)**. _The Eleventh International Conference on Learning Representations（ICLR 2023）_.

**[6]** Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, Soujanya Poria, and Lidong Bing. 2023. **[Contrastive Chain-of-Thought Prompting](https://arxiv.org/abs/2311.09277)**. _arXiv preprint_, arXiv:2311.09277.

**[7]** Ziqi Jin and Wei Lu. 2023. **[Tab-CoT: Zero-shot Tabular Chain of Thought](https://aclanthology.org/2023.findings-acl.651.pdf)**. _Findings of the Association for Computational Linguistics: ACL 2023_, pp. 10259–10277.

**[8]** Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 2024. **[Multimodal Chain-of-Thought Reasoning in Language Models](https://openreview.net/forum?id=y1pPWFVfvR)**. _Transactions on Machine Learning Research（TMLR）_.
