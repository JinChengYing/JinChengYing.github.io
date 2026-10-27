"""Import source-checked papers from the supplied 2026 literature outline.

The source abstracts were read on 2026-10-27. This script is idempotent and
keeps the paper catalogue distinct from source-less candidate names.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
TODAY = "2026-10-27"

def paper(arxiv_id, title, authors, year, date, venue, status, summary, zh_summary,
          capabilities=(), topics=(), domains=(), tags=()):
    return {
        "id": f"arxiv-{arxiv_id}", "title": title, "authors": list(authors),
        "year": year, "date": date, "venue": venue,
        "publication_status": status, "arxiv_id": arxiv_id,
        "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
        "code_url": None, "project_url": None,
        "capabilities": list(capabilities), "rl_topics": list(topics),
        "domains": list(domains), "summary": summary,
        "key_contributions": [], "limitations": [], "open_questions": [],
        "tags": list(tags), "verification_status": "source_checked",
        "verified_at": TODAY,
    }, zh_summary

ADDITIONS = [
    paper("1707.06347", "Proximal Policy Optimization Algorithms",
          ["John Schulman", "Filip Wolski", "Prafulla Dhariwal", "Alec Radford", "Oleg Klimov"],
          2017, "2017-07-20", None, "preprint",
          "Introduces PPO, an online policy-gradient family that reuses sampled data for multiple minibatch optimization epochs.",
          "提出 PPO 策略梯度方法，可对交互采样数据进行多轮小批量优化；属于基础 RL 算法。",
          topics=["Policy Optimization"], tags=["Background", "PPO"]),
    paper("2305.18290", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
          ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"],
          2023, "2023-05-29", None, "preprint",
          "Derives a direct classification objective for preference alignment without online sampling during fine-tuning.",
          "将偏好对齐写成直接分类目标，微调时无需在线采样；是偏好优化背景文献。",
          topics=["Policy Optimization"], tags=["Background", "DPO", "Preference Optimization"]),
    paper("2305.14992", "Reasoning with Language Model is Planning with World Model",
          ["Shibo Hao", "Yi Gu", "Haodi Ma", "Joshua Jiahua Hong", "Zhen Wang", "Daisy Zhe Wang", "Zhiting Hu"],
          2023, "2023-05-24", "EMNLP 2023", "peer_reviewed",
          "RAP uses an LLM as both a reasoning agent and an approximate world model, with tree search guiding planning.",
          "RAP 将语言模型同时用作推理智能体和近似世界模型，通过树搜索引导规划。",
          capabilities=["Planning", "Reasoning"],
          tags=["Background", "RAP", "Search"]),
    paper("2308.11432", "A Survey on Large Language Model based Autonomous Agents",
          ["Lei Wang", "Chen Ma", "Xueyang Feng", "Zeyu Zhang", "Hao Yang", "Jingsen Zhang",
           "Zhiyuan Chen", "Jiakai Tang", "Xu Chen", "Yankai Lin", "Wayne Xin Zhao", "Zhewei Wei", "Ji-Rong Wen"],
          2023, "2023-08-22", None, "preprint",
          "Surveys LLM-based autonomous-agent construction, applications and evaluation; it is background for agent architecture rather than an RL method.",
          "梳理大语言模型智能体的构建、应用和评估；用于理解智能体架构，不属于强化学习训练方法。",
          tags=["Surveys", "Background"]),
    paper("2402.03300", "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
          ["Zhihong Shao", "Peiyi Wang", "Qihao Zhu", "Runxin Xu", "Junxiao Song", "Xiao Bi",
           "Haowei Zhang", "Mingchuan Zhang", "Y.K. Li", "Y. Wu", "Daya Guo"],
          2024, "2024-02-05", None, "preprint",
          "Introduces DeepSeekMath and Group Relative Policy Optimization for mathematical reasoning.",
          "提出 DeepSeekMath 与用于数学推理的组相对策略优化（GRPO）；作为算法基础文献收录。",
          capabilities=["Reasoning"], topics=["Policy Optimization"],
          domains=["Mathematical Reasoning"], tags=["Background", "GRPO"]),
]

budgetmem = {
    "id": "pmlr-zhang26ic",
    "title": "Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory",
    "authors": ["Haozhen Zhang", "Haodong Yue", "Tao Feng", "Quanyu Long", "Jianzhu Bao",
                "Bowen Jin", "Weizhi Zhang", "Xiao Li", "Jiaxuan You", "Chengwei Qin", "Wenya Wang"],
    "year": 2026, "date": None, "venue": "ICML 2026 (PMLR 306)",
    "publication_status": "peer_reviewed", "arxiv_id": None,
    "paper_url": "https://proceedings.mlr.press/v306/zhang26ic.html",
    "code_url": "https://github.com/ViktorAxelsen/BudgetMem", "project_url": None,
    "capabilities": ["Memory"], "rl_topics": ["Policy Optimization", "Efficient Training"],
    "domains": [],
    "summary": "BudgetMem trains a compact routing policy to choose among memory-processing budget tiers according to the query and cost target.",
    "key_contributions": [], "limitations": [], "open_questions": [],
    "tags": ["BudgetMem", "Memory Policy", "NEW"],
    "verification_status": "source_checked", "verified_at": TODAY,
}
ADDITIONS.append((budgetmem, "BudgetMem 训练轻量路由策略，按查询与成本目标选择记忆处理模块的预算档位。"))

# Source-linked works in the supplied outline, checked against their primary
# arXiv abstracts. Empty author arrays avoid publishing incomplete long lists.
ADDITIONS.extend([
    paper("2501.12948", "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
          [], 2025, "2025-01-22", None, "preprint",
          "Uses reinforcement learning to elicit reasoning patterns on verifiable tasks; background for reasoning RL rather than evidence of external agent interaction.",
          "通过强化学习在可验证任务中激发推理模式；作为推理 RL 背景，不据此认定存在外部智能体环境交互。",
          capabilities=["Reasoning"], topics=["Policy Optimization"], domains=["Mathematical Reasoning"], tags=["Background", "RLVR"]),
    paper("2503.14476", "DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
          [], 2025, "2025-03-18", None, "preprint",
          "Introduces DAPO and an open training system for large-scale reasoning-oriented LLM reinforcement learning.",
          "提出 DAPO 算法与开放训练系统，用于大规模推理型语言模型强化学习。",
          capabilities=["Reasoning"], topics=["Policy Optimization", "Efficient Training"], tags=["DAPO", "Background"]),
    paper("2310.08560", "MemGPT: Towards LLMs as Operating Systems",
          ["Charles Packer", "Sarah Wooders", "Kevin Lin", "Vivian Fang", "Shishir G. Patil", "Ion Stoica", "Joseph E. Gonzalez"],
          2023, "2023-10-12", None, "preprint",
          "Introduces virtual context management across memory tiers for document analysis and multi-session chat; it is a memory-system background work.",
          "通过记忆层级间的数据移动管理有限上下文，用于文档分析和跨会话对话；属于记忆机制背景工作。",
          capabilities=["Memory"], tags=["Background", "Memory"]),
    paper("2405.14831", "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models",
          ["Bernal Jiménez Gutiérrez", "Yiheng Shu", "Yu Gu", "Michihiro Yasunaga", "Yu Su"],
          2024, "2024-05-23", None, "preprint",
          "Uses knowledge graphs and Personalized PageRank for long-term retrieval over accumulated information; not an RL-trained memory policy.",
          "结合知识图谱与 Personalized PageRank 检索积累的信息；不应视为经 RL 训练的记忆策略。",
          capabilities=["Memory"], tags=["Background", "Retrieval"]),
    paper("2303.17651", "Self-Refine: Iterative Refinement with Self-Feedback",
          [], 2023, "2023-03-30", None, "preprint",
          "Iteratively generates, critiques and revises outputs with one LLM, without reinforcement learning or additional training.",
          "用同一个模型反复生成、反馈和修订输出；原论文明确不需要强化学习或额外训练。",
          capabilities=["Self-Evolution", "Reasoning"], tags=["Background", "Self-Correction"]),
    paper("2305.11738", "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing",
          ["Zhibin Gou", "Zhihong Shao", "Yeyun Gong", "Yelong Shen", "Yujiu Yang", "Nan Duan", "Weizhu Chen"],
          2023, "2023-05-19", None, "preprint",
          "Uses external tools to validate and revise an LLM's initial output through iterative critiquing.",
          "借助外部工具核查模型初始输出，并通过迭代批评与修订完成自我纠错。",
          capabilities=["Self-Evolution", "Tool Use"], tags=["Background", "Self-Correction"]),
    paper("2505.03335", "Absolute Zero: Reinforced Self-play Reasoning with Zero Data",
          [], 2025, "2025-05-06", None, "preprint",
          "Trains a model to propose and solve code-reasoning tasks, using a code executor for verifiable self-play rewards.",
          "让模型自行提出并求解代码推理任务，使用代码执行器提供可验证的自博弈奖励。",
          capabilities=["Self-Evolution", "Reasoning"], topics=["Policy Optimization", "Exploration"],
          domains=["Code Agents", "Mathematical Reasoning"], tags=["Self-Training", "RLVR"]),
    paper("1706.02275", "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
          ["Ryan Lowe", "Yi Wu", "Aviv Tamar", "Jean Harb", "Pieter Abbeel", "Igor Mordatch"],
          2017, "2017-06-07", None, "preprint",
          "Adapts actor-critic learning to multi-agent settings by accounting for other agents' policies; classical MARL background.",
          "在 actor-critic 学习中考虑其他智能体策略，以处理多智能体协作与竞争；属于传统 MARL 背景。",
          capabilities=["Multi-Agent Collaboration"], topics=["Policy Optimization"], domains=["Multi-Agent Systems"],
          tags=["Background", "MADDPG"]),
    paper("1705.08926", "Counterfactual Multi-Agent Policy Gradients",
          ["Jakob Foerster", "Gregory Farquhar", "Triantafyllos Afouras", "Nantas Nardelli", "Shimon Whiteson"],
          2017, "2017-05-24", None, "preprint",
          "COMA uses a centralized critic and a counterfactual baseline to assign credit to individual agents in cooperative MARL.",
          "COMA 使用集中式 critic 和反事实基线，为合作式 MARL 中各智能体分配信用。",
          capabilities=["Multi-Agent Collaboration"], topics=["Credit Assignment"], domains=["Multi-Agent Systems"],
          tags=["Background", "COMA"]),
    paper("1803.11485", "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning",
          ["Tabish Rashid", "Mikayel Samvelyan", "Christian Schroeder de Witt", "Gregory Farquhar", "Jakob Foerster", "Shimon Whiteson"],
          2018, "2018-03-30", None, "preprint",
          "Combines per-agent values into a monotonic joint action-value for centralized training and decentralized execution.",
          "通过单调混合个体价值估计联合动作价值，支持集中训练与分散执行。",
          capabilities=["Multi-Agent Collaboration"], topics=["Policy Optimization"], domains=["Multi-Agent Systems"],
          tags=["Background", "QMIX"]),
    paper("2103.01955", "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games",
          ["Chao Yu", "Akash Velu", "Eugene Vinitsky", "Jiaxuan Gao", "Yu Wang", "Alexandre Bayen", "Yi Wu"],
          2021, "2021-03-02", None, "preprint",
          "Studies PPO-based methods on cooperative multi-agent benchmarks and reports strong performance with modest tuning.",
          "在合作式多智能体基准中研究 PPO 类方法，作者报告其在较少调参下有较强表现。",
          capabilities=["Multi-Agent Collaboration"], topics=["Policy Optimization"], domains=["Multi-Agent Systems"],
          tags=["Background", "MAPPO"]),
    paper("2207.01206", "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents",
          ["Shunyu Yao", "Howard Chen", "John Yang", "Karthik Narasimhan"],
          2022, "2022-07-04", None, "preprint",
          "Introduces a simulated shopping website where agents navigate pages and act on product-search instructions.",
          "提出模拟购物网站环境，智能体需根据商品需求在网页间导航并执行购买相关动作。",
          capabilities=["Tool Use"], domains=["Web / GUI Agents"], tags=["Benchmark", "WebShop"]),
    paper("2307.13854", "WebArena: A Realistic Web Environment for Building Autonomous Agents",
          [], 2023, "2023-07-25", None, "preprint",
          "Introduces reproducible functional websites and tasks for evaluating web agents' task completion.",
          "构建可复现的功能性网站和任务，用于评估网页智能体的任务完成能力。",
          capabilities=["Tool Use"], domains=["Web / GUI Agents"], tags=["Benchmark", "WebArena"]),
    paper("2306.06070", "Mind2Web: Towards a Generalist Agent for the Web",
          ["Xiang Deng", "Yu Gu", "Boyuan Zheng", "Shijie Chen", "Samuel Stevens", "Boshi Wang", "Huan Sun", "Yu Su"],
          2023, "2023-06-09", None, "preprint",
          "Provides web-instruction tasks and action sequences collected across diverse real websites for generalist web-agent evaluation.",
          "提供跨真实网站的网页指令任务与动作序列，用于通用网页智能体评估。",
          capabilities=["Tool Use"], domains=["Web / GUI Agents"], tags=["Benchmark", "Mind2Web"]),
    paper("2401.13649", "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks",
          [], 2024, "2024-01-24", None, "preprint",
          "Introduces visually grounded web tasks for evaluating multimodal agents that process image and text inputs.",
          "提出视觉定位的网页任务，评估同时处理图像和文本输入的多模态智能体。",
          capabilities=["Perception / Multimodal", "Tool Use"], domains=["Web / GUI Agents"],
          tags=["Benchmark", "VisualWebArena"]),
    paper("2410.07095", "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering",
          [], 2024, "2024-10-09", None, "preprint",
          "Curates machine-learning engineering competition tasks to evaluate agents that prepare data, train models and run experiments.",
          "整理机器学习工程竞赛任务，评估智能体准备数据、训练模型和运行实验的能力。",
          domains=["Code Agents", "Scientific Agents"], tags=["Benchmark", "MLE-bench"]),
    paper("2311.12983", "GAIA: a benchmark for General AI Assistants",
          ["Grégoire Mialon", "Clémentine Fourrier", "Craig Swift", "Thomas Wolf", "Yann LeCun", "Thomas Scialom"],
          2023, "2023-11-21", None, "preprint",
          "Introduces assistant questions combining reasoning, multimodal handling, web browsing and tool use.",
          "提出结合推理、多模态处理、网页浏览与工具使用的通用助理问题集。",
          capabilities=["Reasoning", "Tool Use"], domains=["Search and Deep Research Agents"],
          tags=["Benchmark", "GAIA"]),
    paper("2310.06770", "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
          ["Carlos E. Jimenez", "John Yang", "Alexander Wettig", "Shunyu Yao", "Kexin Pei", "Ofir Press", "Karthik Narasimhan"],
          2023, "2023-10-10", "ICLR 2024", "peer_reviewed",
          "Introduces software-engineering issues from real GitHub repositories for evaluating repository-level code edits.",
          "从真实 GitHub 仓库收集软件工程问题，评估模型修改代码库以解决 issue 的能力。",
          domains=["Code Agents"], tags=["Benchmark", "SWE-bench"]),
])

papers_path = DATA / "papers.json"
zh_path = DATA / "zh.json"
papers = json.loads(papers_path.read_text(encoding="utf-8"))
zh = json.loads(zh_path.read_text(encoding="utf-8"))
by_id = {item["id"]: item for item in papers}
added = 0
for record, zh_summary in ADDITIONS:
    if record["id"] not in by_id:
        papers.append(record)
        by_id[record["id"]] = record
        added += 1
    zh["papers"][record["id"]] = {"summary": zh_summary}
by_id["arxiv-2305.14992"]["rl_topics"] = []

# The pasted checklist gave LATS the SWE-bench identifier. Its actual source is
# arXiv:2310.04406; upgrade the existing legacy record after reading the abstract.
lats = by_id["arxiv-2310.04406"]
lats.update({
    "title": "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models",
    "authors": ["Andy Zhou", "Kai Yan", "Michal Shlapentokh-Rothman", "Haohan Wang", "Yu-Xiong Wang"],
    "date": "2023-10-06", "venue": None, "publication_status": "preprint",
    "summary": "LATS combines language-model value estimates and self-reflection with Monte Carlo tree search and environment feedback.",
    "capabilities": ["Planning", "Reasoning"], "rl_topics": [],
    "tags": ["LATS", "Search", "Background"],
    "verification_status": "source_checked", "verified_at": TODAY,
})
zh["papers"][lats["id"]] = {
    "summary": "LATS 结合语言模型价值估计、自我反思、蒙特卡洛树搜索与环境反馈；属于规划背景工作。"
}

UPDATES = {
    "arxiv-2302.04761": (
        ["Timo Schick", "Jane Dwivedi-Yu", "Roberto Dessì", "Roberta Raileanu", "Maria Lomeli",
         "Luke Zettlemoyer", "Nicola Cancedda", "Thomas Scialom"], "2023-02-09",
        "Trains a model with self-supervised API-call examples to decide when and how to use external tools.",
        "通过自监督 API 调用示例训练模型决定何时及如何使用外部工具；不是在线 Agentic RL。",
        ["Tool Use"], [], ["Background", "Toolformer"]),
    "arxiv-2307.16789": (
        [], "2023-07-31",
        "Introduces ToolLLM and ToolBench, covering tool-use data construction, model training and evaluation over real APIs.",
        "提出 ToolLLM 与 ToolBench，覆盖真实 API 场景下的工具使用数据构建、训练和评估。",
        ["Tool Use"], [], ["Background", "ToolBench"]),
    "arxiv-2303.11366": (
        [], "2023-03-20",
        "Uses verbal reflection and an episodic text buffer to improve later trials without updating model weights.",
        "借助语言反思和情节文本缓冲区改善后续尝试，不更新模型权重。",
        ["Self-Evolution", "Memory"], [], ["Background", "Reflexion"]),
    "arxiv-2305.16291": (
        ["Guanzhi Wang", "Yuqi Xie", "Yunfan Jiang", "Ajay Mandlekar", "Chaowei Xiao", "Yuke Zhu", "Linxi Fan", "Anima Anandkumar"],
        "2023-05-25",
        "Uses an automatic curriculum, executable skill library and iterative prompting for open-ended Minecraft exploration, without parameter fine-tuning.",
        "通过自动课程、可执行技能库和迭代提示在 Minecraft 中自主探索，不进行模型参数微调。",
        ["Self-Evolution", "Planning"], ["Embodied Agents"], ["Background", "Voyager"]),
}
for paper_id, (authors, date, summary, zh_summary, capabilities, domains, tags) in UPDATES.items():
    record = by_id[paper_id]
    record.update({"authors": authors, "date": date, "summary": summary,
                   "capabilities": capabilities, "domains": domains, "tags": tags,
                   "verification_status": "source_checked", "verified_at": TODAY})
    zh["papers"][paper_id] = {"summary": zh_summary}

papers_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
zh_path.write_text(json.dumps(zh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Added {added} papers; reviewed LATS. Total: {len(papers)}")
