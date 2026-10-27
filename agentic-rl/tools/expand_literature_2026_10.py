"""One-time, source-reviewed literature batch for 27 October 2026."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
TODAY = "2026-10-27"

# Bibliography and descriptions were checked against each linked arXiv abstract.
# Dates are first submission dates; preprint status does not imply peer review.
ITEMS = [
    ("2503.12434", "A Survey on the Optimization of Large Language Model-based Agents", "2025-03-16", ["Shangheng Du", "Jiabao Zhao", "Jinxin Shi", "Zhentao Xie", "Xin Jiang", "Yanhong Bai", "Liang He"], "ACM Computing Surveys 58(9), Article 223 (2026)", "peer_reviewed", ["Planning", "Tool Use"], ["Policy Optimization", "Reward Modeling"], [], "Surveys parameter-driven and parameter-free agent optimization, including reinforcement learning, trajectory construction, reward design, prompting, and retrieval.", "综述智能体的参数更新与非参数优化，包括强化学习、轨迹构建、奖励设计、提示和检索。", ["Surveys"]),
    ("2601.12538", "A Survey of Agentic Reasoning for Large Language Models: Towards Recursively Self-Improving and Collective Agents", "2026-01-18", ["Tianxin Wei", "Ting-Wei Li", "Zhining Liu", "et al."], "Transactions on Machine Learning Research", "peer_reviewed", ["Reasoning", "Self-Evolution", "Multi-Agent Collaboration"], ["Policy Optimization", "Continual Learning"], [], "Organizes agentic reasoning into foundational, self-evolving, and collective layers, and distinguishes test-time orchestration from RL and supervised post-training.", "将智能体推理分为基础、自我演化与集体协作层，并区分测试时编排和强化学习等后训练。", ["Surveys"]),
    ("2602.06052", "A Survey of Agent Memory in the Second Half: Towards Self-Evolving and Long-Horizon Agents", "2026-01-14", ["Wei-Chieh Huang", "Weizhi Zhang", "Yueqing Liang", "et al."], "Transactions on Machine Learning Research", "peer_reviewed", ["Memory", "Self-Evolution"], ["Long-Horizon RL", "Continual Learning"], [], "Surveys memory substrates, cognitive mechanisms, and memory subjects; reviews learned memory-operation policies and evaluation of memory utility.", "从记忆载体、认知机制和记忆对象三轴梳理智能体记忆，并讨论可训练的记忆操作与效用评测。", ["Surveys"]),
    ("2608.06663", "The Horizon Gap: Planning, Memory, Execution, Training, and Evaluation for Long-Horizon LLM Agents", "2026-08-07", ["Mingguang Chen", "Licheng Wang", "Bo Qu"], None, "preprint", ["Planning", "Memory"], ["Long-Horizon RL", "Credit Assignment"], [], "Surveys long-horizon agents across planning, memory, execution, training, evaluation, and foundations, emphasizing the limits of outcome-only signals.", "从规划、记忆、执行、训练、评测与基础问题梳理长程智能体，指出只用最终结果反馈的局限。", ["Surveys"]),
    ("2602.16165", "HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents", "2026-02-18", ["Jiangweizhi Peng", "Yuanxin Liu", "Ruida Zhou", "Charles Fleming", "Zhaoran Wang", "Alfredo Garcia", "Mingyi Hong"], "ICML 2026", "peer_reviewed", ["Planning"], ["Credit Assignment", "Hierarchical RL", "Long-Horizon RL"], [], "Separates high-level subgoal planning from multi-step execution and estimates advantages at both levels.", "将高层子目标规划与多步执行分开，并在两个层级估计优势。", ["Credit Assignment", "Hierarchical"]),
    ("2603.08754", "Hindsight Credit Assignment for Long-Horizon LLM Agents", "2026-03-07", ["Hui-Ze Tan", "Xiao-Wen Yang", "Hao Chen", "Jie-Jing Shao", "Yi Wen", "Yuteng Shen", "Weihong Luo", "Xiku Du", "Lan-Zhe Guo", "Yu-Feng Li"], None, "preprint", ["Reasoning"], ["Credit Assignment", "Long-Horizon RL"], [], "HCAPO uses the LLM as a post-hoc critic to refine step-level Q-values and combines advantages at multiple scales.", "HCAPO 用语言模型充当事后评论器，细化逐步 Q 值，并结合多尺度优势。", ["Credit Assignment", "Hindsight"]),
    ("2605.21768", "Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents", "2026-05-20", ["Sikuan Yan", "Ahmed Bahloul", "Ercong Nie", "Susanna Schwarzmann", "Riccardo Trivisonno", "Volker Tresp", "Yunpu Ma"], None, "preprint", ["Memory"], ["Credit Assignment", "Long-Horizon RL", "Continual Learning"], [], "Compares local rerollouts from the same intermediate memory state while retaining a global long-horizon objective for memory operations.", "从相同中间记忆状态重新采样以公平比较记忆操作，同时保留全局长程目标。", ["Credit Assignment", "Memory"]),
    ("2608.08255", "Learning from Environmental Feedback: Credit Assignment across Multiple Timescales for Agentic Reinforcement Learning", "2026-08-08", ["Yifu Huo", "Shunjie Xing", "Chenglong Wang", "Peinan Feng", "Qiaozhi He", "Yan Ding", "Anxiang Ma", "Yuxin Gao", "Tongran Liu", "Tong Xiao", "Jingbo Zhu"], None, "preprint", ["Tool Use"], ["Credit Assignment", "Long-Horizon RL", "Environment Dynamics"], ["Web / GUI Agents"], "EFCA reweights returns using short-term action feedback and medium-term state history alongside long-term outcomes.", "EFCA 将短期动作反馈和中期状态历史与长期结果结合，用于重加权回报。", ["Credit Assignment", "Environmental Feedback"]),
    ("2609.12424", "Granularity-Adaptive Credit Assignment for Long-Horizon LLM Agent Reinforcement Learning", "2026-09-11", ["Taoran Liang", "Yang Liu", "Shang Luo", "et al."], None, "preprint", [], ["Credit Assignment", "Long-Horizon RL", "Efficient Training"], [], "GACA uses a decision criticality score from rollout token log-probabilities to adaptively mix episode- and step-level credit without extra rollouts.", "GACA 由采样轨迹的 token 对数概率估计决策关键性，自适应混合回合级和步骤级信用。", ["Credit Assignment", "Adaptive Granularity"]),
    ("2609.36178", "Targeting Pivotal Decisions for Credit Assignment in Agentic Reinforcement Learning", "2026-09-28", ["Dongwon Jung", "Hemanth Neelgund Ramesh", "Yifan Wang", "Xiaomin Li", "Yuexing Hao", "Yu Hu", "Muhao Chen", "Varun Chandrasekaran", "Andrzej Banburski-Fahey", "Jaron Lanier"], None, "preprint", [], ["Credit Assignment", "Long-Horizon RL"], ["Web / GUI Agents"], "ProVer selects potentially pivotal segments by comparing successful and failed trajectories, then estimates local advantage from sampled continuations before and after each segment.", "ProVer 对比成功与失败轨迹筛选关键片段，再用片段前后的采样续写估计局部优势。", ["Credit Assignment", "Segment-Level"]),
    ("2610.00838", "SHARPO: Segment-Level Credit Assignment for Agentic Reinforcement Learning", "2026-09-30", ["Xinchen Du", "Zhengze Zhou", "Wenhui Zhu", "Han Yu", "Sen Na", "Rohit Jain", "Alborz Geramifard"], None, "preprint", [], ["Credit Assignment", "Long-Horizon RL"], [], "SHARPO uses teacher-student log-probability gaps to reweight GRPO advantages at environment-facing segment granularity.", "SHARPO 利用师生模型的对数概率差，在环境交互片段粒度重加权 GRPO 优势。", ["Credit Assignment", "Segment-Level"]),
    ("2610.03372", "SCAD: Structured Credit Assignment and Distillation for Long-Horizon Agents", "2026-10-02", ["Shangyang Wu", "Shuai Zhao", "Ziyue Zhu", "Jinyang Wu", "Anh Tuan Luu", "Haoran Luo"], None, "preprint", ["Planning"], ["Credit Assignment", "Long-Horizon RL"], [], "SCAD separates planning from bounded subtask execution, combining cross-rollout prefix-tree planning credit with local teacher-guided execution distillation.", "SCAD 分离规划与限定子任务执行，用跨轨迹前缀树分配规划信用，并在局部上下文蒸馏执行。", ["Credit Assignment", "Distillation"]),
    ("2610.08402", "VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents", "2026-10-06", ["Jiaju Chen", "Min Yang", "Jinghua Piao", "Xiaochong Lan", "Xu Xia", "Xiangnan He", "Yong Li"], None, "preprint", [], ["Credit Assignment", "Multi-Turn RL", "Efficient Training"], [], "VETTA estimates turn- and token-level advantages with two heads on a shared lightweight critic for PPO updates.", "VETTA 用轻量共享评论器的两个头估计逐轮与逐 token 优势，协调 PPO 更新。", ["Credit Assignment", "Turn-Level", "Token-Level"]),
    ("2610.15840", "LLM Value Functions for Credit Assignment in Agentic RL", "2026-10-11", ["Kaijie Zhu", "Weixiang Yan", "Omar Attia", "Abbas Kazerouni", "Alexander Toshev", "Andrew Szot"], None, "preprint", [], ["Credit Assignment", "Long-Horizon RL", "Policy Optimization"], ["Code Agents"], "LVRL estimates policy-aware state values with a pretrained LLM, deriving step-level credit from value differences without fitting a separate critic.", "LVRL 用预训练语言模型估计策略相关状态价值，从价值差构造逐步信用，无需另训评论器。", ["Credit Assignment", "Value Function"]),
]


def main() -> None:
    paper_file = ROOT / "papers.json"
    zh_file = ROOT / "zh.json"
    papers = json.loads(paper_file.read_text(encoding="utf-8"))
    zh = json.loads(zh_file.read_text(encoding="utf-8"))
    existing = {paper["id"] for paper in papers}
    for arxiv_id, title, date, authors, venue, status, capabilities, topics, domains, summary, chinese, tags in ITEMS:
        paper_id = f"arxiv-{arxiv_id}"
        if paper_id in existing:
            raise ValueError(f"Already in library: {paper_id}")
        papers.append({
            "id": paper_id, "title": title, "authors": authors,
            "year": int(date[:4]), "date": date, "venue": venue,
            "publication_status": status, "arxiv_id": arxiv_id,
            "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
            "code_url": None, "project_url": None,
            "capabilities": capabilities, "rl_topics": topics, "domains": domains,
            "summary": summary, "key_contributions": [], "limitations": [],
            "open_questions": [], "tags": tags,
            "verification_status": "source_checked", "verified_at": TODAY,
        })
        zh["papers"][paper_id] = {"summary": chinese}
    paper_file.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zh_file.write_text(json.dumps(zh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
