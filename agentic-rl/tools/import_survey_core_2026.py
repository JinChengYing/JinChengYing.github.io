"""Import the source-checked core additions from the 2026 survey checklist.

The source abstracts were read on 2026-10-27. This script is idempotent and
keeps the catalogue distinct from the wider, unverified candidate checklist.
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
          capabilities=["Planning", "Reasoning"], topics=["Model-Based RL"],
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

papers_path = DATA / "papers.json"
zh_path = DATA / "zh.json"
papers = json.loads(papers_path.read_text(encoding="utf-8"))
zh = json.loads(zh_path.read_text(encoding="utf-8"))
by_id = {item["id"]: item for item in papers}
added = 0
for record, zh_summary in ADDITIONS:
    if record["id"] not in by_id:
        papers.append(record)
        added += 1
    zh["papers"][record["id"]] = {"summary": zh_summary}

# The pasted checklist gave LATS the SWE-bench identifier. Its actual source is
# arXiv:2310.04406; upgrade the existing legacy record after reading the abstract.
lats = by_id["arxiv-2310.04406"]
lats.update({
    "title": "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models",
    "authors": ["Andy Zhou", "Kai Yan", "Michal Shlapentokh-Rothman", "Haohan Wang", "Yu-Xiong Wang"],
    "date": "2023-10-06", "venue": None, "publication_status": "preprint",
    "summary": "LATS combines language-model value estimates and self-reflection with Monte Carlo tree search and environment feedback.",
    "capabilities": ["Planning", "Reasoning"], "rl_topics": ["Exploration"],
    "tags": ["LATS", "Search", "Background"],
    "verification_status": "source_checked", "verified_at": TODAY,
})
zh["papers"][lats["id"]] = {
    "summary": "LATS 结合语言模型价值估计、自我反思、蒙特卡洛树搜索与环境反馈；属于规划背景工作。"
}

papers_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
zh_path.write_text(json.dumps(zh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Added {added} papers; reviewed LATS. Total: {len(papers)}")
