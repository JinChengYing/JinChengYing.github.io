"""Add abstract-checked agentic RL papers to the existing bilingual library.

Primary arXiv abstracts were read on 2026-10-27. Keep this import idempotent.
Descriptions report the proposed method, not independently reproduced results.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
CHECKED_AT = "2026-10-27"


def entry(arxiv_id, title, authors, date, summary, zh_summary, *,
          capabilities=(), topics=(), domains=(), tags=(), venue=None,
          code_url=None):
    record = {
        "id": f"arxiv-{arxiv_id}", "title": title, "authors": list(authors),
        "year": int(date[:4]), "date": date, "venue": venue,
        "publication_status": "peer_reviewed" if venue else "preprint",
        "arxiv_id": arxiv_id, "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
        "code_url": code_url, "project_url": None,
        "capabilities": list(capabilities), "rl_topics": list(topics),
        "domains": list(domains), "summary": summary,
        "key_contributions": [], "limitations": [], "open_questions": [],
        "tags": list(tags), "verification_status": "source_checked",
        "verified_at": CHECKED_AT,
    }
    return record, zh_summary


ADDITIONS = [
    entry("2408.07199", "Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents",
          ["Pranav Putta", "Edmund Mills", "Naman Garg", "Sumeet Motwani", "Chelsea Finn", "Divyansh Garg", "Rafael Rafailov"],
          "2024-08-13",
          "Combines guided tree search, self-critique and off-policy preference optimization to learn from successful and failed web-agent trajectories.",
          "结合引导式树搜索、自我批评和离策略偏好优化，从网页智能体成功与失败的轨迹中学习。",
          capabilities=["Planning", "Self-Evolution"], topics=["Policy Optimization", "Exploration"],
          domains=["Web / GUI Agents"], tags=["Agent Q", "MCTS"]),
    entry("2406.11896", "DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning",
          ["Hao Bai", "Yifei Zhou", "Mert Cemri", "Jiayi Pan", "Alane Suhr", "Sergey Levine", "Aviral Kumar"],
          "2024-06-14",
          "Trains a vision-language device-control policy with offline RL followed by offline-to-online RL in an Android environment.",
          "先用离线强化学习初始化视觉语言设备控制策略，再在 Android 环境中进行离线到在线强化学习。",
          capabilities=["Perception / Multimodal"], topics=["Policy Optimization", "Environment Dynamics"],
          domains=["Web / GUI Agents"], tags=["DigiRL", "Android"]),
    entry("2502.18449", "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution",
          ["Yuxiang Wei", "Olivier Duchenne", "Jade Copet", "Quentin Carbonneaux", "Lingming Zhang", "Daniel Fried", "Gabriel Synnaeve", "Rishabh Singh", "Sida I. Wang"],
          "2025-02-25",
          "Uses rule-based rewards over open-source software evolution data to train a code-reasoning model for engineering tasks.",
          "利用开源软件演化数据和规则式奖励训练代码推理模型，面向真实软件工程任务。",
          capabilities=["Reasoning"], topics=["Policy Optimization", "Reward Modeling"],
          domains=["Code Agents"], tags=["SWE-RL"], venue="NeurIPS 2025"),
    entry("2502.10325", "Process Reward Models for LLM Agents: Practical Framework and Directions",
          ["Sanjiban Choudhury"], "2025-02-14",
          "Introduces AgentPRM, which estimates process rewards from agent rollouts, and InversePRM, which learns them from demonstrations.",
          "提出 AgentPRM：从智能体交互轨迹估计过程奖励；并提出从示范学习过程奖励的 InversePRM。",
          capabilities=["Planning"], topics=["Reward Modeling", "Credit Assignment"],
          domains=["Embodied Agents"], tags=["AgentPRM", "InversePRM"],
          code_url="https://github.com/sanjibanc/agent_prm"),
    entry("2503.09516", "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning",
          ["Bowen Jin", "Hansi Zeng", "Zhenrui Yue", "Jinsung Yoon", "Sercan Arik", "Dong Wang", "Hamed Zamani", "Jiawei Han"],
          "2025-03-12",
          "Trains multi-turn query generation during reasoning with live retrieval, outcome rewards and masking of retrieved tokens.",
          "在多轮推理中训练模型生成搜索查询，利用结果奖励与检索文本 token 屏蔽稳定训练。",
          capabilities=["Tool Use", "Reasoning"], topics=["Policy Optimization", "Multi-Turn RL"],
          domains=["Search and Deep Research Agents"], tags=["Search-R1"],
          code_url="https://github.com/PeterGriffinJin/Search-R1"),
    entry("2504.03160", "DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments",
          ["Yuxiang Zheng", "Dayuan Fu", "Xiangkun Hu", "Xiaojie Cai", "Lyumanshan Ye", "Pengrui Lu", "Pengfei Liu"],
          "2025-04-04",
          "Trains research agents end to end with reinforcement learning over authentic web-search interactions rather than a fixed retrieval corpus.",
          "在真实网页搜索交互中端到端训练研究智能体，研究开放网页环境而非固定检索语料。",
          capabilities=["Tool Use", "Planning"], topics=["Long-Horizon RL", "Environment Dynamics"],
          domains=["Search and Deep Research Agents"], tags=["DeepResearcher"],
          code_url="https://github.com/GAIR-NLP/DeepResearcher"),
    entry("2504.11536", "ReTool: Reinforcement Learning for Strategic Tool Use in LLMs",
          ["Jiazhan Feng", "Shijue Huang", "Xingwei Qu", "Ge Zhang", "Yujia Qin", "Baoquan Zhong", "Chengquan Jiang", "Jinxin Chi", "Wanjun Zhong"],
          "2025-04-15",
          "Uses outcome-driven RL over multi-turn code execution to teach a reasoning model when and how to call a code interpreter.",
          "让推理模型在多轮代码执行轨迹上通过结果奖励学习何时、如何调用代码解释器。",
          capabilities=["Tool Use", "Reasoning"], topics=["Policy Optimization", "Multi-Turn RL"],
          domains=["Mathematical Reasoning"], tags=["ReTool", "Code Interpreter"]),
    entry("2505.01441", "Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning",
          ["Joykirat Singh", "Raghav Magazine", "Yash Pandya", "Akshay Nambi"],
          "2025-04-28",
          "ARTIST uses outcome-based RL to train tool choice and invocation inside multi-turn reasoning without step-level supervision.",
          "ARTIST 用结果奖励训练多轮推理中的工具选择与调用，不依赖逐步监督标签。",
          capabilities=["Tool Use", "Reasoning"], topics=["Policy Optimization", "Multi-Turn RL"],
          domains=["Mathematical Reasoning"], tags=["ARTIST"]),
    entry("2505.04588", "ZeroSearch: Incentivize the Search Capability of LLMs without Searching",
          ["Hao Sun", "Zile Qiao", "Jiayan Guo", "Xuanbo Fan", "Yingyan Hou", "Yong Jiang", "Pengjun Xie", "Yan Zhang", "Fei Huang", "Jingren Zhou"],
          "2025-05-07",
          "Replaces live search calls during RL training with a learned retrieval simulator and a curriculum of increasingly noisy documents.",
          "用学习得到的检索模拟器替代训练时的实时搜索，并逐步增加返回文档的噪声。",
          capabilities=["Tool Use", "Reasoning"], topics=["Efficient Training", "Environment Dynamics"],
          domains=["Search and Deep Research Agents"], tags=["ZeroSearch", "Search Simulation"]),
    entry("2505.16421", "WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning",
          ["Zhepei Wei", "Wenlin Yao", "Yao Liu", "Weizhi Zhang", "Qin Lu", "Liang Qiu", "Changlong Yu", "Puyang Xu", "Chao Zhang", "Bing Yin", "Hyokun Yun", "Lihong Li"],
          "2025-05-22",
          "Trains web agents with asynchronous online trajectories and binary task-success rewards across multi-turn interactions.",
          "通过异步采集网页交互轨迹、按任务是否成功给出二值奖励，端到端训练多轮网页智能体。",
          capabilities=["Tool Use", "Planning"], topics=["Multi-Turn RL", "Long-Horizon RL"],
          domains=["Web / GUI Agents"], tags=["WebAgent-R1"], venue="EMNLP 2025"),
    entry("2505.22648", "WebDancer: Towards Autonomous Information Seeking Agency",
          ["Jialong Wu", "Baixuan Li", "Runnan Fang", "Wenbiao Yin", "Liwen Zhang", "Zhengwei Tao", "Dingchu Zhang", "Zekun Xi", "Gang Fu", "Yong Jiang", "Pengjun Xie", "Fei Huang", "Jingren Zhou"],
          "2025-05-28",
          "Builds an information-seeking web agent through browsing data, trajectory sampling, supervised cold start and reinforcement learning.",
          "通过浏览数据构建、轨迹采样、监督冷启动和强化学习四个阶段训练信息搜索智能体。",
          capabilities=["Tool Use", "Planning"], topics=["Policy Optimization", "Long-Horizon RL"],
          domains=["Search and Deep Research Agents"], tags=["WebDancer"]),
    entry("2507.02592", "WebSailor: Navigating Super-human Reasoning for Web Agent",
          ["Kuan Li", "Zhongwang Zhang", "Huifeng Yin", "Liwen Zhang", "Litu Ou", "Jialong Wu", "Wenbiao Yin", "Baixuan Li", "Zhengwei Tao", "Xinyu Wang", "Weizhou Shen", "Junkai Zhang", "Dingchu Zhang", "Xixi Wu", "Yong Jiang", "Ming Yan", "Pengjun Xie", "Fei Huang", "Jingren Zhou"],
          "2025-07-03",
          "Combines high-uncertainty task construction, a supervised cold start and DUPO reinforcement learning for complex web information seeking.",
          "结合高不确定性任务构建、监督冷启动与 DUPO 强化学习，训练复杂网页信息检索能力。",
          capabilities=["Tool Use", "Planning"], topics=["Policy Optimization", "Exploration"],
          domains=["Search and Deep Research Agents"], tags=["WebSailor", "DUPO"]),
    entry("2508.19828", "Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning",
          ["Sikuan Yan", "Xiufeng Yang", "Zuchao Huang", "Ercong Nie", "Zifeng Ding", "Zonggen Li", "Xiaowen Ma", "Jinhe Bi", "Kristian Kersting", "Jeff Z. Pan", "Hinrich Schütze", "Volker Tresp", "Yunpu Ma"],
          "2025-08-27",
          "Trains a memory manager for add, update, delete and no-op actions alongside an answer agent that selects relevant memories.",
          "分别训练执行增、改、删或不操作的记忆管理器，以及挑选相关记忆作答的智能体。",
          capabilities=["Memory"], topics=["Policy Optimization", "Long-Horizon RL"],
          tags=["Memory-R1", "Memory Management"]),
]

papers_path = DATA / "papers.json"
zh_path = DATA / "zh.json"
candidates_path = DATA / "candidates.json"
papers = json.loads(papers_path.read_text(encoding="utf-8"))
zh = json.loads(zh_path.read_text(encoding="utf-8"))
candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
seen_ids = {item["id"] for item in papers}
seen_sources = {item.get("arxiv_id") for item in papers}
added = 0
for record, zh_summary in ADDITIONS:
    if record["id"] not in seen_ids and record["arxiv_id"] not in seen_sources:
        papers.append(record)
        seen_ids.add(record["id"])
        seen_sources.add(record["arxiv_id"])
        added += 1
    zh["papers"][record["id"]] = {"summary": zh_summary}

promoted = {tag.casefold() for record, _ in ADDITIONS for tag in record["tags"]}
for group in candidates["papers"]:
    group["names"] = [name for name in group["names"] if name.casefold() not in promoted]
candidates["papers"] = [group for group in candidates["papers"] if group["names"]]

papers_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
zh_path.write_text(json.dumps(zh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
candidates_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Added {added} papers; total {len(papers)}")
