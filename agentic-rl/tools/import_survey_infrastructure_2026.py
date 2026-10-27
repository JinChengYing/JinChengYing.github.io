"""Merge source-linked benchmark and framework records from the 2026 outline.

The links were checked against primary paper abstracts, official project
repositories, or official documentation on 2026-10-27. Unknown benchmark
properties stay null; a source link alone is not evidence of RL trainability.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
TODAY = "2026-10-27"

BENCHMARKS = [
    ({"id": "mind2web", "name": "Mind2Web", "environment_type": "Web action dataset",
      "modality": "HTML and web actions", "task_horizon": None, "stateful": None,
      "evaluation": "Web action prediction", "source_url": "https://arxiv.org/abs/2306.06070",
      "code_url": None, "verification_status": "source_checked", "verified_at": TODAY},
     {"environment_type": "网页动作数据集", "modality": "HTML 与网页动作", "evaluation": "网页动作预测"}),
    ({"id": "visualwebarena", "name": "VisualWebArena", "environment_type": "Visual web tasks",
      "modality": "Image and text web interaction", "task_horizon": None, "stateful": None,
      "evaluation": "Task completion", "source_url": "https://arxiv.org/abs/2401.13649",
      "code_url": None, "verification_status": "source_checked", "verified_at": TODAY},
     {"environment_type": "视觉网页任务", "modality": "图文网页交互", "evaluation": "任务完成"}),
    ({"id": "androidworld", "name": "AndroidWorld", "environment_type": "Android app environment",
      "modality": "Mobile GUI interaction", "task_horizon": None, "stateful": None,
      "evaluation": None, "source_url": "https://github.com/google-research/android_world",
      "code_url": "https://github.com/google-research/android_world",
      "verification_status": "source_link_only"},
     {"environment_type": "Android 应用环境", "modality": "移动端 GUI 交互"}),
    ({"id": "swegym", "name": "SWE-Gym", "environment_type": "Software engineering training environment",
      "modality": "Code and repository interaction", "task_horizon": None, "stateful": None,
      "evaluation": None, "source_url": "https://github.com/SWE-Gym/SWE-Gym",
      "code_url": "https://github.com/SWE-Gym/SWE-Gym", "verification_status": "source_link_only"},
     {"environment_type": "软件工程训练环境", "modality": "代码与仓库交互"}),
    ({"id": "textworld", "name": "TextWorld", "environment_type": "Text-game environment",
      "modality": "Text actions", "task_horizon": None, "stateful": None,
      "evaluation": None, "source_url": "https://github.com/microsoft/TextWorld",
      "code_url": "https://github.com/microsoft/TextWorld", "verification_status": "source_link_only"},
     {"environment_type": "文本游戏环境", "modality": "文本动作"}),
    ({"id": "mlebench", "name": "MLE-bench", "environment_type": "Machine learning engineering benchmark",
      "modality": "Code, data and experiments", "task_horizon": None, "stateful": None,
      "evaluation": "Competition-style task evaluation", "source_url": "https://arxiv.org/abs/2410.07095",
      "code_url": None, "verification_status": "source_checked", "verified_at": TODAY},
     {"environment_type": "机器学习工程基准", "modality": "代码、数据与实验", "evaluation": "竞赛式任务评测"}),
]

FRAMEWORKS = [
    ({"id": "openrlhf", "title": "OpenRLHF", "year": None,
      "description": "Open-source framework for scalable RLHF and agentic RL training.",
      "source_url": "https://github.com/OpenRLHF/OpenRLHF", "code_url": "https://github.com/OpenRLHF/OpenRLHF",
      "verification_status": "source_checked", "verified_at": TODAY, "layer": "Trainer"},
     {"description": "用于可扩展 RLHF 与 Agentic RL 训练的开源框架。", "layer": "训练器"}),
    ({"id": "trl", "title": "TRL", "year": None,
      "description": "Hugging Face library for transformer post-training with reinforcement and preference-learning methods.",
      "source_url": "https://github.com/huggingface/trl", "code_url": "https://github.com/huggingface/trl",
      "verification_status": "source_checked", "verified_at": TODAY, "layer": "Trainer"},
     {"description": "Hugging Face 的 Transformer 强化学习与偏好学习训练库。", "layer": "训练器"}),
    ({"id": "trlx", "title": "trlX", "year": None,
      "description": "Distributed language-model reinforcement learning from human feedback toolkit.",
      "source_url": "https://github.com/CarperAI/trlx", "code_url": "https://github.com/CarperAI/trlx",
      "verification_status": "source_checked", "verified_at": TODAY, "layer": "Trainer"},
     {"description": "用于语言模型 RLHF 的分布式训练工具。", "layer": "训练器"}),
    ({"id": "rllib", "title": "Ray RLlib", "year": None,
      "description": "General reinforcement-learning library in Ray; an infrastructure baseline rather than an agent-specific trainer.",
      "source_url": "https://docs.ray.io/en/latest/rllib/", "code_url": None,
      "verification_status": "source_checked", "verified_at": TODAY, "layer": "General RL library"},
     {"description": "Ray 的通用强化学习库，可作基建对照，并非 Agentic RL 专用训练器。", "layer": "通用 RL 库"}),
    ({"id": "tianshou", "title": "Tianshou", "year": None,
      "description": "PyTorch deep reinforcement-learning library.",
      "source_url": "https://github.com/thu-ml/tianshou", "code_url": "https://github.com/thu-ml/tianshou",
      "verification_status": "source_checked", "verified_at": TODAY, "layer": "General RL library"},
     {"description": "基于 PyTorch 的通用深度强化学习库。", "layer": "通用 RL 库"}),
    ({"id": "stable-baselines3", "title": "Stable-Baselines3", "year": None,
      "description": "PyTorch implementations of common reinforcement-learning algorithms.",
      "source_url": "https://github.com/DLR-RM/stable-baselines3", "code_url": "https://github.com/DLR-RM/stable-baselines3",
      "verification_status": "source_checked", "verified_at": TODAY, "layer": "General RL library"},
     {"description": "常用强化学习算法的 PyTorch 实现库。", "layer": "通用 RL 库"}),
]

def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))

def write(name, value):
    (DATA / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

benchmarks, frameworks, zh = read("benchmarks.json"), read("frameworks.json"), read("zh.json")
existing_b = {item["id"] for item in benchmarks}
existing_f = {item["id"] for item in frameworks}
for item, translation in BENCHMARKS:
    if item["id"] not in existing_b:
        benchmarks.append(item)
    zh["benchmarks"][item["id"]] = translation
for item, translation in FRAMEWORKS:
    if item["id"] not in existing_f:
        frameworks.append(item)
    zh["frameworks"][item["id"]] = translation
write("benchmarks.json", benchmarks)
write("frameworks.json", frameworks)
write("zh.json", zh)
print(f"Benchmarks: {len(benchmarks)}; frameworks: {len(frameworks)}")
