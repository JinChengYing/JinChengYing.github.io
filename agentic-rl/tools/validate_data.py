"""Validate the static Agentic RL Atlas datasets (offline, no network calls)."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1] / "data"
errors: list[str] = []

def read(name: str):
    with (ROOT / name).open(encoding="utf-8") as handle:
        return json.load(handle)

def fail(message: str):
    errors.append(message)

def valid_url(value, where):
    if value is None:
        return
    if not isinstance(value, str) or urlparse(value).scheme not in {"http", "https"} or not urlparse(value).netloc:
        fail(f"{where}: invalid URL")

taxonomy = read("taxonomy.json")
papers = read("papers.json")
benchmarks = read("benchmarks.json")
frameworks = read("frameworks.json")
candidates = read("candidates.json")
problems = read("open-problems.json")
zh = read("zh.json")
ids: set[str] = set()
canonical: set[str] = set()
for index, paper in enumerate(papers):
    where = f"papers[{index}]"
    required = {"id", "title", "authors", "year", "date", "venue", "publication_status", "arxiv_id",
                "paper_url", "code_url", "project_url", "capabilities", "rl_topics", "domains", "summary",
                "key_contributions", "limitations", "open_questions", "tags", "verification_status", "verified_at"}
    missing = required - paper.keys()
    if missing:
        fail(f"{where}: missing {sorted(missing)}")
        continue
    if not paper["id"] or paper["id"] in ids:
        fail(f"{where}: missing or duplicate id")
    ids.add(paper["id"])
    if not isinstance(paper["title"], str) or not paper["title"].strip():
        fail(f"{where}: missing title")
    if not isinstance(paper["year"], int) or not 1900 <= paper["year"] <= 2100:
        fail(f"{where}: invalid year")
    for key in ("authors","capabilities","rl_topics","domains","key_contributions","limitations","open_questions","tags"):
        if not isinstance(paper[key], list):
            fail(f"{where}: {key} must be a list")
    for key in ("capabilities","rl_topics","domains"):
        for topic in paper[key]:
            if topic not in taxonomy[key]:
                fail(f"{where}: unknown {key} {topic!r}")
    for key in ("paper_url","code_url","project_url"):
        valid_url(paper[key], f"{where}.{key}")
    if not paper["paper_url"] and not paper["code_url"] and not paper["project_url"]:
        fail(f"{where}: no traceable source")
    if paper["verification_status"] not in {"source_checked","unverified"}:
        fail(f"{where}: unknown verification status")
    if paper["verification_status"] == "source_checked" and (not paper["verified_at"] or not paper["summary"]):
        fail(f"{where}: checked entry requires date and summary")
    if paper["date"] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", paper["date"]):
        fail(f"{where}: invalid date")
    key = paper["arxiv_id"] or (paper["paper_url"] or "").rstrip("/")
    if key in canonical:
        fail(f"{where}: duplicate canonical source {key}")
    canonical.add(key)
for index, item in enumerate(benchmarks):
    where = f"benchmarks[{index}]"
    if not item.get("id") or not item.get("name") or not item.get("source_url"):
        fail(f"{where}: id, name and source_url required")
    valid_url(item.get("source_url"), f"{where}.source_url")
    valid_url(item.get("code_url"), f"{where}.code_url")
    valid_url(item.get("huggingface_url"), f"{where}.huggingface_url")
    if item.get("huggingface_url") and urlparse(item["huggingface_url"]).hostname != "huggingface.co":
        fail(f"{where}.huggingface_url: expected huggingface.co")
for index, item in enumerate(frameworks):
    if not item.get("id") or not item.get("source_url"):
        fail(f"frameworks[{index}]: id and source required")
    valid_url(item.get("source_url"), f"frameworks[{index}].source_url")
    valid_url(item.get("code_url"), f"frameworks[{index}].code_url")
for kind in ("papers", "benchmarks", "frameworks"):
    if kind not in candidates or not isinstance(candidates[kind], list):
        fail(f"candidates.{kind}: missing list")
        continue
    for index, group in enumerate(candidates[kind]):
        where = f"candidates.{kind}[{index}]"
        if not group.get("area_en") or not group.get("area_zh"):
            fail(f"{where}: bilingual area labels required")
        names = group.get("names")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name.strip() for name in names):
            fail(f"{where}: nonempty candidate names required")
        elif len(names) != len(set(names)):
            fail(f"{where}: duplicate candidate name")
benchmark_ids = {item["id"] for item in benchmarks}
for index, problem in enumerate(problems):
    where = f"open-problems[{index}]"
    for paper_id in problem["related_paper_ids"]:
        if paper_id not in ids:
            fail(f"{where}: unknown paper id {paper_id}")
    for benchmark_id in problem["suggested_benchmark_ids"]:
        if benchmark_id not in benchmark_ids:
            fail(f"{where}: unknown benchmark id {benchmark_id}")
for key, values in taxonomy.items():
    for value in values:
        if value not in zh["taxonomy"]:
            fail(f"zh.taxonomy: missing {value}")
for item in benchmarks:
    if item["id"] not in zh["benchmarks"]:
        fail(f"zh.benchmarks: missing {item['id']}")
for item in problems:
    translated = zh["problems"].get(item["id"])
    if not translated:
        fail(f"zh.problems: missing {item['id']}")
        continue
    for key in ("title", "definition", "motivation", "existing_limitations", "possible_approaches", "risks"):
        if not translated.get(key):
            fail(f"zh.problems.{item['id']}: missing {key}")
for item in papers:
    if item["verification_status"] == "source_checked" and not zh["papers"].get(item["id"], {}).get("summary"):
        fail(f"zh.papers: missing checked summary for {item['id']}")
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f"Valid: {len(papers)} papers, {len(frameworks)} frameworks, {len(benchmarks)} benchmarks, {len(problems)} problems.")

