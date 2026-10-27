#!/usr/bin/env python3
"""Small, dependency-free draft and publishing workflow for this Jekyll blog."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
POST_NAME = re.compile(r"\d{4}-\d{2}-\d{2}-(.+)\.md\Z")


def fail(message):
    raise ValueError(message)


def front_matter(path):
    text = path.read_text(encoding="utf-8-sig")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", text, re.S)
    if not match:
        fail(f"{path}: 缺少完整的 YAML front matter")
    return match.group(1), match.group(2)


def field(header, name):
    match = re.search(rf"(?m)^{re.escape(name)}:\s*(.*)$", header)
    return match.group(1).strip() if match else ""


def new(args):
    slug = args.slug.lower().strip()
    if not SLUG.fullmatch(slug):
        fail("slug 只能使用小写英文字母、数字和连字符，例如 policy-gradient-notes")
    drafts = ROOT / "_drafts"
    drafts.mkdir(exist_ok=True)
    target = drafts / f"{slug}.md"
    if target.exists():
        fail(f"草稿已存在：{target}")
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    header = "\n".join([
        "---", "layout: post", f"title: {json.dumps(args.title, ensure_ascii=False)}",
        f"description: {json.dumps(args.summary, ensure_ascii=False)}",
        f"tags: {json.dumps(tags, ensure_ascii=False)}", "post_workflow: v1", "---", "",
    ])
    target.write_text(header + "<!-- 在这里写正文，发布前删除本行 -->\n", encoding="utf-8")
    print(f"草稿已创建：{target}")


def publish(args):
    drafts = (ROOT / "_drafts").resolve()
    source = (drafts / args.draft).resolve()
    if source.parent != drafts or source.suffix != ".md" or not source.is_file():
        fail("请传入 _drafts 中的 Markdown 文件名，例如 policy-gradient-notes.md")
    if not SLUG.fullmatch(source.stem):
        fail("草稿文件名必须是英文 slug")
    header, body = front_matter(source)
    if field(header, "post_workflow") != "v1" or not field(header, "title"):
        fail("草稿缺少 title 或 post_workflow: v1")
    if not body.strip() or "<!-- 在这里写正文" in body:
        fail("请先写好正文并删除模板占位文字")
    date = datetime.now().astimezone().date() if not args.date else datetime.strptime(args.date, "%Y-%m-%d").date()
    posts = ROOT / "_posts"
    posts.mkdir(exist_ok=True)
    target = posts / f"{date.isoformat()}-{source.stem}.md"
    if target.exists():
        fail(f"同名文章已存在：{target}")
    clean = re.sub(r"(?m)^date:.*\n", "", header)
    target.write_text(f"---\n{clean}\ndate: {date.isoformat()}\n---\n{body}", encoding="utf-8")
    source.unlink()
    print(f"文章已发布到：{target}")
    print("检查后提交：python scripts/posts.py check && git add _posts _drafts && git commit -m \"Publish post\" && git push")


def check(_args):
    count = 0
    errors = []
    for folder in (ROOT / "_drafts", ROOT / "_posts"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.md")):
            try:
                header, body = front_matter(path)
                if field(header, "post_workflow") != "v1":
                    continue  # Existing posts predate this helper.
                if not field(header, "title") or field(header, "layout") != "post":
                    fail(f"{path}: title/layout 无效")
                if folder.name == "_posts":
                    match = POST_NAME.fullmatch(path.name)
                    if not match or not SLUG.fullmatch(match.group(1)):
                        fail(f"{path}: 文件名应为 YYYY-MM-DD-english-slug.md")
                    if field(header, "date") != path.name[:10]:
                        fail(f"{path}: front matter 日期与文件名不一致")
                    if not body.strip() or "<!-- 在这里写正文" in body:
                        fail(f"{path}: 正文为空或仍含占位文字")
                count += 1
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        fail("\n".join(errors))
    print(f"检查通过：{count} 篇由发布工具管理的文章/草稿")


def main():
    parser = argparse.ArgumentParser(description="博客草稿与发布工具")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("new", help="创建草稿")
    create.add_argument("--title", required=True)
    create.add_argument("--slug", required=True, help="用于网址的小写英文名")
    create.add_argument("--summary", default="")
    create.add_argument("--tags", default="", help="逗号分隔，例如 RL,阅读笔记")
    create.set_defaults(run=new)
    release = sub.add_parser("publish", help="将草稿移至 _posts")
    release.add_argument("draft", help="_drafts 下的文件名")
    release.add_argument("--date", help="发布日期 YYYY-MM-DD，默认本地今天")
    release.set_defaults(run=publish)
    verify = sub.add_parser("check", help="检查新发布文章")
    verify.set_defaults(run=check)
    args = parser.parse_args()
    try:
        args.run(args)
    except ValueError as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
