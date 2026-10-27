# 博客发文指南

博客由 Jekyll 和 GitHub Pages 提供。文章是 Markdown 文件，推送到 `main` 后由 GitHub Pages 发布；`_drafts` 中的草稿不会公开。

## 本地写作（推荐）

1. 克隆完整仓库并安装 Python 3，无需为草稿工具安装依赖。
2. 创建草稿：

   ```bash
   python scripts/posts.py new --title "我的文章标题" --slug my-research-note --summary "一句话摘要" --tags "强化学习,阅读笔记"
   ```

3. 编辑 `_drafts/my-research-note.md`，删除正文中的占位注释，添加内容和图片。图片可放在 `images/`，正文使用站点绝对路径，如 `/images/example.png`。
4. 发布并检查：

   ```bash
   python scripts/posts.py publish my-research-note.md
   python scripts/posts.py check
   ```

5. 检查改动，再提交并推送：

   ```bash
   git status
   git add _drafts _posts images
   git commit -m "Publish my research note"
   git push origin main
   ```

`publish` 会按本机日期生成 `_posts/YYYY-MM-DD-my-research-note.md`，也可传 `--date 2026-10-27` 指定日期。文章 URL 由 `_config.yml` 中的 `/:title/` 规则决定，发布后不要随意改 slug，以免旧链接失效。

## 直接用 GitHub 网页发布

在仓库网页中进入 `_posts`，新建 `YYYY-MM-DD-english-slug.md`。文件开头写：

```yaml
---
layout: post
title: "我的文章标题"
description: "一句话摘要"
date: 2026-10-27
tags: [强化学习, 阅读笔记]
post_workflow: v1
---

这里开始写正文。
```

提交到 `main` 后查看 Actions 的 **Build blog** 检查，再打开博客确认文章、图片和公式。日期太靠后时，Jekyll 可能暂不显示该文章。需要先保存草稿时，在 `_drafts` 新建文件即可。

博客首页、顶部导航和页脚都可以进入 [Agentic RL 研究图谱](https://JinChengYing.github.io/agentic-rl/zh/)；研究图谱顶部也可以返回博客。
