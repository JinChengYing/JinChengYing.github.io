# Agentic RL Atlas

A static research section of the existing JinChengYing GitHub Pages site. The public path remains `/agentic-rl/`. HTML, CSS, JavaScript and JSON live entirely in this directory; no root Jekyll templates or blog files need changes.

The Chinese version is at `/agentic-rl/zh/`. Both language versions share the same research records and rendering code. `data/zh.json` supplies Chinese summaries for checked papers, benchmark descriptions, taxonomy labels and open-problem text. Original paper titles and source URLs remain unchanged.

## Data model and evidence

- `data/papers.json`: paper records with bibliographic fields, three independent taxonomy axes, source URLs, and verification state.
- `data/frameworks.json`: project and software links migrated from the original page.
- `data/benchmarks.json`: environments with explicit unknown values for undocumented properties.
- `data/open-problems.json`: editorial research questions, linked by paper and benchmark IDs. Cost labels are curator estimates, not measured resource requirements.
- `data/taxonomy.json`: allowed capability, RL problem and domain labels.

A `source_checked` paper has a primary abstract or proceedings page reviewed for the displayed title, authors where populated, date where populated, venue where populated, and summary. The `verified_at` date records this review. A `unverified` record was migrated from the original page; its earlier description is retained as `legacy_note` for future review and is not shown as a scientific claim. `source_link_only` on benchmarks means the link is retained but detailed metadata has not been fully audited. Topic labels and open-problem sketches are editorial interpretations.

The verified source set at this revision includes [the landscape survey](https://arxiv.org/abs/2509.02547), [ReAct](https://arxiv.org/abs/2210.03629), [Tree of Thoughts](https://arxiv.org/abs/2305.10601), [τ-bench](https://arxiv.org/abs/2406.12045), [the credit assignment survey](https://arxiv.org/abs/2604.09459), [BEACON](https://proceedings.mlr.press/v306/wang26jm.html), [GiGPO](https://arxiv.org/abs/2505.10978), and [Agent Lightning](https://arxiv.org/abs/2508.03680). Scientific results in those sources are reported by their authors; this atlas has not reproduced them.

## Add or revise a paper

1. Read the primary paper abstract or proceedings record. Confirm the identifier, title, authors, submission date and venue before filling them. Do not infer a venue from the year.
2. Check for an existing arXiv ID or canonical paper URL. Use `arxiv-<ID>` for an arXiv record.
3. Add one JSON object to `data/papers.json`. Use `null` or empty arrays for unknown fields. Keep summaries short and bounded by the abstract. Put your own topic assignments in `capabilities`, `rl_topics` and `domains`.
4. Set `verification_status` to `source_checked` and `verified_at` to the date of the check only after the source is read. Otherwise use `unverified`.
5. Run `python agentic-rl/tools/validate_data.py` and `node agentic-rl/tools/test_filters.js` from the repository root. Inspect the page with a local HTTP server before publishing.

An example submission can copy an existing record and clear all unverified fields. Do not paste HTML into metadata. The interface inserts text with DOM `textContent` and accepts only HTTP(S) external links.

## Local preview

From the repository root:

```sh
python -m http.server 8000
```

Open `http://localhost:8000/agentic-rl/`. Opening `index.html` directly as a file may block JSON `fetch` calls. A full Jekyll build is useful for final site integration, but the atlas itself has no build dependencies and uses relative asset URLs.

The Chinese preview is `http://localhost:8000/agentic-rl/zh/`.

## Current limits

The original page had 24 mixed paper/project entries; this revision retains all of them as 20 paper records and 4 software records, and adds 2 source-checked papers. Most migrated records and most benchmark properties still need source audits. Search is client-side and intentionally small-scale. Link validity has not been checked for every external URL. The local workspace used for this change did not contain a checkout of the complete personal-site repository, so a full Jekyll integration build has not been run here.

