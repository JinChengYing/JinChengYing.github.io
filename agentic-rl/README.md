# Agentic RL Atlas

A static research section of the existing JinChengYing GitHub Pages site. The public path remains `/agentic-rl/`. The blog homepage links to the atlas and its 2026 research update.

The Chinese version is at `/agentic-rl/zh/`, with a longer editorial update at `/agentic-rl/zh/update-2026/`. Both atlas index versions share the same research records and rendering code. `data/zh.json` supplies Chinese summaries for checked papers, benchmark and framework descriptions, taxonomy labels and open-problem text. Original paper titles and source URLs remain unchanged.

The 2026 user-supplied outline is integrated into the atlas datasets, not published as a separate checklist. Source-checked papers belong in `data/papers.json`, environments in `data/benchmarks.json`, software in `data/frameworks.json`, and source-less names in `data/candidates.json`. The candidate panels live inside the respective library sections and do not inflate verified-record counts. The supplied text had two crossed arXiv links: Voyager is 2305.16291 (not the DPO paper 2305.18290), and LATS is 2310.04406 (not the SWE-bench paper 2310.06770).

## Data model and evidence

- `data/papers.json`: paper records with bibliographic fields, three independent taxonomy axes, source URLs, and verification state.
- `data/frameworks.json`: project and software links migrated from the original page.
- `data/benchmarks.json`: environments with explicit unknown values for undocumented properties.
- `data/candidates.json`: method, benchmark, and software names from the supplied outline that still lack a checked primary source. These are displayed as candidates within each library section, not as source-checked records.
- `data/open-problems.json`: editorial research questions, linked by paper and benchmark IDs. Cost labels are curator estimates, not measured resource requirements.
- `data/taxonomy.json`: allowed capability, RL problem and domain labels.

A `source_checked` paper has a primary abstract or proceedings page reviewed for the displayed title, authors where populated, date where populated, venue where populated, and summary. The `verified_at` date records this review. A `unverified` record was migrated from the original page; its earlier description is retained as `legacy_note` for future review and is not shown as a scientific claim. `source_link_only` on benchmarks means the link is retained but detailed metadata has not been fully audited. Topic labels and open-problem sketches are editorial interpretations.

The verified source set now also includes [SHADOW](https://ojs.aaai.org/index.php/AAAI/article/view/39570), [ΔBelief-RL](https://proceedings.mlr.press/v306/auzina26a.html), [TRACE](https://arxiv.org/abs/2607.13988), [ReBel](https://arxiv.org/abs/2605.20061), [AgeMem](https://aclanthology.org/2026.acl-long.981/), [MemoryArena](https://proceedings.mlr.press/v306/he26am.html), [AMA-Bench](https://proceedings.mlr.press/v306/zhao26bs.html), [Mem2ActBench](https://aclanthology.org/2026.acl-long.370/), [H-EPM](https://proceedings.mlr.press/v306/li26bn.html), [SINKFLEX-RL](https://arxiv.org/abs/2608.10357), and [Agent Lightning v1.0](https://arxiv.org/abs/2608.17528). Scientific results in those sources are reported by their authors; this atlas has not reproduced them. The 2026 credit-assignment survey's current arXiv v3 abstract reports 69 related papers, including 56 core methods. Earlier counts should not be presented as current.

On 2026-10-27 the library expanded from 25 to 39 records with four further surveys and ten credit-assignment methods. The source-reviewed batch and its Chinese summaries are recorded in `tools/expand_literature_2026_10.py`; each arXiv identifier links to the abstract used for the displayed metadata and short summary. The added October preprints are not independently reproduced or treated as peer-reviewed. The page offers exact shortcuts for survey records and the `Credit Assignment` RL topic.

The 2026 outline adds 24 source-checked paper records (including PPO, DPO, RAP, DeepSeekMath, BudgetMem, MARL foundations, and source papers for several web/code benchmarks), upgrades five existing records after abstract review, adds six benchmark entries and six framework entries. The idempotent imports are `tools/import_survey_core_2026.py` and `tools/import_survey_infrastructure_2026.py`. The atlas now has 63 papers, 18 benchmarks and 13 frameworks. The wider method-name set remains in the candidate panels pending source checks; no source-less name is represented as a verified publication.

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

The Chinese preview is `http://localhost:8000/agentic-rl/zh/`. The editorial update is `http://localhost:8000/agentic-rl/zh/update-2026/`. When previewing without Jekyll, the front matter will appear above the HTML; GitHub Pages removes it at build time.

## Current limits

The atlas remains a curated selection, not an exhaustive literature search. Some migrated records and benchmark properties still need source audits. Open problems, research priorities and cost labels are editorial analysis. Search is client-side and intentionally small-scale. External links were verified selectively, not exhaustively. The local workspace does not contain a checkout of the complete personal-site repository, so the full Jekyll build must be verified by GitHub Pages after publication.

