"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { filterPapers } = require("../assets/js/filters.js");
const papers = JSON.parse(fs.readFileSync(path.join(__dirname, "../data/papers.json"), "utf8"));
assert.equal(filterPapers(papers, {}).length, papers.length);
assert.equal(filterPapers(papers, { query: "gIgPo" }).length, 1);
assert.equal(filterPapers(papers, { query: "unlikely-nonexistent-phrase" }).length, 0);
const surveys = filterPapers(papers, { collection: "surveys" });
assert(surveys.length >= 6);
assert(surveys.every(p => p.tags.includes("Surveys")));
assert(surveys.some(p => p.id === "arxiv-2308.11432"));
const credit = filterPapers(papers, { collection: "credit" });
assert(credit.some(p => p.id === "arxiv-2610.15840"));
assert(credit.every(p => p.rl_topics.includes("Credit Assignment")));
assert.equal(filterPapers(papers, { collection: "surveys", year: 2026 }).length, 4);
const matched = filterPapers(papers, { capability: "Tool Use", rlTopic: "Credit Assignment", year: 2025 });
assert(matched.some(p => p.id === "arxiv-2505.10978"));
assert(matched.every(p => p.capabilities.includes("Tool Use") && p.rl_topics.includes("Credit Assignment") && p.year === 2025));
const oldest = filterPapers(papers, { sort: "oldest" });
assert(oldest.every((p, i) => i === 0 || (oldest[i-1].date || String(oldest[i-1].year)) <= (p.date || String(p.year))));
const alphabetic = filterPapers(papers, { sort: "title" });
assert(alphabetic.every((p, i) => i === 0 || alphabetic[i-1].title.localeCompare(p.title) <= 0));
const zh = JSON.parse(fs.readFileSync(path.join(__dirname, "../data/zh.json"), "utf8"));
const localized = papers.map(p => ({
  ...p, ...zh.papers[p.id],
  search_terms: [...p.capabilities,...p.rl_topics,...p.domains].map(term => zh.taxonomy[term] || term).join(" ")
}));
assert(filterPapers(localized, { query: "信用分配" }).some(p => p.id === "arxiv-2505.10978"));
assert(filterPapers(localized, { query: "里程碑" }).some(p => p.id === "entry-15"));
console.log("Filter checks passed.");
