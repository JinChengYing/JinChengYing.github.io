"use strict";
(function (root) {
  function filterPapers(papers, criteria) {
    const query = (criteria.query || "").trim().toLocaleLowerCase();
    const filtered = papers.filter(paper => {
      const haystack = [paper.title, ...(paper.authors || []), paper.summary,
        ...(paper.tags || []), ...(paper.capabilities || []), ...(paper.rl_topics || []),
        ...(paper.domains || []), paper.search_terms].filter(Boolean).join(" ").toLocaleLowerCase();
      return (!query || haystack.includes(query)) &&
        (!criteria.capability || (paper.capabilities || []).includes(criteria.capability)) &&
        (!criteria.rlTopic || (paper.rl_topics || []).includes(criteria.rlTopic)) &&
        (!criteria.domain || (paper.domains || []).includes(criteria.domain)) &&
        (!criteria.year || String(paper.year) === String(criteria.year));
    });
    const key = paper => paper.date || String(paper.year || 0) + "-01-01";
    return filtered.sort((a, b) => criteria.sort === "title"
      ? a.title.localeCompare(b.title)
      : criteria.sort === "oldest" ? key(a).localeCompare(key(b)) : key(b).localeCompare(key(a)));
  }
  root.AtlasFilters = { filterPapers };
  if (typeof module !== "undefined" && module.exports) module.exports = { filterPapers };
})(globalThis);
