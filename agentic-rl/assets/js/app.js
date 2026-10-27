"use strict";
const byId = id => document.getElementById(id);
const el = (tag, className, value) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined && value !== null) node.textContent = String(value);
  return node;
};
const safeLink = (url, label) => {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    if (!["https:", "http:"].includes(parsed.protocol)) return null;
    const link = el("a", null, label);
    link.href = parsed.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  } catch { return null; }
};
const add = (parent, ...children) => children.filter(Boolean).forEach(child => parent.append(child));
const shown = value => value === null || value === undefined || value === "" ? "Not documented" : String(value);
const labelStatus = item => item.verification_status === "source_checked" ? "Source checked" : "Pending verification";
let papers = [], benchmarks = [], problems = [], taxonomy = {}, frameworks = [];
function options(id, values) {
  const select = byId(id);
  values.forEach(value => select.add(new Option(value, value)));
}
function setupFilters() {
  options("capability", taxonomy.capabilities);
  options("rl-topic", taxonomy.rl_topics);
  options("domain", taxonomy.domains);
  options("year", [...new Set(papers.map(p => p.year))].sort((a,b) => b-a));
  ["query","capability","rl-topic","domain","year","sort"].forEach(id => byId(id).addEventListener(id === "query" ? "input" : "change", drawPapers));
  byId("reset-filters").addEventListener("click", () => {
    ["query","capability","rl-topic","domain","year"].forEach(id => byId(id).value = "");
    byId("sort").value = "newest";
    drawPapers();
    byId("query").focus();
  });
}
function paperCard(p) {
  const card = el("article","paper-card"); card.id = "paper-" + p.id;
  const top = el("div","card-top");
  add(top,el("span","badge" + (p.verification_status === "source_checked" ? "" : " pending"),labelStatus(p)),
    el("span","card-date",[p.date || p.year, p.venue || (p.publication_status === "preprint" ? "Preprint" : "")].filter(Boolean).join(" · ")));
  add(card,top,el("h3",null,p.title));
  if (p.authors?.length) add(card,el("p","authors",p.authors.join(", ")));
  const description = p.summary || "Summary pending source verification.";
  add(card,el("p","description",description));
  const tags = el("div","tag-list");
  [...p.capabilities,...p.rl_topics,...p.domains].forEach(tag => add(tags,el("span","tag",tag)));
  add(card,tags);
  const details = el("details");
  const detailText = p.verification_status === "source_checked"
    ? (p.limitations?.length ? p.limitations.join(" ") : "Limitations have not been assessed here.")
    : "This migrated record is awaiting source review. Earlier descriptive text is retained in the JSON legacy_note field and is not shown as a verified claim.";
  add(details,el("summary",null,"Evidence & limitations"),el("p",null,detailText));
  if (p.verified_at) add(details,el("p",null,"Metadata checked: " + p.verified_at));
  add(card,details);
  const links = el("div","card-links");
  add(links,safeLink(p.paper_url,"Paper ↗"),safeLink(p.code_url,"Code ↗"),safeLink(p.project_url,"Project ↗"));
  add(card,links);
  return card;
}
function drawPapers() {
  const list = AtlasFilters.filterPapers(papers, {
    query: byId("query").value, capability: byId("capability").value,
    rlTopic: byId("rl-topic").value, domain: byId("domain").value,
    year: byId("year").value, sort: byId("sort").value
  });
  const grid = byId("paper-grid"); grid.replaceChildren();
  if (list.length) list.forEach(p => grid.append(paperCard(p)));
  else add(grid,el("p","empty","No papers match these filters. Clear filters or try a broader term."));
  byId("result-count").textContent = list.length + " of " + papers.length + " paper records";
}
function drawTaxonomy() {
  const grid = byId("taxonomy-grid");
  [["Agent capabilities","capabilities","capability"],["RL research problems","rl_topics","rl-topic"],["Application domains","domains","domain"]].forEach(([heading,key,selectId]) => {
    const card = el("article","tax-card"), list = el("div","topic-list");
    add(card,el("h3",null,heading));
    taxonomy[key].forEach(topic => {
      const count = papers.filter(p => p[key].includes(topic)).length;
      const button = el("button",null,topic + " · " + count);
      button.type = "button";
      button.setAttribute("aria-label","Show " + count + " papers tagged " + topic);
      button.addEventListener("click",() => {
        byId(selectId).value = topic;
        drawPapers();
        byId("papers").scrollIntoView();
      });
      add(list,button);
    });
    add(card,list); add(grid,card);
  });
}
function drawBenchmarks() {
  const grid = byId("benchmark-grid");
  benchmarks.forEach(b => {
    const card = el("article","benchmark-card"); card.id = "benchmark-" + b.id;
    add(card,el("span","badge" + (b.verification_status === "source_checked" ? "" : " pending"),b.verification_status === "source_checked" ? "Source checked" : "Source linked"),
      el("h3",null,b.name));
    const dl = el("dl");
    [["Environment",b.environment_type],["Modality",b.modality],["Horizon",b.task_horizon],["Stateful",b.stateful === null ? null : b.stateful ? "Yes" : "No"],["Evaluation",b.evaluation]].forEach(([key,value]) => add(dl,el("dt",null,key),el("dd",null,shown(value))));
    add(card,dl,safeLink(b.source_url,"Primary source ↗"),safeLink(b.code_url && b.code_url !== b.source_url ? b.code_url : null,"Code ↗"));
    add(grid,card);
  });
}
function drawFrameworks() {
  const grid = byId("framework-grid");
  frameworks.forEach(f => {
    const card = el("article","benchmark-card");
    add(card,el("span","badge pending","Pending verification"),el("h3",null,f.title),el("p",null,"Project description pending source verification."));
    add(card,safeLink(f.source_url,"Source / project ↗"));
    add(grid,card);
  });
}
function drawProblems() {
  const grid = byId("problem-grid");
  problems.forEach((p,index) => {
    const card = el("article","problem-card");
    add(card,el("small",null,"QUESTION " + String(index+1).padStart(2,"0")),el("h3",null,p.title),el("p",null,p.definition));
    const details = el("details");
    add(details,el("summary",null,"Research sketch"));
    const list = el("ul");
    [["Why it matters",p.motivation],["Current gap",p.existing_limitations],["Possible approach",p.possible_approaches.join(" ")],["Cost estimate",p.estimated_experimental_cost + " — " + p.cost_note],["Risk",p.risks.join(" ")]].forEach(([key,value]) => add(list,el("li",null,key + ": " + value)));
    add(details,list);
    const related = el("div","related");
    p.related_paper_ids.forEach(id => {
      const paper = papers.find(x => x.id === id);
      if (paper) {
        const link = el("a",null,paper.title); link.href = "#paper-" + paper.id;
        link.addEventListener("click", () => {
          ["query","capability","rl-topic","domain","year"].forEach(id => byId(id).value = "");
          drawPapers();
        });
        add(related,link);
      }
    });
    if (related.childNodes.length) add(details,el("strong",null,"Related work"),related);
    const bench = el("p",null,"Suggested benchmarks: " + p.suggested_benchmark_ids.map(id => benchmarks.find(b => b.id === id)?.name || id).join(", "));
    add(details,bench); add(card,details); add(grid,card);
  });
}
async function load() {
  const files = ["papers","benchmarks","open-problems","taxonomy","frameworks"];
  const data = await Promise.all(files.map(async name => {
    const response = await fetch(new URL("data/" + name + ".json", document.baseURI));
    if (!response.ok) throw new Error(name + ": HTTP " + response.status);
    return response.json();
  }));
  [papers,benchmarks,problems,taxonomy,frameworks] = data;
  byId("paper-total").textContent = papers.length;
  byId("checked-total").textContent = papers.filter(p => p.verification_status === "source_checked").length;
  byId("benchmark-total").textContent = benchmarks.length;
  byId("problem-total").textContent = problems.length;
  setupFilters(); drawPapers(); drawTaxonomy(); drawBenchmarks(); drawFrameworks(); drawProblems();
}
load().catch(error => {
  byId("result-count").textContent = "The research data could not be loaded. Serve this folder over HTTP to preview it.";
  byId("paper-grid").append(el("p","empty",String(error.message)));
});
