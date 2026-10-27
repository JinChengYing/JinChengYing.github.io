"use strict";
const isZh = document.documentElement.lang.toLowerCase().startsWith("zh");
const t = (en, zh) => isZh ? zh : en;
const atlasBase = new URL("../../", document.querySelector('script[src$="assets/js/app.js"]').src);
let localization = {};
const term = value => isZh ? (localization.taxonomy?.[value] || value) : value;
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
const shown = value => value === null || value === undefined || value === "" ? t("Not documented","未记录") : String(value);
const labelStatus = item => item.verification_status === "source_checked" ? t("Source checked","来源已核对") : t("Pending verification","待核验");
let papers = [], benchmarks = [], problems = [], taxonomy = {}, frameworks = [];
let activeCollection = "";
function options(id, values, translate = false) {
  const select = byId(id);
  values.forEach(value => select.add(new Option(translate ? term(value) : value, value)));
}
function setupFilters() {
  options("capability", taxonomy.capabilities, true);
  options("rl-topic", taxonomy.rl_topics, true);
  options("domain", taxonomy.domains, true);
  options("year", [...new Set(papers.map(p => p.year))].sort((a,b) => b-a));
  ["query","capability","rl-topic","domain","year","sort"].forEach(id => byId(id).addEventListener(id === "query" ? "input" : "change", drawPapers));
  document.querySelectorAll("[data-collection]").forEach(button => button.addEventListener("click", () => {
    activeCollection = activeCollection === button.dataset.collection ? "" : button.dataset.collection;
    document.querySelectorAll("[data-collection]").forEach(item => item.setAttribute("aria-pressed", String(item.dataset.collection === activeCollection)));
    drawPapers();
  }));
  byId("reset-filters").addEventListener("click", () => {
    activeCollection = "";
    document.querySelectorAll("[data-collection]").forEach(item => item.setAttribute("aria-pressed", "false"));
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
    el("span","card-date",[p.date || p.year, p.venue || (p.publication_status === "preprint" ? t("Preprint","预印本") : "")].filter(Boolean).join(" · ")));
  add(card,top,el("h3",null,p.title));
  if (p.authors?.length) add(card,el("p","authors",p.authors.join(", ")));
  const description = p.summary || t("Summary pending source verification.","摘要待来源核验。");
  add(card,el("p","description",description));
  const tags = el("div","tag-list");
  [...p.tags,...p.capabilities,...p.rl_topics,...p.domains].forEach(tag => add(tags,el("span","tag",tag === "Surveys" ? t("Survey","综述") : term(tag))));
  add(card,tags);
  const details = el("details");
  const detailText = p.verification_status === "source_checked"
    ? (p.limitations?.length ? p.limitations.join(" ") : t("Limitations have not been assessed here.","本站尚未评估这篇论文的局限性。"))
    : t("This migrated record is awaiting source review. Earlier descriptive text is retained in the JSON legacy_note field and is not shown as a verified claim.","该记录继承自旧版，仍待来源复核。原描述仅保存在 JSON 的 legacy_note 字段，不作为已核实结论展示。");
  add(details,el("summary",null,t("Evidence & limitations","证据与局限")),el("p",null,detailText));
  if (p.verified_at) add(details,el("p",null,t("Metadata checked: ","元数据核对日期：") + p.verified_at));
  add(card,details);
  const links = el("div","card-links");
  add(links,safeLink(p.paper_url,t("Paper ↗","论文 ↗")),safeLink(p.code_url,t("Code ↗","代码 ↗")),safeLink(p.project_url,t("Project ↗","项目 ↗")));
  add(card,links);
  return card;
}
function drawPapers() {
  const list = AtlasFilters.filterPapers(papers, {
    collection: activeCollection,
    query: byId("query").value, capability: byId("capability").value,
    rlTopic: byId("rl-topic").value, domain: byId("domain").value,
    year: byId("year").value, sort: byId("sort").value
  });
  const grid = byId("paper-grid"); grid.replaceChildren();
  if (list.length) list.forEach(p => grid.append(paperCard(p)));
  else add(grid,el("p","empty",t("No papers match these filters. Clear filters or try a broader term.","没有符合条件的论文。请清除筛选条件或使用更宽泛的关键词。")));
  byId("result-count").textContent = isZh ? `显示 ${list.length} / ${papers.length} 篇论文` : list.length + " of " + papers.length + " paper records";
}
function drawTaxonomy() {
  const grid = byId("taxonomy-grid");
  [[t("Agent capabilities","智能体能力"),"capabilities","capability"],[t("RL research problems","强化学习研究问题"),"rl_topics","rl-topic"],[t("Application domains","应用领域"),"domains","domain"]].forEach(([heading,key,selectId]) => {
    const card = el("article","tax-card"), list = el("div","topic-list");
    add(card,el("h3",null,heading));
    taxonomy[key].forEach(topic => {
      const count = papers.filter(p => p[key].includes(topic)).length;
      const button = el("button",null,term(topic) + " · " + count);
      button.type = "button";
      button.setAttribute("aria-label",isZh ? `查看“${term(topic)}”下的 ${count} 篇论文` : "Show " + count + " papers tagged " + topic);
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
    add(card,el("span","badge" + (b.verification_status === "source_checked" ? "" : " pending"),b.verification_status === "source_checked" ? t("Source checked","来源已核对") : t("Source linked","附来源链接")),
      el("h3",null,b.name));
    const dl = el("dl");
    [[t("Environment","环境类型"),b.environment_type],[t("Modality","交互模态"),b.modality],[t("Horizon","任务跨度"),b.task_horizon],[t("Stateful","有状态"),b.stateful === null ? null : b.stateful ? t("Yes","是") : t("No","否")],[t("Evaluation","评测方式"),b.evaluation]].forEach(([key,value]) => add(dl,el("dt",null,key),el("dd",null,shown(value))));
    add(card,dl,safeLink(b.source_url,t("Primary source ↗","原始来源 ↗")),safeLink(b.code_url && b.code_url !== b.source_url ? b.code_url : null,t("Code ↗","代码 ↗")));
    add(grid,card);
  });
}
function drawFrameworks() {
  const grid = byId("framework-grid");
  frameworks.forEach(f => {
    const card = el("article","benchmark-card");
    const checked = f.verification_status === "source_checked";
    add(card,el("span","badge" + (checked ? "" : " pending"),checked ? t("Project source checked","项目来源已核对") : t("Pending verification","待核验")),el("h3",null,f.title));
    if (f.layer && checked) add(card,el("small","framework-layer",f.layer));
    add(card,el("p",null,checked ? f.description : t("Project description pending source verification.","项目描述待来源核验。")));
    add(card,safeLink(f.source_url,t("Source / project ↗","来源／项目 ↗")));
    add(grid,card);
  });
}
function drawProblems() {
  const grid = byId("problem-grid");
  problems.forEach((p,index) => {
    const card = el("article","problem-card");
    add(card,el("small",null,t("QUESTION ","研究问题 ") + String(index+1).padStart(2,"0")),el("h3",null,p.title),el("p",null,p.definition));
    const details = el("details");
    add(details,el("summary",null,t("Research sketch","研究思路")));
    const list = el("ul");
    [[t("Why it matters","研究动机"),p.motivation],[t("Current gap","现有局限"),p.existing_limitations],[t("Possible approach","可能路径"),p.possible_approaches.join(" ")],[t("Cost estimate","成本估计"),(isZh ? {"High":"较高","Medium":"中等","Low":"较低"}[p.estimated_experimental_cost] || p.estimated_experimental_cost : p.estimated_experimental_cost) + " — " + p.cost_note],[t("Risk","研究风险"),p.risks.join(" ")]].forEach(([key,value]) => add(list,el("li",null,key + t(": ","：") + value)));
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
    if (related.childNodes.length) add(details,el("strong",null,t("Related work","相关工作")),related);
    const bench = el("p",null,t("Suggested benchmarks: ","建议基准：") + p.suggested_benchmark_ids.map(id => benchmarks.find(b => b.id === id)?.name || id).join(isZh ? "、" : ", "));
    add(details,bench); add(card,details); add(grid,card);
  });
}
async function load() {
  const files = ["papers","benchmarks","open-problems","taxonomy","frameworks"];
  const data = await Promise.all(files.map(async name => {
    const response = await fetch(new URL("data/" + name + ".json", atlasBase));
    if (!response.ok) throw new Error(name + ": HTTP " + response.status);
    return response.json();
  }));
  [papers,benchmarks,problems,taxonomy,frameworks] = data;
  if (isZh) {
    const response = await fetch(new URL("data/zh.json", atlasBase));
    if (!response.ok) throw new Error("zh: HTTP " + response.status);
    localization = await response.json();
    papers = papers.map(p => ({
      ...p, ...localization.papers[p.id],
      search_terms: [...p.capabilities,...p.rl_topics,...p.domains].map(term).join(" ")
    }));
    benchmarks = benchmarks.map(b => ({...b,...localization.benchmarks[b.id]}));
    frameworks = frameworks.map(f => ({...f,...localization.frameworks?.[f.id]}));
    problems = problems.map(p => ({
      ...p,...localization.problems[p.id],
      cost_note: "编辑者估计，未经实测。"
    }));
  }
  byId("paper-total").textContent = papers.length;
  byId("checked-total").textContent = papers.filter(p => p.verification_status === "source_checked").length;
  byId("benchmark-total").textContent = benchmarks.length;
  byId("problem-total").textContent = problems.length;
  setupFilters(); drawPapers(); drawTaxonomy(); drawBenchmarks(); drawFrameworks(); drawProblems();
}
load().catch(error => {
  byId("result-count").textContent = t("The research data could not be loaded. Serve this folder over HTTP to preview it.","研究数据加载失败。请通过本地 HTTP 服务预览该页面。");
  byId("paper-grid").append(el("p","empty",String(error.message)));
});
