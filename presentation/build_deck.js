// Builds PT-PVCGN_project_overview.pptx — the paper, the research gap, and every file in the project.
// Run: node presentation/build_deck.js   (from the project root)
const path = require("path");
const pptxgen = require("pptxgenjs");
// Optional: a helper that writes THEME's colours into the .pptx theme part. Without it the deck still builds,
// with PowerPoint's default theme colours. Point APPLY_THEME_JS at an apply_theme.js module to enable it.
let applyTheme = async () => console.warn("apply_theme.js not found: theme colours not applied");
try {
  ({ applyTheme } = require(process.env.APPLY_THEME_JS || "./apply_theme.js"));
} catch (e) { /* keep the no-op */ }

const THEME = {
  name: "Metro Night",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14213D", lt1: "FFFFFF", dk2: "46546B", lt2: "EEF2F7",
    accent1: "FCA311", accent2: "2A9D8F", accent3: "E76F51",
    accent4: "8D99AE", accent5: "264653", accent6: "F4D58D",
    hlink: "2A9D8F", folHlink: "264653",
  },
};
const HEX = THEME.colors;
const OUT = path.join(__dirname, "PT-PVCGN_project_overview.pptx");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "PT-PVCGN: project overview";
pres.author = "Nitin Bharti";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 40, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.4, fontSize: 18, color: C.accent6, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 2.6, y: 2.6, w: 10.0, h: 1.2, fontSize: 40, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 2.6, y: 3.8, w: 10.0, h: 1.0, fontSize: 18, color: C.accent6, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: HEX.dk2, align: "right" },
  objects: [
    { text: { text: "PT-PVCGN  ·  minor project", options: { x: 0.6, y: 7.0, w: 5, h: 0.3, fontSize: 10, color: HEX.dk2, margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle", margin: 0 }, text: "" } },
  ],
});

// ---------- helpers ----------
let objId = 0;
const nid = (p) => `${p}-${++objId}`;

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

// station dot = the deck's motif (metro-map stop)
function dot(s, x, y, d, label, fill = C.accent1, txt = C.text1, size = 14) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: C.background1, width: 2 }, objectName: nid("dot") });
  if (label !== undefined && label !== "") {
    s.addText(String(label), { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: size, bold: true, color: txt, margin: 0, isTextBox: true, objectName: nid("dotlabel") });
  }
}

// a file / concept card: tinted rounded box, title line with a station dot, body text
function card(s, x, y, w, h, title, body, opts = {}) {
  const fill = opts.fill || C.background2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: nid("card") });
  dot(s, x + 0.2, y + 0.22, 0.26, "", opts.dot || C.accent1);
  s.addText(title, { x: x + 0.56, y: y + 0.12, w: w - 0.7, h: 0.46, fontSize: opts.titleSize || 16, bold: true, color: opts.titleColor || C.text1, fontFace: opts.mono ? "Courier New" : undefined, margin: 0, valign: "middle", isTextBox: true, objectName: nid("cardtitle") });
  const items = Array.isArray(body) ? body : [body];
  s.addText(
    items.map((t, i) => ({ text: t, options: { bullet: items.length > 1, breakLine: i < items.length - 1, paraSpaceAfter: items.length > 1 ? 4 : 0 } })),
    { x: x + 0.2, y: y + 0.62, w: w - 0.4, h: h - 0.74, fontSize: opts.bodySize || 16, color: opts.bodyColor || C.text1, valign: "top", margin: 0, isTextBox: true, objectName: nid("cardbody") }
  );
}

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt" };

// ======================================================================
// 1. Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
s.addText("Metro ridership forecasting with periodic and calendar context", { placeholder: "title" });
s.addText("Extending PVCGN (Liu et al., IEEE T-ITS 2022): the paper, the research gap we chose, and a guide to every file in the project", { placeholder: "body" });
for (let i = 0; i < 6; i++) dot(s, 0.8 + i * 0.55, 1.4, 0.28, "", i === 5 ? C.accent2 : C.accent1);
s.addText("Nitin Bharti", { x: 0.8, y: 6.3, w: 6, h: 0.4, fontSize: 16, color: C.background1, margin: 0, isTextBox: true, objectName: "author" });
s.addNotes("Title. This deck explains the original PVCGN paper, the research gap we are addressing (PVCGN ignores periodic and calendar information), our extension PT-PVCGN, and what every folder and code file in the project does.");

// 2. Agenda as a metro line
s = content("Route map for this talk", "Introduction");
const stops = [
  ["1", "The original paper", "What PVCGN does and why"],
  ["2", "Research gap", "What PVCGN misses, with evidence"],
  ["3", "Our solution", "PT-PVCGN and how we test it fairly"],
  ["4", "Code walkthrough", "Every folder and file, and its purpose"],
  ["5", "Results", "Final test results and conclusion"],
];
s.addShape(pres.shapes.LINE, { x: 1.4, y: 3.05, w: 10.6, h: 0, line: { color: C.accent1, width: 6 }, objectName: "agenda-line" });
stops.forEach(([n, t, d], i) => {
  const x = 0.95 + i * 2.65;
  dot(s, x, 2.6, 0.9, n, i === 2 ? C.accent2 : C.accent1, i === 2 ? C.background1 : C.text1, 22);
  s.addText(t, { x: x - 0.65, y: 3.75, w: 2.2, h: 0.5, fontSize: 18, bold: true, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: nid("stop-title") });
  s.addText(d, { x: x - 0.65, y: 4.25, w: 2.2, h: 0.9, fontSize: 14, color: C.text2, align: "center", margin: 0, isTextBox: true, objectName: nid("stop-desc") });
});
s.addNotes("Five stops: the original paper, the gap, our solution, a walkthrough of every code file, and current status.");

// ======================================================================
pres.addSection({ title: "The original paper" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "The original paper" });
s.addText("The original paper: PVCGN", { placeholder: "title" });
s.addText("Physical-Virtual Collaboration Graph Network — Liu, Chen, Wu, Zhen, Li, Lin · IEEE T-ITS 23(4), 2022", { placeholder: "body" });
dot(s, 0.9, 2.75, 1.2, "1", C.accent1, C.text1, 32);
s.addNotes("Section 1: the paper we start from.");

// 3. The problem
s = content("The problem: forecast every station's next hour", "The original paper");
[["8.8 M", "trips per day on the Shanghai metro (2016 data)"], ["15 min", "time resolution: inflow and outflow per station"], ["1 hour", "forecast horizon: the next four 15-minute slots"]].forEach(([big, small], i) => {
  const x = 0.6 + i * 4.15;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: 3.85, h: 2.2, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: nid("stat-bg") });
  s.addText(big, { x: x + 0.25, y: 1.7, w: 3.4, h: 1.1, fontSize: 54, bold: true, color: C.accent5, fontFace: THEME.headFontFace, margin: 0, isTextBox: true, objectName: nid("stat") });
  s.addText(small, { x: x + 0.25, y: 2.85, w: 3.4, h: 0.8, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, objectName: nid("stat-label") });
});
s.addText([
  { text: "Why it matters: ", options: { bold: true } },
  { text: "operators use these forecasts to add trains before crowds build, place staff at busy stations, control platform crowding and warn passengers. A wrong forecast at rush hour can mean dangerous overcrowding or wasted trains." },
], { x: 0.6, y: 4.2, w: 12.1, h: 1.2, fontSize: 16, color: C.text1, margin: 0, isTextBox: true, objectName: "why" });
s.addText([
  { text: "Inflow", options: { bold: true } }, { text: " = passengers entering a station.  " },
  { text: "Outflow", options: { bold: true } }, { text: " = passengers leaving it.  Both are counted every 15 minutes from smart-card taps." },
], { x: 0.6, y: 5.55, w: 12.1, h: 0.8, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, objectName: "def" });
s.addNotes("Station-level metro ridership prediction: forecast inflow and outflow of every station for the next hour, in 15-minute steps. Shanghai alone had 8.82 million trips per day in the dataset.");

// 4. Three graphs
s = content("PVCGN's idea: describe the metro with three graphs", "The original paper");
const graphs = [
  ["Physical graph", "Stations joined if they are next to each other on a metro line. Crowds physically move along the tracks.", C.accent1],
  ["Similarity graph (virtual)", "Stations joined if their daily ridership curves look alike (dynamic time warping), even when far apart — e.g. two office districts.", C.accent2],
  ["Correlation graph (virtual)", "Stations joined if many passengers travel between them (origin–destination ratios). If most of A's riders go to B, A helps predict B.", C.accent3],
];
graphs.forEach(([t, b, col], i) => card(s, 0.6 + i * 4.15, 1.55, 3.85, 3.0, t, b, { dot: col }));
const nodes = [[1.2, 5.45], [2.6, 5.45], [4.0, 5.45], [5.4, 5.45], [6.8, 5.45], [8.2, 5.45], [9.6, 5.45], [11.0, 5.45], [12.2, 5.45]];
s.addShape(pres.shapes.LINE, { x: 1.35, y: 5.6, w: 11.0, h: 0, line: { color: C.accent1, width: 4 }, objectName: "phys-edge" });
s.addShape(pres.shapes.ARC, { x: 2.75, y: 4.85, w: 6.9, h: 1.5, angleRange: [180, 360], line: { color: C.accent2, width: 2, dashType: "dash" }, objectName: "sim-edge" });
nodes.forEach(([x, y]) => dot(s, x, y, 0.3, "", C.accent5));
s.addText("A metro line (solid) and a virtual similarity edge (dashed) linking two distant stations", { x: 0.6, y: 6.2, w: 12.1, h: 0.4, fontSize: 12, color: C.text2, margin: 0, isTextBox: true, objectName: "net-caption" });
s.addNotes("The key idea of the paper: the physical track map is not enough. Two 'virtual' graphs connect stations that behave alike or exchange many passengers. All three share the same nodes (stations) but have different edges.");

// 5. Architecture
s = content("How PVCGN forecasts", "The original paper");
const flow = [
  ["Last hour", "4 × 15-min inflow / outflow for all stations"],
  ["Encoder", "2 stacked CGRM layers read the hour step by step"],
  ["Decoder", "2 CGRM layers produce one 15-min step at a time"],
  ["Next hour", "4 forecasts: 15, 30, 45, 60 min ahead"],
];
s.addShape(pres.shapes.LINE, { x: 1.85, y: 2.05, w: 9.3, h: 0, line: { color: C.accent1, width: 6 }, objectName: "flow-line" });
flow.forEach(([t, d], i) => {
  const x = 0.6 + i * 3.1;
  dot(s, x + 0.8, 1.6, 0.9, String(i + 1), C.accent1, C.text1, 22);
  s.addText(t, { x, y: 2.65, w: 2.5, h: 0.45, fontSize: 18, bold: true, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: nid("flow-t") });
  s.addText(d, { x, y: 3.1, w: 2.5, h: 0.8, fontSize: 14, color: C.text2, align: "center", margin: 0, isTextBox: true, objectName: nid("flow-d") });
});
card(s, 0.6, 4.25, 5.95, 2.45, "Inside a CGRM: GC-GRU (local)", "A gated recurrent unit whose gates use graph convolution: each station mixes its own state with its neighbours' in all three graphs, every time step.", { dot: C.accent2 });
card(s, 6.75, 4.25, 5.95, 2.45, "Inside a CGRM: FC-GRU (global)", "A fully-connected GRU over all stations at once captures the city-wide trend; its output is fused with the local GC-GRU state.", { dot: C.accent3 });
s.addNotes("Sequence-to-sequence model. Each Collaborative Gated Recurrent Module (CGRM) combines a graph-convolution GRU for local spatial-temporal patterns and a fully-connected GRU for global patterns. Loss is mean absolute error.");

// 6. Datasets
s = content("Two benchmark datasets released with the paper", "The original paper");
card(s, 0.6, 1.55, 5.95, 3.4, "SHMetro — Shanghai", ["288 stations, 958 physical edges", "811.8 M transactions, Jul–Sep 2016", "Train Jul 1–Aug 31 · Val Sep 1–9 · Test Sep 10–30", "Test period includes a holiday (Sep 15–17)"], { dot: C.accent1 });
card(s, 6.75, 1.55, 5.95, 3.4, "HZMetro — Hangzhou", ["80 stations, 248 physical edges", "2.35 M trips per day, January 2019", "Train Jan 1–18 · Val Jan 19–20 · Test Jan 21–25", "Validation days are a weekend; test days are weekdays"], { dot: C.accent2 });
s.addText("Service hours 05:30–23:30 → 73 slots of 15 minutes per day, 66 forecast windows per day. Splits are by date, so the model is always tested on days after the ones it trained on.", { x: 0.6, y: 5.25, w: 12.1, h: 1.0, fontSize: 16, color: C.text1, margin: 0, isTextBox: true, objectName: "ds-note" });
s.addNotes("Both datasets are public in the authors' GitHub repository. We use exactly the same splits so our numbers are comparable with the paper.");

// 7. Paper results chart
s = content("Paper result: PVCGN beat all compared methods", "The original paper");
const methods = ["HA", "RF", "GBDT", "MLP", "LSTM", "GRU", "ASTGCN", "STG2Seq", "DCRNN", "GCRNN", "Graph-WaveNet", "PVCGN"];
const maes = [36.37, 32.19, 30.88, 26.57, 25.76, 25.69, 27.34, 23.8, 23.76, 23.84, 24.07, 22.68];
s.addChart(pres.charts.BAR, [{ name: "MAE", labels: methods, values: maes }], {
  x: 0.6, y: 1.45, w: 8.3, h: 5.3, barDir: "bar",
  chartColors: maes.map((_, i) => (i === maes.length - 1 ? HEX.accent2 : HEX.accent4)),
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.00",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2,
  valGridLine: { color: "E3E8EF", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "HZMetro test MAE, 15 minutes ahead (lower is better)", titleFontSize: 14, titleColor: HEX.dk1,
  valAxisMinVal: 0, catAxisOrientation: "maxMin", ...chartText, objectName: "paper-chart",
});
card(s, 9.2, 1.55, 3.5, 4.9, "What the paper reports", ["Best MAE, RMSE and MAPE on both cities at all 4 horizons", "Also best at rush hours and at the busiest stations", "Numbers from Table III of the paper"], { dot: C.accent2 });
s.addNotes("Values are copied from Table III of the paper (HZMetro, 15-minute horizon). PVCGN reaches MAE 22.68. These are the paper's own numbers, not ours.");

// ======================================================================
pres.addSection({ title: "Research gap" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Research gap" });
s.addText("The research gap", { placeholder: "title" });
s.addText("PVCGN never looks at the clock, the calendar or earlier days", { placeholder: "body" });
dot(s, 0.9, 2.75, 1.2, "2", C.accent1, C.text1, 32);
s.addNotes("Section 2: what is missing in PVCGN.");

// 8. Gaps found
s = content("Gaps we found — and the one we chose", "Research gap");
const gaps = [
  ["Gap 1 · CHOSEN", "No periodic or calendar context: PVCGN sees only the last hour, never the time of day, weekday/holiday, or earlier days.", true],
  ["Gap 2 · future work", "Similarity and correlation graphs are fixed all day, though which stations behave alike changes between peaks.", false],
  ["Gap 3 · future work", "Point forecasts only: no prediction intervals, although busy stations are far noisier.", false],
  ["Gap 4 · handled in our tests", "Missing baselines (same-time-last-week, boosting with periodic features), single runs, holidays not analysed.", false],
];
gaps.forEach(([t, b, chosen], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.65;
  card(s, x, y, 5.95, 2.45, t, b, chosen ? { fill: C.accent5, titleColor: C.background1, bodyColor: C.background1, dot: C.accent1 } : { dot: C.accent4 });
});
s.addNotes("We chose Gap 1 because it has the strongest evidence in the data and the PVCGN authors themselves list periodicity as future work (Section VI). Gap 4 is handled by our evaluation protocol. Gaps 2 and 3 are left for future work.");

// 9. Evidence
s = content("Evidence: past days predict far better than one hour ago", "Research gap");
const lagLabels = ["15 min earlier", "1 hour earlier", "Same slot, yesterday", "Same slot, last week"];
s.addChart(pres.charts.BAR, [
  { name: "Hangzhou", labels: lagLabels, values: [0.953, 0.724, 0.929, 0.95] },
  { name: "Shanghai", labels: lagLabels, values: [0.954, 0.663, 0.868, 0.958] },
], {
  x: 0.6, y: 1.45, w: 7.6, h: 5.2, barDir: "col", barGrouping: "clustered",
  chartColors: [HEX.accent1, HEX.accent5], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.00",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2,
  valGridLine: { color: "E3E8EF", size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0.5, valAxisMaxVal: 1,
  showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk2,
  showTitle: true, title: "Correlation of each look-back with current station inflow", titleFontSize: 14, titleColor: HEX.dk1,
  ...chartText, objectName: "evidence-chart",
});
card(s, 8.5, 1.55, 4.2, 2.3, "Why this matters", "To forecast 60 minutes ahead, PVCGN's newest input is already an hour old (r ≈ 0.66–0.72). The same slot last week is much closer (r ≈ 0.95).", { dot: C.accent3 });
card(s, 8.5, 4.05, 4.2, 2.5, "And on training data", "Same slot on the last comparable day vs. ridership 60 min ahead: r = 0.96 (HZ), 0.99 (SH). The last observation: r = 0.70 / 0.66.", { dot: C.accent2 });
s.addNotes("Computed from the released data (ml_pipeline/figures/02_periodicity_lags.png and 07_reference_feature_strength.png). Metro ridership repeats very strongly from day to day and week to week, and PVCGN cannot see it.");

// ======================================================================
pres.addSection({ title: "Our solution" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Our solution" });
s.addText("Our solution: PT-PVCGN", { placeholder: "title" });
s.addText("PVCGN + Periodic inputs + Time (calendar) context", { placeholder: "body" });
dot(s, 0.9, 2.75, 1.2, "3", C.accent2, C.background1, 32);
s.addNotes("Section 3: what we change.");

// 10. PT-PVCGN
s = content("PT-PVCGN adds two switchable parts to PVCGN", "Our solution");
card(s, 0.6, 1.5, 5.95, 2.75, "P — periodic reference inputs", ["Same time slots on the most recent day of the same type (working / non-working)", "…and on the same weekday one week earlier", "Only past days: no future information"], { dot: C.accent1 });
card(s, 6.75, 1.5, 5.95, 2.75, "T — calendar context", ["Learned embeddings of time of day, day of week and working day", "Official calendar: holidays and make-up working days (e.g. Sun 18 Sep 2016)"], { dot: C.accent2 });
const ladder = [["PVCGN", "published model"], ["+ P", "periodic inputs"], ["+ P + T", "= PT-PVCGN"]];
s.addShape(pres.shapes.LINE, { x: 2.4, y: 5.35, w: 8.4, h: 0, line: { color: C.accent2, width: 6 }, objectName: "ladder-line" });
ladder.forEach(([t, d], i) => {
  const x = 1.9 + i * 4.2;
  dot(s, x, 4.9, 0.9, String(i + 1), i === 2 ? C.accent2 : C.accent4, C.background1, 20);
  s.addText(t, { x: x + 1.0, y: 4.85, w: 2.7, h: 0.5, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: nid("lad-t") });
  s.addText(d, { x: x + 1.0, y: 5.35, w: 2.7, h: 0.45, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, objectName: nid("lad-d") });
});
s.addText("Ablation study: train all three under identical settings to measure what each part adds. With P and T switched off the code is exactly PVCGN.", { x: 0.6, y: 6.05, w: 12.1, h: 0.75, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, objectName: "ablation-note" });
s.addNotes("The decoder receives 'what usually happens next' from the reference days, while the encoder sees how today differs from those days so far. Everything else (graphs, GRUs, training settings) is the original PVCGN.");

// 10b. Which models and why
s = content("Which models we use, and why", "Our solution");
const models = [
  ["PVCGN (reproduced)", "The published state of the art on these datasets — the model we must beat. Re-built exactly so the comparison is fair.", C.accent5],
  ["PT-PVCGN (ours)", "PVCGN + periodic inputs + calendar. Tests the research gap directly while keeping all of PVCGN's strengths.", C.accent2],
  ["+P only (ablation)", "PVCGN + periodic inputs, without calendar. Separates what each addition contributes.", C.accent2],
  ["Simple baselines", "Persistence, seasonal naive, historical average. Show how much of the task is plain repetition.", C.accent4],
  ["Ridge · gradient boosting", "Cheap models given the same periodic features. Show whether a deep graph model is worth its cost.", C.accent4],
  ["Why graph + GRU at all?", "A metro is a graph (space) and ridership is a sequence (time): graph convolution handles space, a GRU handles time.", C.accent1],
];
models.forEach(([t, b, col], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 2.65;
  card(s, x, y, 3.9, 2.45, t, b, { dot: col, bodySize: 15 });
});
s.addNotes("PT-PVCGN is the proposed model. PVCGN is the published reference, reproduced and verified against the authors' code. The +P variant and the non-deep baselines exist to show exactly where any improvement comes from.");

// 10c. How PT-PVCGN works
s = content("How PT-PVCGN makes a forecast", "Our solution");
const steps = [
  ["Inputs", "Each 15-min step, per station: in/out now, same slot on the reference day and last week, time-of-day and day-type codes"],
  ["Graph convolution", "Each station mixes its inputs with neighbours in the physical, similarity and correlation graphs"],
  ["GC-GRU memory", "Keeps 256 numbers per station, updated step by step over the last hour"],
  ["FC-GRU (global)", "Summarises the whole network and fuses the city-wide trend back in"],
  ["Decoder", "4 steps; each sees its target slot's reference values and time, then outputs in/out"],
];
s.addShape(pres.shapes.LINE, { x: 1.25, y: 1.95, w: 10.4, h: 0, line: { color: C.accent2, width: 6 }, objectName: "how-line" });
steps.forEach(([t, d], i) => {
  const x = 0.6 + i * 2.5;
  dot(s, x + 0.65, 1.55, 0.8, String(i + 1), i === 0 || i === 4 ? C.accent2 : C.accent1, i === 0 || i === 4 ? C.background1 : C.text1, 18);
  s.addText(t, { x: x - 0.1, y: 2.5, w: 2.3, h: 0.45, fontSize: 16, bold: true, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: nid("how-t") });
  s.addText(d, { x: x - 0.1, y: 2.95, w: 2.3, h: 1.9, fontSize: 14, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true, objectName: nid("how-d") });
});
card(s, 0.6, 5.0, 12.1, 1.7, "The key idea", "The model already knows what normally happens at, say, 08:45 on a working day, so it only has to learn how today differs. Steps 2–4 are unchanged PVCGN; our additions enter at steps 1 and 5.", { fill: C.accent5, titleColor: C.background1, bodyColor: C.background1, dot: C.accent1 });
s.addNotes("Steps 1 and 5 (teal) carry our additions: periodic reference values and calendar embeddings are appended to every station's input at every encoder and decoder step. The graph convolution, GC-GRU and FC-GRU are exactly PVCGN's.");

// 10d. Training
s = content("How the models are trained", "Our solution");
const train = [
  ["Loss", "Mean absolute error on real passenger counts — the paper's main metric."],
  ["Optimiser", "Adam, learning rate 0.001, cut ×10 at epochs 100 and 150 (paper schedule); gradient clipping at 5."],
  ["Curriculum learning", "Early on, the decoder gets the true previous values; gradually it must use its own predictions."],
  ["Picking the model", "Validation MAE after every epoch; keep the best; stop after 50 epochs without improvement."],
  ["Repeats", "3 seeds per model on Hangzhou, 2 on Shanghai; results reported as mean ± std."],
  ["Budget", "Hangzhou: 200 epochs for every model (~1–1.5 h per run). Shanghai: 30 epochs (~5–8 h per run)."],
];
train.forEach(([t, b], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 2.65;
  card(s, x, y, 3.9, 2.45, t, b, { dot: i < 3 ? C.accent1 : C.accent2, bodySize: 15 });
});
s.addNotes("All settings follow the authors' released PVCGN configuration, except the epoch budget, which is reduced to fit a laptop (at most 100 epochs). Hangzhou models use the paper's 200-epoch schedule; Shanghai models 30 epochs because of laptop compute. Every model in a city gets the same budget.");

// 11. Fair evaluation
s = content("How we make the comparison fair", "Our solution");
const gates = [["A", "Problem defined", "passed"], ["B", "Data and plan ready", "passed"], ["C", "Validation results", "passed"], ["D", "Final test + report", "passed"]];
s.addShape(pres.shapes.LINE, { x: 1.3, y: 2.0, w: 10.8, h: 0, line: { color: C.accent1, width: 6 }, objectName: "gate-line" });
gates.forEach(([g, t, st], i) => {
  const x = 0.85 + i * 3.6;
  const done = st === "passed";
  dot(s, x, 1.55, 0.9, g, done ? C.accent2 : C.accent1, done ? C.background1 : C.text1, 22);
  s.addText(t, { x: x - 0.6, y: 2.55, w: 2.1, h: 0.45, fontSize: 16, bold: true, color: C.text1, align: "center", margin: 0, isTextBox: true, objectName: nid("gate-t") });
  s.addText(st, { x: x - 0.6, y: 3.0, w: 2.1, h: 0.4, fontSize: 14, color: C.text2, align: "center", margin: 0, isTextBox: true, objectName: nid("gate-s") });
});
card(s, 0.6, 3.75, 3.9, 2.9, "Same data, same metric", "Paper's exact date splits and its MAE / RMSE / MAPE code. PVCGN re-implemented and checked equal to the authors' code.", { dot: C.accent2 });
card(s, 4.72, 3.75, 3.9, 2.9, "Strong baselines", "Persistence, seasonal naive, historical average, ridge and gradient boosting with periodic features.", { dot: C.accent1 });
card(s, 8.84, 3.75, 3.86, 2.9, "Test set used once", "All choices made on validation. 2–3 seeds per model. Every result reported honestly, good or bad.", { dot: C.accent3 });
s.addNotes("The workflow has four review gates where the project owner signs off; all four are passed. The test set was used once, after the third gate.");

// ======================================================================
pres.addSection({ title: "Code walkthrough" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Code walkthrough" });
s.addText("Code walkthrough", { placeholder: "title" });
s.addText("Every folder and file in Desktop/Minor, and why it exists", { placeholder: "body" });
dot(s, 0.9, 2.75, 1.2, "4", C.accent1, C.text1, 32);
s.addNotes("Section 4: a guided tour of the project folder.");

// 11b. Tech stack
s = content("Tech stack", "Code walkthrough");
const stack = [
  ["Python 3.14", "The language for all data, model and analysis code."],
  ["PyTorch 2.14 + Apple MPS", "Builds and trains the neural networks on the Mac's built-in GPU."],
  ["NumPy 2.5 · pandas 3.1", "Arrays and tables: loading data, building features, computing metrics."],
  ["scikit-learn 1.9", "Ridge regression and gradient-boosting baselines."],
  ["matplotlib 3.11", "Every analysis and results figure."],
  ["marimo 0.25", "Notebooks (01_eda, 02_prep, 03_model) that also run as plain scripts."],
  ["torch_geometric 2.8", "Only to run the authors' original code in the equivalence test."],
  ["git", "Version history, with a checkpoint at every review gate."],
  ["LaTeX · Node + pptxgenjs", "The paper (IEEE template) and this presentation."],
];
stack.forEach(([t, b], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.45 + Math.floor(i / 3) * 1.8;
  card(s, x, y, 3.9, 1.65, t, b, { dot: i < 2 ? C.accent2 : C.accent1, bodySize: 14 });
});
s.addNotes("Everything runs locally on an Apple M1 laptop with 8 GB of memory. PyTorch uses the Metal (MPS) GPU backend. Our PVCGN does not need torch_geometric; it is installed only so the test can run the authors' code side by side.");

// 12. Folder map
s = content("Project folder map: Desktop/Minor", "Code walkthrough");
const folders = [
  ["metro/", "Our Python package: data loading, features, model, training, metrics, baselines."],
  ["ml_pipeline/", "The step-by-step workflow: notebooks, guard, progress log, figures, saved splits and run results."],
  ["reference_PVCGN/", "The authors' original code and data, cloned from GitHub. Not modified."],
  ["tests/", "Automatic check that our PVCGN equals the authors' version."],
  ["paper/", "Draft of the new research paper (LaTeX) and its references."],
  ["presentation/", "This deck and the script that builds it."],
];
folders.forEach(([t, b], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 2.45;
  card(s, x, y, 3.9, 2.25, t, b, { mono: true, dot: i < 2 ? C.accent2 : C.accent1 });
});
s.addText("Also at the top level: run_queue.sh (runs all training jobs), .gitignore, .venv/ (Python environment), checkpoints/ (saved model weights).", { x: 0.6, y: 6.4, w: 12.1, h: 0.5, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, objectName: "folders-note" });
s.addNotes("The two folders we wrote and use most are metro/ (the code) and ml_pipeline/ (the workflow and its records).");

// 13. Data flow
s = content("How the files connect", "Code walkthrough");
const chain = [
  ["reference_PVCGN/data", "released .pkl files"],
  ["data.py", "rebuild daily series"],
  ["features.py", "calendar + reference days"],
  ["02_prep.py", "split, scaler, features.npz"],
  ["batches.py", "model-ready batches"],
  ["model.py", "PVCGN / PT-PVCGN"],
  ["train.py", "train + pick best epoch"],
];
s.addShape(pres.shapes.LINE, { x: 1.45, y: 2.25, w: 10.5, h: 0, line: { color: C.accent1, width: 6 }, objectName: "chain-line" });
chain.forEach(([t, d], i) => {
  const x = 0.6 + i * 1.75;
  dot(s, x + 0.45, 1.85, 0.8, String(i + 1), i >= 5 ? C.accent2 : C.accent1, i >= 5 ? C.background1 : C.text1, 18);
  s.addText(t, { x: x - 0.1, y: 2.8, w: 1.7, h: 0.7, fontSize: 13, bold: true, color: C.text1, align: "center", fontFace: "Courier New", margin: 0, isTextBox: true, objectName: nid("ch-t") });
  s.addText(d, { x: x - 0.1, y: 3.5, w: 1.7, h: 0.8, fontSize: 12, color: C.text2, align: "center", margin: 0, isTextBox: true, objectName: nid("ch-d") });
});
card(s, 0.6, 4.6, 3.9, 2.1, "Side branch: baselines", "baselines.py fits simple models on the same features; 03_model.py scores them.", { dot: C.accent4 });
card(s, 4.72, 4.6, 3.9, 2.1, "Shared: metrics.py", "One MAE / RMSE / MAPE implementation used by every model, identical to the paper's.", { dot: C.accent4 });
card(s, 8.84, 4.6, 3.86, 2.1, "Outputs", "ml_pipeline/runs/*.json (scores per epoch) and checkpoints/*.pt (best weights).", { dot: C.accent4 });
s.addNotes("Data moves left to right: raw released windows become continuous daily series, get calendar and reference-day information, are split and scaled, batched, fed to the model and trained.");

// 14. metro/ part 1
s = content("metro/ — data side", "Code walkthrough");
card(s, 0.6, 1.5, 3.9, 5.15, "data.py", ["Loads the authors' train / val / test .pkl files and graphs", "Rebuilds continuous days (73 slots × stations × in/out) from overlapping windows, and checks the rebuild exactly", "long_table(): one row per station per 15 min, used for analysis"], { mono: true, dot: C.accent2 });
card(s, 4.72, 1.5, 3.9, 5.15, "features.py", ["Official calendar: holidays and make-up working days", "Working-day flag for every date", "Reference days: last earlier day of the same type, and same weekday a week earlier", "Only uses past days"], { mono: true, dot: C.accent2 });
card(s, 8.84, 1.5, 3.86, 5.15, "batches.py", ["Loads the prepared features.npz and scaler", "Scales counts (z-score from training data)", "Builds each batch: last hour, targets, reference-day slots, time slot, weekday, working day"], { mono: true, dot: C.accent2 });
s.addNotes("These three files turn the released data into exactly what the model needs. The key research-gap feature (reference days) is computed in features.py.");

// 15. metro/ part 2
s = content("metro/ — model side", "Code walkthrough");
card(s, 0.6, 1.5, 5.95, 2.5, "model.py", ["PVCGN rewritten in plain PyTorch: graph convolution, GC-GRU + FC-GRU, encoder–decoder", "Switches periodic= and temporal= add our P and T parts"], { mono: true, dot: C.accent2 });
card(s, 6.75, 1.5, 5.95, 2.5, "train.py", ["Trains one model (city, variant, seed) on training days", "Scores validation every epoch; keeps the best epoch", "Never reads the test set"], { mono: true, dot: C.accent2 });
card(s, 0.6, 4.2, 5.95, 2.45, "metrics.py", ["MAE, RMSE and MAPE, ignoring zero-passenger slots — same rule as the paper's code", "Per horizon (15/30/45/60 min)"], { mono: true, dot: C.accent1 });
card(s, 6.75, 4.2, 5.95, 2.45, "baselines.py", ["Persistence, seasonal naive, historical average", "Ridge and gradient boosting on recent + reference-day + calendar features"], { mono: true, dot: C.accent1 });
s.addNotes("model.py holds both the reproduction of PVCGN and our extension; train.py runs the experiments; metrics.py and baselines.py make the comparison fair.");

// 16. guard.py
s = content("ml_pipeline/guard.py — the safety rails", "Code walkthrough");
s.addText("A helper library that stops the most common mistakes in machine-learning projects, above all data leakage: letting test information influence training or model choice.", { x: 0.6, y: 1.4, w: 12.1, h: 0.8, fontSize: 16, color: C.text1, margin: 0, isTextBox: true, objectName: "guard-intro" });
const gfun = [
  ["profile()", "Inspects the raw data: size, column types, missing values, duplicates, leakage suspects. Writes data_profile.json."],
  ["eda_figures() · fig()", "Saves every chart to figures/ and logs a one-line explanation in PIPELINE.md."],
  ["split()", "Splits by time (refuses random splits on dated data), checks no row is in two splits, and freezes a fingerprint of the test set."],
  ["final_test()", "Scores the final model on the test set exactly once; refuses a second try or a tampered test set; logs the score."],
];
gfun.forEach(([t, b], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 2.4 + Math.floor(i / 2) * 2.2;
  card(s, x, y, 5.95, 2.0, t, b, { mono: true, dot: i === 3 ? C.accent3 : C.accent1 });
});
s.addNotes("guard.py comes from the ML-pipeline workflow plugin and was copied into the project at step 1. In our project, split() reproduced the paper's date split exactly and froze the test set; final_test() will be used once at step 14.");

// 17. ml_pipeline notebooks
s = content("ml_pipeline/ — notebooks and progress log", "Code walkthrough");
card(s, 0.6, 1.5, 5.95, 2.5, "01_eda.py", ["Steps 1–2: inspects the data and draws the analysis charts", "Found the periodicity, weekday/weekend and holiday patterns"], { mono: true, dot: C.accent2 });
card(s, 6.75, 1.5, 5.95, 2.5, "02_prep.py", ["Steps 4–8: cleaning checks, calendar and reference days, guard split, scaler", "Saves features.npz and scaler.json per city"], { mono: true, dot: C.accent2 });
card(s, 0.6, 4.2, 5.95, 2.45, "03_model.py", ["Step 9 onward: scores all baselines on validation", "Will compare baselines, PVCGN and PT-PVCGN"], { mono: true, dot: C.accent2 });
card(s, 6.75, 4.2, 5.95, 2.45, "PIPELINE.md", ["The project's logbook: status of all 16 steps, review sign-offs, research gaps, decisions, figure notes"], { mono: true, dot: C.accent1 });
s.addNotes("The .py notebooks are marimo notebooks: open them with 'marimo edit ml_pipeline/01_eda.py' to see the cells, or run them as normal Python scripts.");

// 18. ml_pipeline records
s = content("ml_pipeline/ — saved results and records", "Code walkthrough");
card(s, 0.6, 1.5, 5.95, 2.5, "data_profile.json", ["Output of guard.profile(): data size, column kinds, leakage checks"], { mono: true, dot: C.accent1 });
card(s, 6.75, 1.5, 5.95, 2.5, "figures/", ["Every chart, numbered by pipeline step (e.g. 02_periodicity_lags.png)"], { mono: true, dot: C.accent1 });
card(s, 0.6, 4.2, 5.95, 2.45, "hz/  and  sh/", ["Per city: features.npz (prepared data), scaler.json, split_state.json (frozen test fingerprint)"], { mono: true, dot: C.accent1 });
card(s, 6.75, 4.2, 5.95, 2.45, "runs/", ["One .json per training run: settings and scores per epoch", "baselines_val.json and logs/"], { mono: true, dot: C.accent1 });
s.addNotes("These files are the evidence trail for the paper: every number in it will be generated from runs/ and the figures.");

// 19. Other files
s = content("Other project files", "Code walkthrough");
card(s, 0.6, 1.5, 5.95, 2.5, "tests/test_equivalence.py", ["Copies the same weights into our PVCGN and the authors' PVCGN", "Checks both give the same output (to 1e-5)"], { mono: true, dot: C.accent2 });
card(s, 6.75, 1.5, 5.95, 2.5, "run_queue.sh", ["Runs every training job one after another", "Skips jobs already finished, so it can restart safely"], { mono: true, dot: C.accent2 });
card(s, 0.6, 4.2, 5.95, 2.45, "paper/main.tex · refs.bib", ["Draft of the new paper (IEEE format) and its bibliography", "Results sections fill in after the final test"], { mono: true, dot: C.accent1 });
card(s, 6.75, 4.2, 5.95, 2.45, ".gitignore · .venv/ · checkpoints/", ["Files kept out of git; the Python environment; saved model weights"], { mono: true, dot: C.accent4 });
s.addNotes("The test file is what lets us call our PVCGN a faithful reproduction. run_queue.sh is what runs overnight.");

// 20. Authors' code
s = content("reference_PVCGN/ — the authors' original code", "Code walkthrough");
const ref = [
  ["ggnn/multigraph.py", "PVCGN model: GGRU cell (GC-GRU + FC-GRU) and the encoder–decoder"],
  ["ggnn/rgcn.py", "Relational graph convolution layer (one weight per graph)"],
  ["ggnn_train.py", "Original training script; YAML configs in data/model/"],
  ["ggnn_evaluation*.py", "Original evaluation: all data, rush hours, top stations"],
  ["ggnn_utils.py · lib/", "Batching (torch_geometric), data loading, scaler, metrics"],
  ["data/", "Released datasets (.pkl) and the three graphs per city"],
];
ref.forEach(([t, b], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 2.6;
  card(s, x, y, 3.9, 2.4, t, b, { mono: true, titleSize: 14, dot: C.accent4 });
});
s.addNotes("We read this code to reproduce PVCGN exactly, use its data and graphs, and test against it. We do not modify it. It needs torch_geometric; our version does not.");

// ======================================================================
// Results — every number is read from ml_pipeline/runs/test_results.json (single test evaluation)
const TEST = require(path.join(__dirname, "..", "ml_pipeline", "runs", "test_results.json"));
const MAIN = "PT-PVCGN (+P+T)";
const mae = (c, m, sl = "all") => TEST[c][m][sl].mean.mae;
const f2 = (v) => v.toFixed(2);
const pct = (a, b) => (100 * (a - b) / a).toFixed(1);

pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Results" });
s.addText("Final results", { placeholder: "title" });
s.addText("Test set of each city evaluated once, all models together, after every choice was made on validation", { placeholder: "body" });
dot(s, 0.9, 2.75, 1.2, "5", C.accent2, C.background1, 32);
s.addNotes("Section 5: the final test results. Nothing here was used to choose or tune a model.");

// R1. Headline test MAE
s = content("Test MAE: PT-PVCGN beats PVCGN on both cities", "Results");
const cats = ["PVCGN (paper-reported)", "PVCGN (re-trained)", "Ridge (lags+refs)", "PT-PVCGN (ours)"];
const keyOf = { "PVCGN (paper-reported)": "PVCGN (paper-reported)", "PVCGN (re-trained)": "PVCGN", "Ridge (lags+refs)": "Ridge (lags+refs)", "PT-PVCGN (ours)": MAIN };
s.addChart(pres.charts.BAR, [
  { name: "Hangzhou", labels: cats, values: cats.map((k) => +f2(mae("hz", keyOf[k]))) },
  { name: "Shanghai", labels: cats, values: cats.map((k) => +f2(mae("sh", keyOf[k]))) },
], {
  x: 0.6, y: 1.45, w: 7.9, h: 5.2, barDir: "col", barGrouping: "clustered",
  chartColors: [HEX.accent1, HEX.accent5], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.00",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2,
  valGridLine: { color: "E3E8EF", size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0,
  showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk2,
  showTitle: true, title: "Test MAE, mean of 4 horizons (lower is better)", titleFontSize: 14, titleColor: HEX.dk1,
  ...chartText, objectName: "final-chart",
});
card(s, 8.8, 1.55, 3.9, 2.45, "Hangzhou: " + pct(mae("hz", "PVCGN"), mae("hz", MAIN)) + "% lower error",
  "Same 200-epoch training for both. Our PVCGN matches the paper (" + f2(mae("hz", "PVCGN")) + " vs " + f2(mae("hz", "PVCGN (paper-reported)")) + ").", { dot: C.accent1 });
card(s, 8.8, 4.2, 3.9, 2.45, "Shanghai: " + pct(mae("sh", "PVCGN"), mae("sh", MAIN)) + "% lower error",
  "Both trained 30 epochs. Ours also beats the paper's 200-epoch PVCGN (" + f2(mae("sh", MAIN)) + " vs " + f2(mae("sh", "PVCGN (paper-reported)")) + ").", { dot: C.accent5 });
s.addNotes("Mean over 3 seeds (Hangzhou) and 2 seeds (Shanghai). The paper-reported PVCGN numbers are copied from the original paper's tables for context. Shanghai models had a much shorter training budget than the original work.");

// R2. Gain grows with horizon
s = content("The gain grows the further ahead we forecast", "Results");
const hzPV = TEST.hz.PVCGN.all.per_horizon.mae, hzPT = TEST.hz[MAIN].all.per_horizon.mae;
const hl = ["15 min", "30 min", "45 min", "60 min"];
s.addChart(pres.charts.BAR, [
  { name: "PVCGN (re-trained)", labels: hl, values: hzPV.map((v) => +f2(v)) },
  { name: "PT-PVCGN (ours)", labels: hl, values: hzPT.map((v) => +f2(v)) },
], {
  x: 0.6, y: 1.45, w: 7.9, h: 5.2, barDir: "col", barGrouping: "clustered",
  chartColors: [HEX.accent4, HEX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.00",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2,
  valGridLine: { color: "E3E8EF", size: 0.5 }, catGridLine: { style: "none" }, valAxisMinVal: 0,
  showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: HEX.dk2,
  showTitle: true, title: "Hangzhou test MAE by forecast horizon", titleFontSize: 14, titleColor: HEX.dk1,
  ...chartText, objectName: "horizon-chart",
});
card(s, 8.8, 1.55, 3.9, 5.1, "Exactly what the gap predicts",
  ["15 min ahead: " + pct(hzPV[0], hzPT[0]) + "% lower error", "60 min ahead: " + pct(hzPV[3], hzPT[3]) + "% lower error",
   "The further ahead, the less the last hour helps — and the more the same time on a past day helps"], { dot: C.accent2 });
s.addNotes("Per-horizon test MAE on Hangzhou, mean over 3 seeds, both models with the same 200-epoch schedule.");

// R2b. Comparison tables in the original paper's format (data: paper/tables/comparison.json, from make_tables.py)
const COMP = require(path.join(__dirname, "..", "paper", "tables", "comparison.json"));
const compSlides = [
  ["sh_whole", "Shanghai: whole test set", "Lowest MAE at every horizon; the paper's 200-epoch PVCGN† still has lower MAPE (and RMSE at 15–30 min). Our models trained 30 epochs."],
  ["hz_whole", "Hangzhou: whole test set", "PT-PVCGN beats every non-ours method on all metrics and horizons; only our +P ablation is better. Same 200-epoch budget for our models."],
  ["sh_rush", "Shanghai: rush hours", "Lowest rush-hour MAE at every horizon; the paper's PVCGN† has the lowest MAPE."],
  ["hz_rush", "Hangzhou: rush hours", "Lowest MAE and RMSE of all non-ours methods at every horizon; PVCGN† keeps marginally lower MAPE at 15–45 min."],
  ["sh_top", "Shanghai: busiest 1/4 of stations", "PVCGN† (200 epochs) is better at 15–30 min and on MAPE; PT-PVCGN is better at 45–60 min. Inconclusive at our 30-epoch budget."],
  ["hz_top", "Hangzhou: busiest 1/4 of stations", "PT-PVCGN beats every non-ours method on MAE and RMSE at every horizon; +P is best overall."],
];
compSlides.forEach(([key, title, note]) => {
  const d = COMP[key];
  s = content(title, "Results");
  const hdr = (t, ours) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: ours ? HEX.accent2 : HEX.dk1 }, align: "center", valign: "middle" } });
  const rows = [[hdr("Time"), hdr("Metric"), ...d.columns.map((c, i) => hdr(c, d.ours[i]))]];
  d.rows.forEach((r, i) => {
    const band = Math.floor(i / 3) % 2 === 0 ? "FFFFFF" : HEX.lt2;
    const row = [];
    if (i % 3 === 0) row.push({ text: r.time, options: { rowspan: 3, bold: true, valign: "middle", align: "center", fill: { color: band } } });
    row.push({ text: r.metric, options: { align: "center", fill: { color: band } } });
    r.cells.forEach((c, j) => row.push({ text: c, options: { align: "center", bold: r.best[j], color: r.best[j] ? HEX.accent5 : HEX.dk1, fill: { color: r.best[j] ? "D6EFEB" : band } } }));
    rows.push(row);
  });
  const nCols = d.columns.length;
  const colW = [0.8, 0.75, ...Array(nCols).fill((12.5 - 1.55) / nCols)];
  s.addTable(rows, { x: 0.4, y: 1.3, w: 12.5, colW, rowH: 0.36, fontSize: 11, fontFace: "Calibri", color: HEX.dk1,
    border: { type: "solid", pt: 0.5, color: "D5DCE6" }, valign: "middle", margin: 0.02, objectName: `table-${key}` });
  s.addText("† reported in the original PVCGN paper (Liu et al., 2022), same test split, not re-run.  HA: historical average · SN: seasonal naive · Bold: best in row · Teal columns: our trained models (mean over seeds).",
    { x: 0.4, y: 6.15, w: 12.5, h: 0.5, fontSize: 11, color: HEX.dk2, margin: 0, isTextBox: true, objectName: `note-${key}` });
  s.addNotes(note);
});

// R3. What we learned
s = content("What the ablation and error analysis show", "Results");
const learn = [
  ["Periodic inputs carry the gain", "+P alone: test MAE " + f2(mae("hz", "+P")) + " on Hangzhou — the lowest error under equal (200-epoch) training.", C.accent2],
  ["Calendar part adds nothing", "+P+T (" + f2(mae("hz", MAIN)) + ") is not better than +P. Reported honestly in the paper.", C.accent4],
  ["Wins where it matters", "Rush hours and busiest stations, both cities; Shanghai working days " + f2(mae("sh", MAIN, "working days")) + " vs " + f2(mae("sh", "PVCGN", "working days")) + ".", C.accent1],
  ["Weak on irregular days", "Holiday " + f2(mae("sh", MAIN, "holiday (Sep 15-17)")) + " vs PVCGN " + f2(mae("sh", "PVCGN", "holiday (Sep 15-17)")) + "; make-up Sunday " + f2(mae("sh", MAIN, "make-up workday (Sep 18)")) + " vs " + f2(mae("sh", "PVCGN", "make-up workday (Sep 18)")) + ".", C.accent3],
];
learn.forEach(([t, b, col], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.65;
  card(s, x, y, 5.95, 2.45, t, b, { dot: col });
});
s.addNotes("The main model was fixed on validation before the test, so PT-PVCGN stays the named model even though +P alone is slightly better. On holidays and make-up working days the reference days are ordinary days, which misleads the model.");

// R4. Conclusion and next steps
s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Results" });
s.addText("Conclusion", { placeholder: "title" });
s.addText("Giving PVCGN the same time slots on comparable past days cuts forecast error by " + pct(mae("hz", "PVCGN"), mae("hz", MAIN)) + "% (Hangzhou) and " + pct(mae("sh", "PVCGN"), mae("sh", MAIN)) + "% (Shanghai). Next: holiday-aware reference days, longer Shanghai training, and submission of the paper.", { placeholder: "body" });
for (let i = 0; i < 6; i++) dot(s, 0.8 + i * 0.55, 1.4, 0.28, "", i === 5 ? C.accent2 : C.accent1);
s.addNotes("Limitations to mention: Shanghai trained 30 epochs only; Hangzhou has one month of data and a weekend-only validation set; calendar embeddings did not help; irregular days are the main failure mode.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
