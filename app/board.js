// Drag-and-drop tactics board (Streamlit components v2, no build step).
//
// data (from Python): { bg, width, height, bounds, players: [{id, kind, px, py}],
//                       version, mode: "move" | "add", add_kind }
// state (to Python):  players = { version, items: [{id, kind, px, py}] }
//
// The board keeps its own copy of the players and only replaces it when `version` changes
// (scenario loaded / reset), so a rerun triggered by a drag never fights with the drag.
// Move mode: drag players. Add mode: click on empty grass to add a player of `add_kind`.
// Both modes: double-click (or right-click) a player to remove it; the shooter cannot be removed.

const NS = "http://www.w3.org/2000/svg";
const R = 7; // player radius in px (~0.9 yd)
const FILL = { shooter: "#ffd400", teammate: "#1f77b4", defender: "#d62728", goalkeeper: "#111111" };
const LABEL = { shooter: "shooter", teammate: "teammate", defender: "defender", goalkeeper: "goalkeeper" };

function starPoints(r) {
  const pts = [];
  for (let k = 0; k < 10; k++) {
    const a = -Math.PI / 2 + (k * Math.PI) / 5;
    const rr = k % 2 === 0 ? r * 1.6 : r * 0.7;
    pts.push(`${(rr * Math.cos(a)).toFixed(2)},${(rr * Math.sin(a)).toFixed(2)}`);
  }
  return pts.join(" ");
}

function shape(kind) {
  let el;
  if (kind === "shooter") {
    el = document.createElementNS(NS, "polygon");
    el.setAttribute("points", starPoints(R));
  } else if (kind === "goalkeeper") {
    el = document.createElementNS(NS, "rect");
    el.setAttribute("x", -R); el.setAttribute("y", -R);
    el.setAttribute("width", 2 * R); el.setAttribute("height", 2 * R);
  } else {
    el = document.createElementNS(NS, "circle");
    el.setAttribute("r", R);
  }
  el.setAttribute("fill", FILL[kind]);
  el.setAttribute("stroke", kind === "goalkeeper" ? "#ffffff" : "#000000");
  el.setAttribute("stroke-width", "1.5");
  return el;
}

function draw(s) {
  const layer = s.layer;
  while (layer.firstChild) layer.removeChild(layer.firstChild);
  // shooter last so it is on top
  const ordered = [...s.players].sort((a, b) => (a.kind === "shooter") - (b.kind === "shooter"));
  for (const p of ordered) {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "xgb-player");
    g.dataset.id = p.id;
    g.setAttribute("transform", `translate(${p.px},${p.py})`);
    const title = document.createElementNS(NS, "title");
    title.textContent = LABEL[p.kind] + (p.kind === "shooter" ? "" : " (double-click to remove)");
    g.appendChild(title);
    g.appendChild(shape(p.kind));
    layer.appendChild(g);
  }
}

function eventPx(s, e) {
  const r = s.svg.getBoundingClientRect();
  return [(e.clientX - r.left) * (s.data.width / r.width), (e.clientY - r.top) * (s.data.height / r.height)];
}

function clamp(s, px, py) {
  const b = s.data.bounds;
  return [Math.min(Math.max(px, b.left), b.right), Math.min(Math.max(py, b.top), b.bottom)];
}

function commit(s) {
  s.setStateValue("players", {
    version: s.version,
    items: s.players.map((p) => ({ id: p.id, kind: p.kind, px: p.px, py: p.py })),
  });
}

function playerAt(s, e) {
  const g = e.target.closest ? e.target.closest(".xgb-player") : null;
  return g ? s.players.find((p) => p.id === g.dataset.id) : null;
}

function remove(s, p) {
  if (!p || p.kind === "shooter") return;
  s.players = s.players.filter((q) => q !== p);
  draw(s);
  commit(s);
}

function attach(s) {
  const svg = s.svg;
  svg.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    const p = playerAt(s, e);
    if (s.data.mode === "move") {
      if (!p) return;
      s.drag = { p, moved: false };
      svg.setPointerCapture(e.pointerId);
      e.preventDefault();
    } else if (!p) {
      const [px, py] = clamp(s, ...eventPx(s, e));
      s.players.push({ id: `n${s.nextId++}`, kind: s.data.add_kind, px, py });
      draw(s);
      commit(s);
    }
  });
  svg.addEventListener("pointermove", (e) => {
    if (!s.drag) return;
    const [px, py] = clamp(s, ...eventPx(s, e));
    s.drag.p.px = px; s.drag.p.py = py; s.drag.moved = true;
    const g = s.layer.querySelector(`[data-id="${s.drag.p.id}"]`);
    if (g) g.setAttribute("transform", `translate(${px},${py})`);
  });
  const end = (e) => {
    if (!s.drag) return;
    const moved = s.drag.moved;
    s.drag = null;
    try { svg.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
    if (moved) commit(s);
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("dblclick", (e) => remove(s, playerAt(s, e)));
  svg.addEventListener("contextmenu", (e) => {
    const p = playerAt(s, e);
    if (p) { e.preventDefault(); remove(s, p); }
  });
}

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let root = parentElement.querySelector(".xgb-root");
  if (!root) {
    root = document.createElement("div");
    root.className = "xgb-root";
    const img = document.createElement("img");
    img.className = "xgb-bg";
    img.draggable = false;
    img.alt = "pitch";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "xgb-svg");
    const layer = document.createElementNS(NS, "g");
    svg.appendChild(layer);
    root.append(img, svg);
    parentElement.appendChild(root);
    root._s = { img, svg, layer, players: [], version: null, drag: null, nextId: 1, bg: null };
    attach(root._s);
  }
  const s = root._s;
  s.data = data;
  s.setStateValue = setStateValue;
  // responsive: at most the native size, same aspect ratio; eventPx() rescales pointer positions
  root.style.width = "100%";
  root.style.maxWidth = `${data.width}px`;
  root.style.aspectRatio = `${data.width} / ${data.height}`;
  s.svg.setAttribute("viewBox", `0 0 ${data.width} ${data.height}`);
  s.svg.setAttribute("width", data.width);
  s.svg.setAttribute("height", data.height);
  if (s.bg !== data.bg) { s.bg = data.bg; s.img.src = data.bg; }
  root.dataset.mode = data.mode;
  if (data.version !== s.version && !s.drag) {
    s.version = data.version;
    s.players = data.players.map((p) => ({ ...p }));
    draw(s);
  }
}
