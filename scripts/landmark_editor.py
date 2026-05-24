"""Manual landmark + measurement editor for front+side silhouettes.

Schema (v2):
  {
    "lines_y":  {name: {"front": Y_px, "side": Y_px}, ...}
      # Horizontal lines (shoulder, armpit, bust, underbust, waist,
      # highhip, hip, crotch). Y in original-image pixels per view.
    "front_points": {"bust_apex_L":[x,y], "bust_apex_R":[x,y]}
    "center_axis_x_front": X_px           # vertical body axis on front
    "measurements": {bust, highbust, waist, highhip, hip, armscye, …}
    "image_size": {"front":[W,H], "side":[W,H]}
  }

Launch:
  python scripts/landmark_editor.py \
      --front-seg ..._sapiens/out/front_seg.npy \
      --side-seg  ..._sapiens/out/side_seg.npy \
      --out       data/results/PAIR_landmarks.json
"""
from __future__ import annotations
import argparse, base64, io, json, sys, threading, webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request


# Horizontal-line landmarks (Y level per view). Top-down anatomical order.
Y_LANDMARKS = [
    "jawline", "neck", "shoulder", "armpit",
    "bust", "underbust",
    "waist", "highhip", "hip", "crotch",
    "knee",
]
# Front-only point landmarks (X,Y per point)
FRONT_POINTS = ["bust_apex_L", "bust_apex_R"]
# Measurement fields (label → seamly code if any)
MEASUREMENT_FIELDS = [
    ("height",     "A01", "cm"),
    ("neck",       "G02", "cm"),
    ("highbust",   "G03", "cm"),
    ("bust",       "G04", "cm"),
    ("underbust",  "G05", "cm"),
    ("waist",      "G07", "cm"),
    ("highhip",    "G08", "cm"),
    ("hip",        "G09", "cm"),
    ("bicep",      "L11", "cm"),
    ("thigh",      "M03", "cm"),
    ("knee_circ",  "M05", "cm"),
]


HTML = r"""\
<!doctype html>
<html><head><title>Landmark editor v2</title><style>
body { font-family: sans-serif; background: #222; color: #eee; margin: 0; padding: 12px; }
h1 { margin: 0 0 10px 0; font-size: 16px; }
.row { display: flex; gap: 12px; }
.view { background: #111; padding: 6px; border-radius: 6px; }
.view h2 { margin: 0 0 4px 0; font-size: 12px; color: #aaa; }
canvas { display: block; border: 1px solid #444; cursor: crosshair; }
#panel { width: 240px; flex-shrink: 0; }
#panel h2 { margin: 12px 0 4px 0; font-size: 13px; color: #aaa; }
.lm { padding: 3px 6px; cursor: pointer; border-radius: 3px;
      display: flex; justify-content: space-between; font-size: 12px; }
.lm:hover { background: #333; }
.lm.active { background: #064; }
.lm .coords { color: #888; font-size: 10px; }
button { background: #4a4; border: 0; color: #fff; padding: 5px 10px;
         border-radius: 3px; cursor: pointer; margin: 4px 4px 0 0; font-size: 12px; }
button:hover { background: #5b5; }
button.danger { background: #844; }
button.danger:hover { background: #955; }
button.small { padding: 2px 6px; font-size: 11px; }
#status { margin-top: 8px; padding: 5px; background: #333;
          border-radius: 3px; font-size: 11px; min-height: 14px; }
.mrow { display: flex; gap: 4px; margin: 2px 0; align-items: center; font-size: 11px; }
.mrow label { width: 70px; color: #ccc; }
.mrow input { width: 60px; background: #333; color: #fff; border: 1px solid #555;
              padding: 2px 4px; border-radius: 2px; font-size: 11px; }
.mrow .code { color: #777; width: 30px; font-size: 10px; }
.opt { display: flex; gap: 6px; margin-top: 6px; align-items: center; font-size: 11px; }
.opt input[type="number"] { width: 50px; }
</style></head><body>
<h1>Landmark + measurement editor</h1>
<div class="row">
  <div id="panel">
    <h2>Active landmark</h2>
    <div id="landmarks"></div>

    <h2>Measurements (cm)</h2>
    <div id="measurements"></div>

    <h2>Options</h2>
    <label class="opt"><input type="checkbox" id="mirror-apex" checked>
      Mirror bust apex L↔R</label>

    <h2>Actions</h2>
    <button id="save">Save JSON</button>
    <button id="clear-current" class="danger small">Clear active</button>
    <button id="clear-all" class="danger small">Clear all</button>
    <div id="status"></div>
  </div>
  <div class="view"><h2>FRONT</h2><canvas id="cv-front"></canvas></div>
  <div class="view"><h2>SIDE</h2><canvas id="cv-side"></canvas></div>
</div>
<script>
const Y_NAMES = {{ y_names|tojson }};
const FRONT_POINTS = {{ front_points|tojson }};
const MEAS_FIELDS = {{ meas_fields|tojson }};
const IMG = {{ images|tojson }};
const BBOX = {{ bboxes|tojson }};
const PRIOR = {{ prior|tojson }};

// State
let active = Y_NAMES[0];           // active landmark name (in Y_NAMES, FRONT_POINTS or "center_axis")
let lines = {};                    // name -> {front: y, side: y}
let frontPoints = {};              // name -> [x, y]
let centerAxisX = null;
let measurements = {};
let dragging = null;               // {view, name} when dragging a line/axis

// Load prior
(function(){
  for (const k of Y_NAMES) {
    const e = PRIOR.lines_y?.[k];
    if (e) lines[k] = {...e};
  }
  for (const k of FRONT_POINTS) {
    const e = PRIOR.front_points?.[k];
    if (e) frontPoints[k] = [...e];
  }
  centerAxisX = PRIOR.center_axis_x_front ?? Math.round(BBOX.front.xc);
  measurements = {...(PRIOR.measurements || {})};
})();

const COLOR = {
  "neck":"#fb4","shoulder":"#f44","armpit":"#f88",
  "bust":"#ff4","underbust":"#cf4",
  "waist":"#4fc","highhip":"#48f","hip":"#84f","crotch":"#f8f",
  "bust_apex_L":"#fa4","bust_apex_R":"#fa4",
  "center_axis":"#4af",
};

// Canvas setup
const canvases = {}, images = {};
for (const v of ["front","side"]) {
  const cv = document.getElementById("cv-"+v); canvases[v] = cv;
  const img = new Image();
  img.onload = () => {
    const scale = Math.min(420/img.width, 760/img.height);
    cv.width = img.width * scale; cv.height = img.height * scale;
    cv.dataset.scale = scale; cv.dataset.imgw = img.width; cv.dataset.imgh = img.height;
    draw(v);
  };
  img.src = "data:image/png;base64," + IMG[v];
  images[v] = img;
  cv.addEventListener("mousedown", (e) => onMouseDown(v, e));
  cv.addEventListener("mousemove", (e) => onMouseMove(v, e));
  cv.addEventListener("mouseup",   (e) => onMouseUp(v, e));
  cv.addEventListener("mouseleave",(e) => onMouseUp(v, e));
}

function imgCoord(v, e) {
  const cv = canvases[v]; const r = cv.getBoundingClientRect();
  const s = parseFloat(cv.dataset.scale);
  return {x: (e.clientX - r.left) / s, y: (e.clientY - r.top) / s};
}

function nearestHandle(v, x, y, tol_px) {
  // Returns {kind: "line"|"axis"|"point", name} or null
  const tol = tol_px / parseFloat(canvases[v].dataset.scale);
  // Center axis (front only)
  if (v === "front" && centerAxisX != null && Math.abs(x - centerAxisX) < tol)
    return {kind: "axis", name: "center_axis"};
  // Lines (closest Y)
  for (const n of Y_NAMES) {
    const ly = lines[n]?.[v];
    if (ly != null && Math.abs(y - ly) < tol)
      return {kind: "line", name: n};
  }
  // Front points
  if (v === "front") {
    for (const n of FRONT_POINTS) {
      const p = frontPoints[n];
      if (p && Math.hypot(x - p[0], y - p[1]) < tol*1.5)
        return {kind: "point", name: n};
    }
  }
  return null;
}

function onMouseDown(v, e) {
  const {x, y} = imgCoord(v, e);
  const hit = nearestHandle(v, x, y, 8);
  if (hit) {
    active = hit.name;
    dragging = {view: v, kind: hit.kind, name: hit.name};
    refresh();
  } else {
    // Place active landmark
    placeActive(v, x, y);
  }
}

function onMouseMove(v, e) {
  if (!dragging || dragging.view !== v) return;
  const {x, y} = imgCoord(v, e);
  applyDrag(dragging, x, y);
  draw(v);
}

function onMouseUp(v, e) {
  if (dragging) { dragging = null; refresh(); }
}

function applyDrag(d, x, y) {
  if (d.kind === "axis") centerAxisX = Math.round(x);
  else if (d.kind === "line") {
    lines[d.name] = lines[d.name] || {};
    lines[d.name][d.view] = Math.round(y);
  } else if (d.kind === "point") {
    frontPoints[d.name] = [Math.round(x), Math.round(y)];
  }
}

function placeActive(v, x, y) {
  if (active === "center_axis") {
    if (v === "front") centerAxisX = Math.round(x);
  } else if (FRONT_POINTS.includes(active)) {
    if (v !== "front") return;  // points are front-only
    frontPoints[active] = [Math.round(x), Math.round(y)];
    // Optional mirror
    if (document.getElementById("mirror-apex").checked && centerAxisX != null) {
      const other = active === "bust_apex_L" ? "bust_apex_R" : "bust_apex_L";
      const mx = 2 * centerAxisX - x;
      frontPoints[other] = [Math.round(mx), Math.round(y)];
    }
  } else if (Y_NAMES.includes(active)) {
    lines[active] = lines[active] || {};
    lines[active][v] = Math.round(y);
    // Auto-sync Y to other view at same body fraction
    const other = v === "front" ? "side" : "front";
    if (lines[active][other] == null) {
      const src = BBOX[v], tgt = BBOX[other];
      const y_frac = (y - src.y0) / (src.y1 - src.y0);
      lines[active][other] = Math.round(tgt.y0 + y_frac * (tgt.y1 - tgt.y0));
    }
  }
  refresh();
}

function draw(v) {
  const cv = canvases[v]; const ctx = cv.getContext("2d");
  ctx.drawImage(images[v], 0, 0, cv.width, cv.height);
  const s = parseFloat(cv.dataset.scale);
  // Lines
  for (const n of Y_NAMES) {
    const ly = lines[n]?.[v];
    if (ly == null) continue;
    ctx.strokeStyle = COLOR[n] || "#fff";
    ctx.lineWidth = (n === active) ? 3 : 1.5;
    ctx.beginPath(); ctx.moveTo(0, ly*s); ctx.lineTo(cv.width, ly*s); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = "11px sans-serif";
    ctx.fillText(n, 4, ly*s - 3);
  }
  // Center axis (front)
  if (v === "front" && centerAxisX != null) {
    ctx.strokeStyle = COLOR["center_axis"];
    ctx.lineWidth = (active === "center_axis") ? 3 : 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(centerAxisX*s, 0); ctx.lineTo(centerAxisX*s, cv.height); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ctx.strokeStyle; ctx.fillText("axis", centerAxisX*s + 4, 12);
  }
  // Front points
  if (v === "front") {
    for (const n of FRONT_POINTS) {
      const p = frontPoints[n]; if (!p) continue;
      ctx.fillStyle = COLOR[n]; ctx.strokeStyle = "#000"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p[0]*s, p[1]*s, 6, 0, 2*Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.fillText(n, p[0]*s + 8, p[1]*s + 4);
    }
  }
}

function refresh() {
  draw("front"); draw("side");
  // Landmark panel
  const lp = document.getElementById("landmarks");
  lp.innerHTML = "";
  const items = [...Y_NAMES, ...FRONT_POINTS, "center_axis"];
  for (const n of items) {
    const div = document.createElement("div");
    div.className = "lm";
    if (n === active) div.classList.add("active");
    let status = "";
    if (Y_NAMES.includes(n)) {
      const l = lines[n] || {};
      status = (l.front != null ? "F" : "·") + (l.side != null ? "S" : "·");
    } else if (FRONT_POINTS.includes(n)) {
      status = frontPoints[n] ? "F·" : "··";
    } else if (n === "center_axis") {
      status = centerAxisX != null ? "F·" : "··";
    }
    const c = COLOR[n] || "#fff";
    div.innerHTML = `<span><span style="color:${c}">●</span> ${n}</span>` +
      `<span class="coords">${status}</span>`;
    div.onclick = () => { active = n; refresh(); };
    lp.appendChild(div);
  }
  // Measurements
  const mp = document.getElementById("measurements");
  mp.innerHTML = "";
  for (const [name, code, unit] of MEAS_FIELDS) {
    const row = document.createElement("div"); row.className = "mrow";
    const v = measurements[name];
    row.innerHTML = `<label>${name}</label>` +
      `<input type="number" step="0.1" data-name="${name}" value="${v ?? ''}" placeholder="—">` +
      `<span class="code">${code}</span>`;
    mp.appendChild(row);
  }
  for (const el of mp.querySelectorAll("input")) {
    el.onchange = () => {
      const n = el.dataset.name;
      const v = el.value.trim();
      if (v === "") delete measurements[n];
      else measurements[n] = parseFloat(v);
    };
  }
}

document.getElementById("save").onclick = async () => {
  const payload = {
    lines_y: lines,
    front_points: frontPoints,
    center_axis_x_front: centerAxisX,
    measurements,
  };
  const r = await fetch("/save", {method:"POST",
    headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
  const j = await r.json();
  document.getElementById("status").textContent = j.msg;
};
document.getElementById("clear-current").onclick = () => {
  if (Y_NAMES.includes(active)) delete lines[active];
  else if (FRONT_POINTS.includes(active)) delete frontPoints[active];
  else if (active === "center_axis") centerAxisX = null;
  refresh();
};
document.getElementById("clear-all").onclick = () => {
  lines = {}; frontPoints = {}; centerAxisX = Math.round(BBOX.front.xc);
  refresh();
};

refresh();
</script>
</body></html>
"""


def encode_silhouette_png(seg_path: str, drop_classes=(),
                          photo_path: str | None = None,
                          alpha: float = 0.40
                          ) -> tuple[str, tuple[int,int], dict]:
    seg = np.load(seg_path)
    if seg.ndim == 3:
        seg = (seg.argmax(0) if seg.shape[0] < seg.shape[-1]
               else seg.argmax(-1))
    mask = seg > 0
    for c in drop_classes:
        mask &= seg != c
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n > 1:
        best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        mask = lab == best
    H, W = mask.shape

    if photo_path:
        photo = cv2.imread(photo_path)
        if photo is None:
            raise FileNotFoundError(f"could not read photo: {photo_path}")
        if photo.shape[:2] != (H, W):
            photo = cv2.resize(photo, (W, H), interpolation=cv2.INTER_AREA)
        cyan = np.zeros_like(photo); cyan[:, :] = (255, 255, 0)
        blended = (photo.astype(np.float32) * (1 - alpha) +
                   cyan.astype(np.float32) * alpha).astype(np.uint8)
        dim = (photo.astype(np.float32) * 0.4).astype(np.uint8)
        img = np.where(mask[..., None], blended, dim)
    else:
        img = (mask.astype(np.uint8) * 255)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"png encode failed: {seg_path}")
    ys, xs = np.where(mask)
    bbox = {"x0": int(xs.min()), "x1": int(xs.max()),
            "y0": int(ys.min()), "y1": int(ys.max()),
            "xc": int((xs.min()+xs.max())/2),
            "yc": int((ys.min()+ys.max())/2)}
    return base64.b64encode(buf).decode("ascii"), (W, H), bbox


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--front-seg",   type=Path, required=True)
    ap.add_argument("--side-seg",    type=Path, required=True)
    ap.add_argument("--front-photo", type=Path, default=None)
    ap.add_argument("--side-photo",  type=Path, default=None)
    ap.add_argument("--out",         type=Path, required=True)
    ap.add_argument("--port",        type=int,  default=5050)
    ap.add_argument("--host",        default="127.0.0.1")
    ap.add_argument("--no-open",     action="store_true")
    ap.add_argument("--front-drop-classes", type=int, nargs="*",
                    default=[6, 7, 11, 15, 20])
    ap.add_argument("--side-drop-classes",  type=int, nargs="*",
                    default=[1])
    args = ap.parse_args(argv)

    def auto_photo(seg_path: Path, hint: str) -> Path | None:
        cand = seg_path.parent.parent / "in" / f"{hint}.jpg"
        return cand if cand.exists() else None
    front_photo = args.front_photo or auto_photo(args.front_seg, "front")
    side_photo  = args.side_photo  or auto_photo(args.side_seg,  "side")
    if front_photo: print(f"front photo: {front_photo}")
    if side_photo:  print(f"side  photo: {side_photo}")

    front_b64, front_size, front_bbox = encode_silhouette_png(
        str(args.front_seg), tuple(args.front_drop_classes),
        photo_path=str(front_photo) if front_photo else None)
    side_b64, side_size, side_bbox = encode_silhouette_png(
        str(args.side_seg), tuple(args.side_drop_classes),
        photo_path=str(side_photo) if side_photo else None)

    # Load prior (try v2 then v1 fallback)
    prior = {"lines_y": {}, "front_points": {},
             "center_axis_x_front": None, "measurements": {}}
    if args.out.exists():
        try:
            existing = json.loads(args.out.read_text())
            # v2 keys
            if "lines_y" in existing:
                prior.update({k: existing.get(k, prior[k]) for k in prior})
                print(f"loaded v2 landmarks from {args.out}")
            else:  # v1 → migrate Y from side points
                print(f"migrating v1 → v2 from {args.out}")
                for view in ("front", "side"):
                    for n, xy in (existing.get(view, {}) or {}).items():
                        if not xy: continue
                        if n in Y_LANDMARKS:
                            prior["lines_y"].setdefault(n, {})[view] = int(xy[1])
                        elif n in ("shoulder_L", "shoulder_R"):
                            # treat as shoulder line
                            prior["lines_y"].setdefault("shoulder", {})[view] = int(xy[1])
        except Exception as e:
            print(f"warn: failed to load existing {args.out}: {e}")

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            HTML, y_names=Y_LANDMARKS, front_points=FRONT_POINTS,
            meas_fields=MEASUREMENT_FIELDS, prior=prior,
            images={"front": front_b64, "side": side_b64},
            bboxes={"front": front_bbox, "side": side_bbox})

    @app.route("/save", methods=["POST"])
    def save():
        data = request.get_json()
        out = {
            "lines_y":             data.get("lines_y", {}),
            "front_points":        data.get("front_points", {}),
            "center_axis_x_front": data.get("center_axis_x_front"),
            "measurements":        data.get("measurements", {}),
            "image_size": {"front": list(front_size),
                           "side":  list(side_size)},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        nl = sum(1 for v in out["lines_y"].values() if v)
        np_ = sum(1 for v in out["front_points"].values() if v)
        nm = sum(1 for v in out["measurements"].values() if v is not None)
        msg = (f"saved {nl} lines + {np_} apex points + "
               f"{nm} measurements to {args.out}")
        print(msg)
        return jsonify(msg=msg)

    url = f"http://{args.host}:{args.port}/"
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"Landmark editor v2 → {url}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
