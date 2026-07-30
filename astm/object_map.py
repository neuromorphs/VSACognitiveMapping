"""Constructive object map: place detected objects at metric locations, bind
them into one VSA hypervector, and query the map live in a browser.

This is the step past ``classroom_pipeline.py``: instead of binding a whole
frame to a *known* pose and recalling it, we (1) compute a metric world
location for every YOLO detection from its box + D455 depth + robot pose,
(2) cluster detections into unique objects, and (3) build a spatial memory

        M = sum_objects  SEM_class  (x)  ctx_pos(x, y)

that answers the two questions a map should:

    "where are the chairs?"  ->  M (/) SEM_chair   -> similarity field over the
                                  floor, peaks at each chair.
    "what is at (x, y)?"     ->  M (/) ctx_pos(x,y) -> ranked class list.

Stages:
    localize   detections.csv + depth + poses -> object_map.json
    serve      load object_map.json, build the VSA map, run a local web app

Examples:
    python vsa_cognitive_mapping/object_map.py localize
    python vsa_cognitive_mapping/object_map.py serve --port 8010

`localize` needs the `depth_d455` config downloaded (HF cache) and the
`detections.csv` produced by `classroom_pipeline.py embed`. `serve` only needs
`object_map.json`, so the browser demo runs offline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import Phasor  # noqa: E402
from vsa_cognitive_mapping.classroom_pipeline import (  # noqa: E402
    DATASET, quat_to_yaw, load_poses_interpolated,
)

DEFAULT_OUT = os.path.join("outputs", "classroom")
# RealSense D455 colour intrinsics (from camera_info); depth assumed registered.
K_FX, K_FY, K_CX, K_CY = 378.22, 377.92, 320.44, 249.25
# Approximate camera mount on Spot body (metres): a bit forward and up.
CAM_OFFSET = np.array([0.30, 0.0, 0.20])


# --------------------------------------------------------------------------
# localize
# --------------------------------------------------------------------------

def _read_detections(out_dir):
    rows = []
    with open(os.path.join(out_dir, "detections.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["timestamp_ns"]), int(r["class_id"]), r["class_name"],
                         float(r["confidence"]), float(r["x1"]), float(r["y1"]),
                         float(r["x2"]), float(r["y2"])))
    return rows


def _pixel_to_world(u, v, z_m, px, py, yaw):
    """Box-centre pixel + depth -> world (x, y, z) via optical->body->world."""
    X = (u - K_CX) / K_FX * z_m
    Y = (v - K_CY) / K_FY * z_m
    Z = z_m
    # optical (x right, y down, z fwd) -> body (x fwd, y left, z up)
    body = np.array([Z, -X, -Y]) + CAM_OFFSET
    c, s = np.cos(yaw), np.sin(yaw)
    xw = px + c * body[0] - s * body[1]
    yw = py + s * body[0] + c * body[1]
    return xw, yw, body[2]


def _cluster(points, radius, min_count):
    """Greedy proximity clustering of (x, y, conf) points for one class."""
    pts = np.array([[p[0], p[1]] for p in points])
    conf = np.array([p[2] for p in points])
    used = np.zeros(len(pts), bool)
    clusters = []
    order = np.argsort(-conf)  # seed from most confident
    for i in order:
        if used[i]:
            continue
        d = np.linalg.norm(pts - pts[i], axis=1)
        member = (d <= radius) & (~used)
        used |= member
        if member.sum() >= min_count:
            clusters.append({
                "x": float(np.median(pts[member, 0])),
                "y": float(np.median(pts[member, 1])),
                "count": int(member.sum()),
                "conf": float(conf[member].mean()),
            })
    return clusters


def cmd_localize(args):
    from datasets import load_dataset

    dets = _read_detections(args.out_dir)
    det_ts = np.array([d[0] for d in dets], np.int64)
    x, y, yaw = load_poses_interpolated(det_ts)

    depth = load_dataset(DATASET, "depth_d455", split="train")
    dts = np.array(depth["timestamp_ns"], np.int64)
    dorder = np.argsort(dts)
    dts_sorted = dts[dorder]

    # nearest depth row per detection, grouped so each depth frame decodes once
    idx = np.searchsorted(dts_sorted, det_ts)
    idx = np.clip(idx, 0, len(dts_sorted) - 1)
    left = np.clip(idx - 1, 0, len(dts_sorted) - 1)
    take_left = np.abs(dts_sorted[left] - det_ts) < np.abs(dts_sorted[idx] - det_ts)
    nearest = np.where(take_left, left, idx)
    det_depth_row = dorder[nearest]

    by_frame = defaultdict(list)
    for di, drow in enumerate(det_depth_row):
        by_frame[int(drow)].append(di)

    per_class = defaultdict(list)
    n_ok = 0
    print(f"localizing {len(dets)} detections over {len(by_frame)} depth frames ...")
    for fi, (drow, det_ids) in enumerate(by_frame.items()):
        dimg = np.array(depth[drow]["depth"]).astype(np.float32)  # uint16 mm
        for di in det_ids:
            ts, cid, cname, conf, x1, y1, x2, y2 = dets[di]
            u1, u2 = int(max(0, x1)), int(min(dimg.shape[1], x2))
            v1, v2 = int(max(0, y1)), int(min(dimg.shape[0], y2))
            if u2 <= u1 or v2 <= v1:
                continue
            patch = dimg[v1:v2, u1:u2]
            valid = patch[(patch > 200) & (patch < 8000)]  # 0.2 - 8 m
            if valid.size < max(10, 0.02 * patch.size):
                continue
            z_m = float(np.median(valid)) / 1000.0
            u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            xw, yw, zw = _pixel_to_world(u, v, z_m, x[di], y[di], yaw[di])
            per_class[cname].append((xw, yw, conf))
            n_ok += 1
        if fi % 200 == 0:
            print(f"  frame {fi}/{len(by_frame)}  placed={n_ok}")

    objects = []
    for cname, pts in per_class.items():
        for c in _cluster(pts, args.cluster_radius, args.min_count):
            c["class"] = cname
            objects.append(c)
    objects.sort(key=lambda o: (-o["count"]))

    # trajectory (raw LIO-SAM poses) for the map background
    odo = load_dataset(DATASET, "odometry_lio_sam", split="train")
    ot = np.array(odo["timestamp_ns"], np.float64)
    oo = np.argsort(ot)
    traj = np.stack([np.array(odo["x"])[oo], np.array(odo["y"])[oo]], 1)
    traj = traj[:: max(1, len(traj) // 400)]

    allx = np.array([o["x"] for o in objects] + list(traj[:, 0]))
    ally = np.array([o["y"] for o in objects] + list(traj[:, 1]))
    out = {
        "objects": objects,
        "classes": sorted({o["class"] for o in objects}),
        "trajectory": [[float(a), float(b)] for a, b in traj],
        "bounds": {"xmin": float(allx.min()), "xmax": float(allx.max()),
                   "ymin": float(ally.min()), "ymax": float(ally.max())},
        "n_placed": n_ok,
    }
    path = os.path.join(args.out_dir, "object_map.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nplaced {n_ok} detections -> {len(objects)} objects across "
          f"{len(out['classes'])} classes")
    for o in objects:
        print(f"  {o['class']:>14}  ({o['x']:+5.2f}, {o['y']:+5.2f})  n={o['count']:3d}")
    print(f"saved {path}")


# --------------------------------------------------------------------------
# VSA map
# --------------------------------------------------------------------------

class ObjectMap:
    """The VSA spatial memory built from placed objects, with live decoders."""

    def __init__(self, data, hd_dim=4096, seed=0, length_scale=0.6, grid=110):
        self.data = data
        self.hd = hd_dim
        self.L = length_scale
        self.classes = data["classes"]
        self.Bx = Phasor(dim=hd_dim, seed=seed + 1)
        self.By = Phasor(dim=hd_dim, seed=seed + 2)
        self.SEM = {c: Phasor(dim=hd_dim, seed=seed + 100 + i)
                    for i, c in enumerate(self.classes)}

        M = np.zeros(hd_dim, np.complex128)
        for o in data["objects"]:
            M += (self.SEM[o["class"]] * self._ctx(o["x"], o["y"])).values
        self.M = M / max(1, len(data["objects"]))

        b = data["bounds"]
        pad = 1.0
        self.gx = np.linspace(b["xmin"] - pad, b["xmax"] + pad, grid)
        self.gy = np.linspace(b["ymin"] - pad, b["ymax"] + pad,
                              max(8, int(grid * (b["ymax"] - b["ymin"] + 2 * pad) /
                                         (b["xmax"] - b["xmin"] + 2 * pad))))
        CTX = np.empty((len(self.gy), len(self.gx), hd_dim), np.complex128)
        for j, yy in enumerate(self.gy):
            for i, xx in enumerate(self.gx):
                CTX[j, i] = self._ctx(xx, yy).values
        self.CTX_flat = CTX.reshape(-1, hd_dim)

    def _ctx(self, x, y):
        return (self.Bx ** float(x / self.L)) * (self.By ** float(y / self.L))

    def query_class(self, cname):
        residual = np.conj(self.M / self.SEM[cname].values)
        field = (np.real(self.CTX_flat @ residual) / self.hd).reshape(len(self.gy), len(self.gx))
        return field

    def query_point(self, x, y):
        residual = self.M / self._ctx(x, y).values
        scores = {c: float(np.real(np.sum(np.conj(self.SEM[c].values) * residual)) / self.hd)
                  for c in self.classes}
        return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------

def cmd_serve(args):
    import http.server
    import socketserver
    from urllib.parse import urlparse, parse_qs

    with open(os.path.join(args.out_dir, "object_map.json")) as f:
        data = json.load(f)
    omap = ObjectMap(data, hd_dim=args.hd_dim, length_scale=args.length_scale)
    print(f"map: {len(data['objects'])} objects, {len(data['classes'])} classes, "
          f"hd_dim={args.hd_dim}, grid {len(omap.gx)}x{len(omap.gy)}")

    page = _PAGE

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, obj, ctype="application/json"):
            body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif u.path == "/api/map":
                self._send({"objects": data["objects"], "classes": data["classes"],
                            "trajectory": data["trajectory"], "bounds": data["bounds"],
                            "grid": {"gx": omap.gx.tolist(), "gy": omap.gy.tolist()}})
            elif u.path == "/api/query_class":
                field = omap.query_class(q["class"][0])
                self._send({"z": field.tolist(),
                            "vmax": float(np.max(np.abs(field)))})
            elif u.path == "/api/query_point":
                res = omap.query_point(float(q["x"][0]), float(q["y"][0]))
                self._send({"ranked": [{"class": k, "score": v} for k, v in res.items()]})
            else:
                self.send_response(404)
                self.end_headers()

    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), Handler) as httpd:
        httpd.daemon_threads = True
        print(f"\n  VSA object map running at  http://localhost:{args.port}\n")
        print("  Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


# --------------------------------------------------------------------------

def cmd_snapshot(args):
    """Render a static PNG of the map (objects + trajectory + one class heat
    field) -- a non-interactive preview of what the server shows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.out_dir, "object_map.json")) as f:
        data = json.load(f)
    omap = ObjectMap(data, hd_dim=args.hd_dim, length_scale=args.length_scale)

    classes = data["classes"]
    palette = plt.cm.tab20(np.linspace(0, 1, max(20, len(classes))))
    col = {c: palette[i % len(palette)] for i, c in enumerate(classes)}
    traj = np.array(data["trajectory"])

    fig, ax = plt.subplots(figsize=(11, 9.5), constrained_layout=True)
    if args.query_class:
        field = omap.query_class(args.query_class)
        lim = np.max(np.abs(field)) or 1.0
        ax.imshow(np.clip(field, 0, None), origin="lower", cmap="cividis",
                  extent=[omap.gx[0], omap.gx[-1], omap.gy[0], omap.gy[-1]],
                  aspect="equal", alpha=0.85, vmin=0, vmax=lim)
    ax.plot(traj[:, 0], traj[:, 1], "-", color="0.6", lw=1.2, alpha=0.7, label="robot path")
    for o in data["objects"]:
        hot = (not args.query_class) or (o["class"] == args.query_class)
        ax.scatter([o["x"]], [o["y"]], s=40 + 7 * o["count"],
                   color=col[o["class"]] if hot else "0.6",
                   edgecolor="white", linewidth=0.7, zorder=3, alpha=0.95 if hot else 0.4)
        if hot:
            ax.annotate(o["class"], (o["x"], o["y"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    title = "VSA object map — 83 objects placed from YOLO + D455 depth"
    if args.query_class:
        title = f"query: 'where are the {args.query_class}s?'  —  M ⊘ SEM_{args.query_class}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal", "box")
    path = os.path.join(args.out_dir, "object_map_snapshot.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("localize", help="place detections in the world, cluster to objects")
    pl.add_argument("--cluster-radius", type=float, default=0.6, help="metres")
    pl.add_argument("--min-count", type=int, default=4, help="min detections to accept an object")

    ps = sub.add_parser("serve", help="run the interactive map server")
    ps.add_argument("--port", type=int, default=8010)
    ps.add_argument("--hd-dim", type=int, default=4096)
    ps.add_argument("--length-scale", type=float, default=0.6)

    pn = sub.add_parser("snapshot", help="render a static PNG preview of the map")
    pn.add_argument("--query-class", default=None, help="overlay this class's query heat field")
    pn.add_argument("--hd-dim", type=int, default=4096)
    pn.add_argument("--length-scale", type=float, default=0.6)

    for p in (pl, ps, pn):
        p.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    {"localize": cmd_localize, "serve": cmd_serve, "snapshot": cmd_snapshot}[args.cmd](args)


# --------------------------------------------------------------------------
# Front-end (served at /). Canvas map + live class / point queries.
# --------------------------------------------------------------------------

_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>VSA Object Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--bg:#0f1319;--panel:#161d27;--line:#26303c;--ink:#e7ecf2;--muted:#93a2b3;
        --accent:#2bb8c6;--accent2:#e0a02e;}
  *{box-sizing:border-box} html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--ink);
       font-family:-apple-system,"Segoe UI",system-ui,sans-serif;display:flex;height:100vh}
  .side{width:290px;flex:0 0 290px;background:var(--panel);border-right:1px solid var(--line);
        padding:20px;overflow-y:auto}
  .main{flex:1;position:relative;min-width:0}
  canvas{display:block;width:100%;height:100%}
  h1{font-size:16px;margin:0 0 4px;letter-spacing:-.01em}
  .sub{font-size:12px;color:var(--muted);margin:0 0 18px;line-height:1.5}
  .lab{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);
       margin:18px 0 8px;font-weight:600}
  .chip{display:flex;align-items:center;gap:9px;width:100%;text-align:left;
        background:transparent;border:1px solid var(--line);color:var(--ink);
        padding:7px 10px;border-radius:8px;margin-bottom:6px;cursor:pointer;font-size:13px;
        font-family:inherit}
  .chip:hover{border-color:var(--accent)}
  .chip.active{border-color:var(--accent);background:rgba(43,184,198,.12)}
  .dot{width:11px;height:11px;border-radius:50%;flex:0 0 auto}
  .cnt{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums;font-size:12px}
  .hint{font-size:12px;color:var(--muted);line-height:1.6;margin-top:6px}
  #readout{position:absolute;left:16px;bottom:16px;background:rgba(15,19,25,.92);
           border:1px solid var(--line);border-radius:10px;padding:12px 14px;max-width:280px;
           font-size:12.5px;display:none}
  #readout b{color:var(--accent)}
  .bar{height:6px;background:var(--line);border-radius:3px;margin:3px 0 8px;overflow:hidden}
  .bar>i{display:block;height:100%;background:var(--accent)}
  .legend{position:absolute;right:16px;top:16px;background:rgba(15,19,25,.85);
          border:1px solid var(--line);border-radius:8px;padding:8px 11px;font-size:11px;color:var(--muted)}
  code{background:#0b0f14;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px}
</style></head>
<body>
  <div class="side">
    <h1>VSA Object Map</h1>
    <p class="sub">Objects placed from YOLO + D455 depth, bound into one
      hypervector <code>M = &Sigma; SEM<sub>class</sub> &otimes; SSP(x,y)</code>.
      Queries below run a live unbind on the server.</p>

    <div class="lab">Where are the &hellip;? (class &rarr; place)</div>
    <div id="classes"></div>

    <div class="lab">What's here? (place &rarr; class)</div>
    <p class="hint">Click anywhere on the map. The server unbinds
      <code>M &oslash; SSP(x,y)</code> and ranks the classes.</p>
  </div>
  <div class="main">
    <canvas id="cv"></canvas>
    <div class="legend">○ objects &nbsp;·&nbsp; — trajectory &nbsp;·&nbsp; heat = class query</div>
    <div id="readout"></div>
  </div>
<script>
const PALETTE=["#2bb8c6","#e0a02e","#e0556f","#7d5be0","#43c383","#4a90d9","#d98c4a","#b45be0","#5bd0e0","#c0d94a"];
let MAP=null, colorOf={}, active=null, heat=null, view={};
const cv=document.getElementById("cv"), ctx=cv.getContext("2d");

async function boot(){
  MAP=await (await fetch("/api/map")).json();
  MAP.classes.forEach((c,i)=>colorOf[c]=PALETTE[i%PALETTE.length]);
  const box=document.getElementById("classes");
  const counts={}; MAP.objects.forEach(o=>counts[o.class]=(counts[o.class]||0)+1);
  MAP.classes.forEach(c=>{
    const b=document.createElement("button"); b.className="chip";
    b.innerHTML=`<span class="dot" style="background:${colorOf[c]}"></span>${c}<span class="cnt">${counts[c]}</span>`;
    b.onclick=()=>selectClass(c,b); box.appendChild(b);
  });
  resize(); window.addEventListener("resize",resize);
  cv.addEventListener("click",onClick);
}
function resize(){
  const r=cv.getBoundingClientRect(), dpr=devicePixelRatio||1;
  cv.width=r.width*dpr; cv.height=r.height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  const b=MAP.bounds, pad=1.0;
  const wx=(b.xmax-b.xmin)+2*pad, wy=(b.ymax-b.ymin)+2*pad;
  const s=Math.min(r.width/wx, r.height/wy)*0.95;
  view={s, ox:r.width/2-(b.xmin+b.xmax)/2*s, oy:r.height/2+(b.ymin+b.ymax)/2*s, w:r.width, h:r.height};
  draw();
}
const X=x=>view.ox+x*view.s, Y=y=>view.oy-y*view.s;
const invX=px=>(px-view.ox)/view.s, invY=py=>(view.oy-py)/view.s;

async function selectClass(c,btn){
  document.querySelectorAll(".chip").forEach(e=>e.classList.remove("active"));
  if(active===c){active=null;heat=null;draw();return;}
  active=c; btn.classList.add("active");
  const r=await (await fetch("/api/query_class?class="+encodeURIComponent(c))).json();
  heat={z:r.z, vmax:r.vmax, gx:MAP.grid.gx, gy:MAP.grid.gy};
  draw();
}
async function onClick(e){
  const r=cv.getBoundingClientRect();
  const wx=invX(e.clientX-r.left), wy=invY(e.clientY-r.top);
  const res=await (await fetch(`/api/query_point?x=${wx}&y=${wy}`)).json();
  const top=res.ranked.slice(0,5);
  const mx=Math.max(...top.map(t=>Math.abs(t.score)))||1;
  let html=`<b>What's at (${wx.toFixed(2)}, ${wy.toFixed(2)})?</b><br><br>`;
  top.forEach(t=>{html+=`${t.class} <span style="color:var(--muted)">${t.score.toFixed(3)}</span>
    <div class="bar"><i style="width:${Math.max(0,t.score)/mx*100}%;background:${colorOf[t.class]}"></i></div>`;});
  const ro=document.getElementById("readout"); ro.innerHTML=html; ro.style.display="block";
  view.pick=[wx,wy]; draw();
}
function draw(){
  ctx.clearRect(0,0,view.w,view.h);
  if(heat) drawHeat();
  // trajectory
  ctx.strokeStyle="rgba(147,162,179,.45)"; ctx.lineWidth=1.4; ctx.beginPath();
  MAP.trajectory.forEach((p,i)=>{const px=X(p[0]),py=Y(p[1]); i?ctx.lineTo(px,py):ctx.moveTo(px,py);});
  ctx.stroke();
  // objects
  MAP.objects.forEach(o=>{
    const px=X(o.x),py=Y(o.y), on=(!active||active===o.class);
    ctx.beginPath(); ctx.arc(px,py,on?6:4,0,7);
    ctx.fillStyle=on?colorOf[o.class]:"rgba(147,162,179,.35)"; ctx.fill();
    ctx.lineWidth=1.5; ctx.strokeStyle="rgba(15,19,25,.9)"; ctx.stroke();
    if(active===o.class){ctx.fillStyle="#e7ecf2";ctx.font="11px system-ui";ctx.fillText(o.class,px+8,py+3);}
  });
  if(view.pick){const px=X(view.pick[0]),py=Y(view.pick[1]);
    ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.beginPath();ctx.arc(px,py,9,0,7);ctx.stroke();}
}
function drawHeat(){
  const {z,vmax,gx,gy}=heat, ny=z.length, nx=z[0].length;
  const cw=(X(gx[1])-X(gx[0])), ch=(Y(gy[0])-Y(gy[1]));
  for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){
    const v=z[j][i]/(vmax||1); if(v<0.06)continue;
    ctx.fillStyle=`rgba(43,184,198,${Math.min(0.72,v*0.85)})`;
    ctx.fillRect(X(gx[i])-cw/2, Y(gy[j])-ch/2, cw+1, ch+1);
  }
}
boot();
</script></body></html>"""


if __name__ == "__main__":
    main()
