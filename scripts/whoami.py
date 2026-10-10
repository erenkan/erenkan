#!/usr/bin/env python3
"""Render a photo as a retro CRT, neofetch-style `whoami` SVG card for a GitHub profile README.

Usage:
  python scripts/whoami.py --photo https://avatars.githubusercontent.com/u/29619699 --out whoami.svg
Requires: Pillow, numpy, opencv-python-headless (see scripts/requirements.txt).
"""
import argparse, base64, io, os, urllib.request
from xml.sax.saxutils import escape
import numpy as np, cv2
from PIL import Image, ImageFilter

RAMP = " .:-=+*#%@"
PHOSPHOR = {"green": "#3dff6e", "amber": "#ffb000"}
ACCENT = {"amber": "#ffb000", "cyan": "#5ff1ff", "green": "#9dffb8", "white": "#e8e8e8"}
CGA = ["#0f0f0f", "#aa0000", "#00aa00", "#aa5500", "#0000aa", "#aa00aa", "#00aaaa", "#aaaaaa",
       "#555555", "#ff5555", "#55ff55", "#ffff55", "#5555ff", "#ff55ff", "#55ffff", "#ffffff"]
SCREEN = {"green": ("#0b140c", "#030603"), "amber": ("#160f00", "#070400")}
ADV, ASC = 0.4, 0.8           # VT323: advance 0.4 em, ascent 0.8 em, line height 1.0 em

def load_photo(src):
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"User-Agent": "whoami-card"})
        return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=30).read())).convert("RGB")
    return Image.open(src).convert("RGB")

def prep(im, crop, clahe=1.5, gate=0.12):
    """Key out the blue backdrop, drop clutter, boost local contrast; returns (cropped RGB image, cropped mask)."""
    rgb = np.asarray(im, dtype=np.float32); H, W, _ = rgb.shape
    yy, xx = np.mgrid[0:H, 0:W]
    fx, fy = xx / W, yy / H                # fractions, so any avatar size works
    fade = np.clip((fy - 0.73) / (0.91 - 0.73), 0, 1)
    # Tone mask (drives the composite and the contrast stretch): distance from the backdrop colour.
    bg = np.concatenate([rgb[5:40, 5:40].reshape(-1, 3), rgb[5:40, -40:-5].reshape(-1, 3)]).mean(0)
    mask = (np.sqrt(((rgb - bg) ** 2).sum(-1)) > 38).astype(np.float32)
    mask[(fx > 0.685) & (fy > 0.40)] = 0   # wall / sign on the right
    mask[(fx < 0.13) & (fy > 0.62)] = 0    # left edge
    mask *= (1 - fade) ** 1.5              # torso fades out below the collar
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)[..., None] / 255
    # Silhouette mask (drives the hair floor only): the backdrop is blue-dominant in every shade,
    # hair and jacket are neutral, skin is warm. Cleaned up morphologically.
    blue_excess = rgb[..., 2] - np.maximum(rgb[..., 0], rgb[..., 1])
    key = (blue_excess < 22).astype(np.uint8)
    key[(fx > 0.685) & (fy > 0.40)] = 0
    key[(fx < 0.13) & (fy > 0.62)] = 0
    ell = cv2.getStructuringElement
    key = cv2.morphologyEx(key, cv2.MORPH_OPEN, ell(cv2.MORPH_ELLIPSE, (5, 5)))     # drop specks
    key = cv2.morphologyEx(key, cv2.MORPH_CLOSE, ell(cv2.MORPH_ELLIPSE, (11, 11)))  # bridge gaps
    n, labels, stats, _ = cv2.connectedComponentsWithStats(key)
    if n > 1:
        key = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)  # the person
    ff = key.copy(); border = np.zeros((H + 2, W + 2), np.uint8)
    for x0, y0 in [(x, y) for x in range(W) for y in (0, H - 1)] + [(x, y) for y in range(H) for x in (0, W - 1)]:
        if ff[y0, x0] == 0:
            cv2.floodFill(ff, border, (x0, y0), 1)
    key |= (ff == 0).astype(np.uint8)      # fill enclosed holes (dark hair that matched the backdrop)
    sil = key.astype(np.float32) * (1 - fade) ** 1.5
    img = cv2.bilateralFilter(rgb.astype(np.uint8), 9, 45, 7)
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8)).apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    comp = img.astype(np.float32) * m
    lum = comp @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    fg = lum[m[..., 0] > 0.5]
    lo, hi = np.percentile(fg, 1), np.percentile(fg, 99.5)
    comp = np.clip((comp - lo) / max(1.0, hi - lo), 0, 1)
    comp = np.clip((comp - gate) / (1 - gate), 0, 1) ** 0.9
    out = Image.fromarray((comp * 255).astype(np.uint8)).filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))
    sil_img = Image.fromarray((sil * 255).astype(np.uint8))
    return out.crop(crop), sil_img.crop(crop)

def to_grid(img, mask, cols, cell_ratio, edge=0.0, dark=0.5, knee=0.8, contrast=1.0, floor=0.0, floor_until=0.3):
    """Luminance per character cell.

    dark:     blend of the cell mean with the cell minimum (erode), so thin dark features
              such as glasses frames and pupils survive the downsampling.
    knee:     soft-compress highlights above this level (lens reflections, forehead shine).
    contrast: S-curve strength around mid grey (1.0 = off).
    edge:     brighten strong edges (off by default: it washes out the eyes).
    floor:    minimum luminance inside the subject mask, applied to the top of the portrait only
              (rows above `floor_until`, as a fraction of the height), so dark hair renders as a
              silhouette attached to the forehead instead of floating specks."""
    g = img.convert("L"); w, h = g.size
    rows = round(cols * (h / w) * cell_ratio)
    a = np.asarray(g.resize((cols, rows), Image.LANCZOS), dtype=np.float32) / 255
    if dark > 0:
        kw, kh = max(1, int(w / cols * 0.8)), max(1, int(h / rows * 0.8))
        er = cv2.erode(np.asarray(g), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))
        mn = np.asarray(Image.fromarray(er).resize((cols, rows), Image.BOX), dtype=np.float32) / 255
        a = (1 - dark) * a + dark * mn
    if knee < 1:
        a = np.where(a > knee, knee + (a - knee) * 0.5, a)
    if contrast != 1.0:
        a = np.clip(0.5 + np.sign(a - 0.5) * np.abs(a - 0.5) ** (1 / contrast) * 0.5 ** (1 - 1 / contrast), 0, 1)
    if floor > 0:
        mg = np.asarray(mask.resize((cols, rows), Image.LANCZOS), dtype=np.float32) / 255
        rf = (np.arange(rows, dtype=np.float32) / max(1, rows - 1))[:, None]
        wgt = np.clip((floor_until - rf) / 0.08, 0, 1)         # hair zone only: full above, off below
        a = np.maximum(a, floor * wgt * np.clip((mg - 0.35) / 0.3, 0, 1))
    if edge > 0:
        src = np.asarray(g, dtype=np.float32) / 255
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.asarray(Image.fromarray(np.sqrt(gx ** 2 + gy ** 2)).resize((cols, rows), Image.BOX), dtype=np.float32)
        mag = np.clip(mag / max(1e-6, np.percentile(mag, 97)), 0, 1)
        a = np.clip(a + edge * mag * (a > 0.03), 0, 1)
    return a

def art_rows(grid):
    rows = []
    for r in grid:
        cells = []
        for v in r:
            ch = RAMP[int(round(v * (len(RAMP) - 1)))]
            op = 1.0 if v > 0.75 else 0.85 if v > 0.5 else 0.7 if v > 0.25 else 0.55
            cells.append((ch, op))
        while cells and cells[-1][0] == " ":
            cells.pop()
        rows.append(cells)
    return rows

def tspans(cells):
    out, run, cur = [], [], None
    for ch, op in cells + [(None, None)]:
        if ch is not None and (cur is None or op == cur):
            run.append(ch); cur = op; continue
        if run:
            txt = escape("".join(run))
            out.append(txt if cur == 1.0 else f'<tspan opacity="{cur}">{txt}</tspan>')
        run, cur = ([ch], op) if ch is not None else ([], None)
    return "".join(out)

def build(a):
    fg, key, scr0, scr1 = PHOSPHOR[a.phosphor], ACCENT[a.accent], *SCREEN[a.phosphor]
    im = load_photo(a.photo)
    fr = [float(v) for v in a.crop.split(",")]       # crop as fractions of width/height
    crop = (int(fr[0] * im.width), int(fr[1] * im.height), int(fr[2] * im.width), int(fr[3] * im.height))
    img, mask = prep(im, crop)
    rows = art_rows(to_grid(img, mask, a.cols, ADV, a.edge, a.dark, a.knee, a.contrast, a.floor, a.floor_until))
    FA, FI = a.art_size, a.info_size
    advA, advI = FA * ADV, FI * ADV
    info = [l.rstrip("\n") for l in open(a.info, encoding="utf-8") if l.strip()]
    user_host = info[0]
    prompt = [(user_host, "u"), (":~$ ", ""), (a.cmd, "c")]
    prompt_len = sum(len(t) for t, _ in prompt)
    PAD, BEZEL, GAP = 26, 14, 30
    art_w, art_h = a.cols * advA, len(rows) * FA
    info_w = max(len(l) for l in info + [" " * prompt_len]) * advI
    line = FI + 6
    W = int(round(max(BEZEL + PAD + art_w + GAP + info_w + PAD + BEZEL, a.width)))   # extra width = blank screen on the right
    H = int(round(BEZEL + PAD + line + max(art_h, (len(info) + 4) * FI) + line + PAD + BEZEL))
    x_art, x_info = BEZEL + PAD, BEZEL + PAD + art_w + GAP
    y_prompt, y_top = BEZEL + PAD + FI * ASC, BEZEL + PAD + line
    font_b64 = base64.b64encode(open(a.font, "rb").read()).decode()
    anim = not a.static
    typing_dt = 0.055
    t_out = 0.4 + typing_dt * prompt_len + 0.35
    t_end = t_out + a.row_dt * max(len(rows), len(info) + 4) + 0.2
    appear = (lambda t: f'<set attributeName="opacity" to="1" begin="{t:.2f}s" fill="freeze"/>') if anim else (lambda t: "")
    op0 = ' opacity="0"' if anim else ""
    P = []
    P.append(f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(a.alt)}">')
    P.append(f"<title>{escape(a.alt)}</title>")
    P.append(f"""<defs>
<style>
@font-face{{font-family:"VT323";src:url(data:font/woff2;base64,{font_b64}) format("woff2");}}
text{{font-family:"VT323",ui-monospace,Menlo,Consolas,"DejaVu Sans Mono",monospace;white-space:pre;fill:{fg}}}
.a{{font-size:{FA}px}} .i{{font-size:{FI}px}} .k{{fill:{key}}} .u{{fill:{key}}} .c{{fill:{ACCENT['green'] if a.phosphor == 'green' else '#ffd56a'}}} .d{{opacity:.5}}
@media (prefers-reduced-motion:reduce){{[opacity="0"]{{opacity:1 !important}}}}
</style>
<radialGradient id="scr" cx="50%" cy="45%" r="75%"><stop offset="0" stop-color="{scr0}"/><stop offset="1" stop-color="{scr1}"/></radialGradient>
<radialGradient id="vig" cx="50%" cy="50%" r="72%"><stop offset="0.55" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.65"/></radialGradient>
<linearGradient id="bz" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2b2b2b"/><stop offset="1" stop-color="#1c1c1c"/></linearGradient>
<pattern id="scan" width="2" height="3" patternUnits="userSpaceOnUse"><rect width="2" height="1" fill="#000" opacity="{a.scanline}"/></pattern>
<filter id="glow" x="-5%" y="-5%" width="110%" height="110%"><feGaussianBlur stdDeviation="{a.glow}" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>""")
    P.append(f'<rect width="{W}" height="{H}" rx="18" fill="url(#bz)"/>')
    P.append(f'<rect x="{BEZEL}" y="{BEZEL}" width="{W-2*BEZEL}" height="{H-2*BEZEL}" rx="10" fill="url(#scr)"/>')
    P.append('<g filter="url(#glow)">')
    # prompt line, typed character by character
    chars, i = [], 0
    for txt, cls in prompt:
        c = f' class="{cls}"' if cls else ""
        for ch in txt:
            chars.append(f'<tspan{c}{op0}>{escape(ch)}{appear(0.4 + typing_dt * i)}</tspan>'); i += 1
    P.append(f'<text class="i" x="{x_art}" y="{y_prompt:.1f}" textLength="{prompt_len*advI:.1f}" lengthAdjust="spacingAndGlyphs">{"".join(chars)}</text>')
    # portrait
    for r_i, cells in enumerate(rows):
        if not cells: continue
        y = y_top + r_i * FA + FA * ASC
        P.append(f'<text class="a" x="{x_art}" y="{y:.1f}"{op0} textLength="{len(cells)*advA:.1f}" lengthAdjust="spacingAndGlyphs">{tspans(cells)}{appear(t_out + a.row_dt * r_i)}</text>')
    # info block
    def info_text(r_i, inner, n):
        y = y_top + r_i * FI + FI * ASC
        return f'<text class="i" x="{x_info}" y="{y:.1f}"{op0} textLength="{n*advI:.1f}" lengthAdjust="spacingAndGlyphs">{inner}{appear(t_out + a.row_dt * r_i)}</text>'
    P.append(info_text(0, f'<tspan class="u">{escape(user_host)}</tspan>', len(user_host)))
    P.append(info_text(1, f'<tspan class="d">{"-" * len(user_host)}</tspan>', len(user_host)))
    for j, l in enumerate(info[1:], start=2):
        k, v = l.split(":", 1) if ":" in l else ("", l)
        inner = f'<tspan class="k">{escape(k)}:</tspan>{escape(v)}' if k else escape(l)
        P.append(info_text(j, inner, len(l)))
    base = len(info) + 2
    for r_i in range(2):
        blocks = "".join(f'<tspan fill="{c}">███</tspan>' for c in CGA[r_i * 8:(r_i + 1) * 8])
        P.append(info_text(base + r_i, blocks, 24))
    # closing prompt with blinking cursor
    y_end = max(y_top + len(rows) * FA, y_top + (base + 2) * FI) + FI * ASC + 4
    cur = '<tspan>█<animate attributeName="opacity" values="1;1;0;0" dur="1.1s" repeatCount="indefinite"/></tspan>' if anim else "█"
    P.append(f'<text class="i" x="{x_art}" y="{y_end:.1f}"{op0}><tspan class="u">{escape(user_host)}</tspan>:~$ {cur}{appear(t_end)}</text>')
    P.append("</g>")
    P.append(f'<rect x="{BEZEL}" y="{BEZEL}" width="{W-2*BEZEL}" height="{H-2*BEZEL}" rx="10" fill="url(#scan)"/>')
    P.append(f'<rect x="{BEZEL}" y="{BEZEL}" width="{W-2*BEZEL}" height="{H-2*BEZEL}" rx="10" fill="url(#vig)"/>')
    P.append("</svg>\n")
    svg = "\n".join(P)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(svg)
    print(f"{a.out}: {W}x{H}px, portrait {a.cols}x{len(rows)} cells, {len(svg)/1024:.0f} KB, animation ends ~{t_end:.1f}s")

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", default="https://avatars.githubusercontent.com/u/29619699?v=4&s=1024")
    ap.add_argument("--out", default=os.path.join(here, "..", "whoami.svg"))
    ap.add_argument("--info", default=os.path.join(here, "info.txt"))
    ap.add_argument("--font", default=os.path.join(here, "..", "fonts", "VT323-subset.woff2"))
    ap.add_argument("--cmd", default="whoami"); ap.add_argument("--width", type=float, default=900, help="minimum card width in px; content stays left-aligned")
    ap.add_argument("--phosphor", choices=PHOSPHOR, default="green"); ap.add_argument("--accent", choices=ACCENT, default="amber")
    ap.add_argument("--cols", type=int, default=100); ap.add_argument("--art-size", type=float, default=8.5); ap.add_argument("--info-size", type=float, default=17.0)
    ap.add_argument("--crop", default="0.087,0,0.87,0.913", help="left,top,right,bottom as fractions"); ap.add_argument("--edge", type=float, default=0.0)
    ap.add_argument("--dark", type=float, default=0.6); ap.add_argument("--knee", type=float, default=0.75); ap.add_argument("--contrast", type=float, default=1.3)
    ap.add_argument("--floor", type=float, default=0.18); ap.add_argument("--floor-until", type=float, default=0.3)
    ap.add_argument("--glow", type=float, default=1.1); ap.add_argument("--scanline", type=float, default=0.32)
    ap.add_argument("--row-dt", type=float, default=0.045); ap.add_argument("--static", action="store_true")
    ap.add_argument("--alt", default="Retro terminal card: ASCII portrait of Eren Kan with a whoami-style profile summary")
    build(ap.parse_args())
