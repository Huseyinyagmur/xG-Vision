"""Record the README demo GIF of the tactics board with a headless browser.

    pip install playwright
    python scripts/record_demo.py [--out results/plots/board_demo.gif] [--channel msedge]

Counter-attack scenario: a defender is dragged onto the shot line, the goalkeeper steps
sideways, the xG map is switched on and the shooter moves to a better spot. Frames are
screenshots taken step by step (deterministic, smooth), with a drawn cursor because headless
browsers do not show one.
"""
from __future__ import annotations

import argparse
import io
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import board as B  # noqa: E402

CURSOR_JS = """
window.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.id = 'demo-cursor';
  Object.assign(c.style, {position: 'fixed', width: '18px', height: '18px', borderRadius: '50%',
    border: '2px solid #111', background: 'rgba(255,255,255,0.75)', zIndex: 2147483647,
    pointerEvents: 'none', transform: 'translate(-50%,-50%)', left: '-50px', top: '-50px',
    transition: 'background 80ms'});
  document.body.appendChild(c);
  // pointer events: the board calls preventDefault() on pointerdown, which suppresses mouse events
  document.addEventListener('pointermove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('pointerdown', () => { c.style.background = 'rgba(255,212,0,0.95)'; }, true);
  document.addEventListener('pointerup', () => { c.style.background = 'rgba(255,255,255,0.75)'; }, true);
});
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "plots" / "board_demo.gif"))
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--width", type=int, default=800, help="GIF width in px")
    ap.add_argument("--colors", type=int, default=192)
    args = ap.parse_args()

    from PIL import Image
    from playwright.sync_api import sync_playwright

    port = free_port()
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                            "--server.headless", "true", "--server.port", str(port),
                            "--browser.gatherUsageStats", "false"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames: list[tuple[Image.Image, int]] = []
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://localhost:{port}", timeout=2)
                break
            except OSError:
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel or None, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.add_init_script(CURSOR_JS)
            page.goto(f"http://localhost:{port}/?scenario=counter")
            page.locator('[data-testid="stMetricValue"]').first.wait_for(timeout=90000)
            page.wait_for_timeout(2500)

            root = page.locator(".xgb-root").bounding_box()
            top = page.get_by_text("Scenario", exact=True).first.bounding_box()["y"]
            clip = {"x": root["x"] - 10, "y": top - 10, "width": root["width"] + 32 + 340 + 20,
                    "height": root["y"] + root["height"] + 56 - (top - 10)}      # down to the map legend
            s = root["width"] / B.CANVAS_W                      # screen px per canvas px

            def snap(ms: int, n: int = 1, gap: int = 0):
                for _ in range(n):
                    if gap:
                        page.wait_for_timeout(gap)
                    img = Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB")
                    frames.append((img, ms))

            def to_screen(px: float, py: float) -> tuple[float, float]:
                return root["x"] + px * s, root["y"] + py * s

            def player_center(sel: str) -> tuple[float, float]:
                b = page.locator(sel).bounding_box()
                return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2

            def glide(x0, y0, x1, y1, steps=14, ms=55, drag=False):
                page.mouse.move(x0, y0)
                if drag:
                    page.mouse.down()
                for k in range(1, steps + 1):
                    page.mouse.move(x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps)
                    snap(ms)
                if drag:
                    page.mouse.up()

            def settle(seconds: float):
                snap(220, n=max(1, int(seconds / 0.22)), gap=200)

            # 0) the situation
            page.mouse.move(clip["x"] + clip["width"] - 60, clip["y"] + 60)
            settle(1.2)

            # 1) a defender onto the shot line, 4 yd in front of the shooter
            sx, sy = player_center(".xgb-player:has(polygon)")
            gx, gy = to_screen(*B.to_px(120.0, 40.0))
            d = ((gx - sx) ** 2 + (gy - sy) ** 2) ** 0.5
            tx, ty = sx + (gx - sx) * (4 * B.PPY * s) / d, sy + (gy - sy) * (4 * B.PPY * s) / d
            defenders = page.locator(".xgb-player:has(circle[fill='#d62728'])")
            boxes = [defenders.nth(k).bounding_box() for k in range(defenders.count())]
            near = min(boxes, key=lambda b: (b["x"] - sx) ** 2 + (b["y"] - sy) ** 2)
            dx, dy = near["x"] + near["width"] / 2, near["y"] + near["height"] / 2
            glide(clip["x"] + clip["width"] - 60, clip["y"] + 60, dx, dy, steps=10)
            glide(dx, dy, tx, ty, steps=18, drag=True)
            settle(1.8)

            # 2) the goalkeeper steps 4 yd to the side
            kx, ky = player_center(".xgb-player:has(rect)")
            glide(tx, ty, kx, ky, steps=10)
            glide(kx, ky, kx + 4 * B.PPY * s, ky, steps=12, drag=True)
            settle(1.8)

            # 3) switch the xG map on
            tog = page.get_by_text("xG map", exact=True).first.bounding_box()
            glide(kx + 4 * B.PPY * s, ky, tog["x"] + 10, tog["y"] + tog["height"] / 2, steps=12)
            page.mouse.click(tog["x"] + 10, tog["y"] + tog["height"] / 2)
            page.get_by_text("Best spot", exact=False).first.wait_for(timeout=120000)
            settle(2.4)

            # 4) the shooter moves to a brighter spot of the map
            sx, sy = player_center(".xgb-player:has(polygon)")
            bx, by = to_screen(*B.to_px(104.0, 44.0))
            glide(tog["x"] + 10, tog["y"] + tog["height"] / 2, sx, sy, steps=12)
            glide(sx, sy, bx, by, steps=20, drag=True)
            glide(bx, by, bx + 70 * s, by + 50 * s, steps=6)        # cursor off the shooter
            settle(2.6)
            browser.close()
    finally:
        app.terminate()

    # GIF: one shared adaptive palette built from 12 frames spread over the whole recording
    # (so the map colours and the yellow shooter are in it), every frame mapped onto it
    w = args.width
    small = [(im.resize((w, round(im.height * w / im.width)), Image.LANCZOS), ms) for im, ms in frames]
    h = small[0][0].height
    picks = [round(k * (len(small) - 1) / 11) for k in range(12)]
    sample = Image.new("RGB", (w, h * len(picks)))
    for k, idx in enumerate(picks):
        sample.paste(small[idx][0], (0, k * h))
    palette = sample.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
    pal_frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im, _ in small]
    out = Path(args.out)
    pal_frames[0].save(out, save_all=True, append_images=pal_frames[1:], duration=[ms for _, ms in small],
                       loop=0, optimize=True, disposal=1)
    total = sum(ms for _, ms in small) / 1000
    print(f"{len(small)} frames, {total:.1f} s, {out.stat().st_size / 1e6:.2f} MB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
