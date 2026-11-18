"""Optional end-to-end smoke test of the tactics board in a real (headless) browser.

    pip install playwright            # uses an installed Edge/Chrome; or `playwright install chromium`
    python scripts/ui_smoke.py [--channel msedge] [--out DIR]

Starts the app, then: reads xG, drags the shooter towards goal (xG must change), switches to
Add mode and adds a defender (one more player), removes it with a double-click, and saves
screenshots of every real scenario to --out. Not part of pytest (it needs a browser).
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 90) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError(url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="msedge", help="installed browser: msedge, chrome; '' = bundled chromium")
    ap.add_argument("--out", default=None, help="screenshot directory (default: a temp dir)")
    ap.add_argument("--width", type=int, default=1500, help="browser viewport width")
    args = ap.parse_args()
    out = Path(args.out or tempfile.mkdtemp(prefix="xg_board_"))
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    port = free_port()
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                            "--server.headless", "true", "--server.port", str(port),
                            "--browser.gatherUsageStats", "false"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    checks = {}
    try:
        base = f"http://localhost:{port}"
        wait_http(base)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel or None, headless=True)
            page = browser.new_page(viewport={"width": args.width, "height": 1000})

            def xg() -> str:
                return page.locator('[data-testid="stMetricValue"]').first.inner_text()

            def distance() -> float:
                txt = page.get_by_text("distance to goal centre").first.inner_text()
                return float(txt.split("distance to goal centre")[1].split("yd")[0])

            def n_players() -> int:
                return page.locator(".xgb-player").count()

            def settle(old: str, timeout: float = 20) -> str:
                t0 = time.time()
                while time.time() - t0 < timeout:
                    page.wait_for_timeout(300)
                    v = xg()
                    if v != old:
                        page.wait_for_timeout(500)
                        return xg()
                return xg()

            page.goto(f"{base}/?scenario=counter")
            page.locator(".xgb-player").first.wait_for(timeout=60000)
            page.locator('[data-testid="stMetricValue"]').first.wait_for(timeout=60000)
            page.wait_for_timeout(1500)
            xg0, n0, d0 = xg(), n_players(), distance()

            # 1) drag the shooter 5 yd towards goal (40 px up)
            shooter = page.locator(".xgb-player:has(polygon)")
            box = shooter.bounding_box()
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            page.mouse.move(cx, cy)
            page.mouse.down()
            for k in range(1, 11):
                page.mouse.move(cx, cy - 4 * k)
                page.wait_for_timeout(20)
            page.mouse.up()
            xg1 = settle(xg0, timeout=6)
            d1 = distance()
            # the model's answer is not checked here (step 4 covers that); the UI must move the shooter
            # 40 screen px; the board may be drawn smaller than its native 640 px (8 px per yd)
            scale = page.locator(".xgb-root").bounding_box()["width"] / 640
            expected = 40 / scale / 8
            checks[f"drag moves the shooter {expected:.1f} yd closer"] = abs((d0 - d1) - expected) < 0.3
            checks["board is fully visible (not clipped)"] = page.evaluate(
                """() => { const host = [...document.querySelectorAll('*')].find(
                       e => e.shadowRoot && e.shadowRoot.querySelector('.xgb-root'));
                   const root = host.shadowRoot.querySelector('.xgb-root');
                   return root.getBoundingClientRect().width <= host.getBoundingClientRect().width + 0.5; }""")
            page.screenshot(path=str(out / "after_drag.png"))

            # 2) Add mode: add a defender 2 yd in front of the shooter
            page.get_by_text("Add", exact=True).first.click()
            page.wait_for_timeout(800)
            page.get_by_text("Defender", exact=True).first.click()
            page.wait_for_timeout(800)
            box = shooter.bounding_box()
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 - 16)
            xg2 = settle(xg1)
            checks["add mode adds a player"] = n_players() == n0 + 1

            # 3) double-click the new defender to remove it
            new = page.locator(".xgb-player").nth(n_players() - 2)        # shooter is drawn last
            nb = new.bounding_box()
            page.mouse.dblclick(nb["x"] + nb["width"] / 2, nb["y"] + nb["height"] / 2)
            settle(xg2)
            checks["double-click removes it"] = n_players() == n0
            page.screenshot(path=str(out / "after_remove.png"))

            # 4) screenshots of every scenario (fresh session each)
            for name in json.loads((ROOT / "app" / "assets" / "scenarios.json").read_text(encoding="utf-8")):
                pg = browser.new_page(viewport={"width": 1500, "height": 1000})
                pg.goto(f"{base}/?scenario={name['name']}")
                pg.locator(".xgb-player").first.wait_for(timeout=60000)
                pg.wait_for_timeout(2500)
                pg.locator(".xgb-root").screenshot(path=str(out / f"board_{name['name']}.png"))
                pg.close()
            browser.close()
    finally:
        app.terminate()

    print(f"distance {d0:.1f} -> {d1:.1f} yd; xG: start {xg0}, after drag {xg1}, "
          f"after adding a defender 2 yd in front {xg2}")
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(f"screenshots -> {out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
