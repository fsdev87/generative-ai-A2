"""Capture application screenshots for the report by driving the running app with Playwright.

Requires the app at http://localhost:8000 (docker compose up) and `playwright install chromium`.
Usage: python scripts/screenshots.py [--criteria 6]
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures" / "screens"
URL = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteria", type=int, default=6, help="criteria kept from the example")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.goto(URL)
        page.wait_for_selector("text=Ready", timeout=600_000)

        # 1. Invalid input: run without a note.
        page.click("button.primary")
        page.wait_for_selector(".alert.bad")
        page.locator("section.panel").first.screenshot(path=OUT / "invalid_input.png")

        # 2. Screening a validation example with a few criteria.
        page.select_option("select.full", index=1)
        rows = page.locator(".criterion-edit")
        while rows.count() > args.criteria:
            rows.nth(rows.count() - 1).locator("button.icon").click()
        page.click("button.primary")
        page.wait_for_selector("text=Download report (HTML)", timeout=1_200_000)
        page.locator(".card").nth(0).click()
        page.screenshot(path=OUT / "screen.png", full_page=True)

        # 3. Live decoding benchmark (cached, uncached, speculative).
        page.click("button.tab:has-text('Inference benchmark')")
        page.locator("input[type=range]").evaluate(
            "(el) => { const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;"
            " set.call(el, '24'); el.dispatchEvent(new Event('input', { bubbles: true })); }")
        page.click("div:not([hidden]) > .grid button.primary")
        page.wait_for_selector("text=Same tokens", timeout=1_800_000)
        page.click("text=Compare")
        page.wait_for_selector("text=Max |err|", timeout=300_000)
        page.screenshot(path=OUT / "benchmark.png", full_page=True)

        # 4. Experiments page rendered from result files.
        page.click("button.tab:has-text('Experiments')")
        page.wait_for_selector("text=Model evaluations")
        page.screenshot(path=OUT / "experiments.png", full_page=True)
        browser.close()
    print("screenshots in", OUT)


if __name__ == "__main__":
    main()
