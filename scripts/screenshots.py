"""Capture application screenshots for the report by driving the running app with Playwright.

Requires the app at http://localhost:8000 (docker compose up) and `playwright install chromium`.
Usage: python scripts/screenshots.py [--steps screen,benchmark,experiments] [--criteria 6]
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures" / "screens"
URL = "http://localhost:8000"


def screen(page, args):
    # Invalid input first: run without a note.
    page.click("button.primary")
    page.wait_for_selector(".alert.bad")
    page.locator("section.panel").first.screenshot(path=OUT / "invalid_input.png")

    # Screening a validation example, keeping a few criteria.
    page.select_option("select.full", index=1)
    rows = page.locator(".criterion-edit")
    while rows.count() > args.criteria:
        rows.nth(rows.count() - 1).locator("button.icon").click()
    page.click("button.primary")
    page.wait_for_selector("text=Download report (HTML)", timeout=1_200_000)
    page.locator(".card").nth(0).click()
    page.screenshot(path=OUT / "screen.png", full_page=True)


def benchmark(page, args):
    page.click("button.tab:has-text('Inference benchmark')")
    page.locator("input[type=range]").evaluate(
        "(el, v) => { const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;"
        " set.call(el, v); el.dispatchEvent(new Event('input', { bubbles: true })); }", str(args.bench_tokens))
    page.get_by_role("button", name="Run benchmark").click()
    page.wait_for_selector("text=Same tokens", timeout=1_800_000)
    page.get_by_role("button", name="Compare").click()
    page.wait_for_selector("text=Max |err|", timeout=300_000)
    page.wait_for_timeout(1500)
    page.screenshot(path=OUT / "benchmark.png", full_page=True)


def experiments(page, args):
    page.click("button.tab:has-text('Experiments')")
    page.wait_for_selector("text=Model evaluations")
    page.wait_for_timeout(1500)
    panels = page.locator("main > .stack > section.panel")  # the other tabs stay mounted but hidden
    panels.nth(0).screenshot(path=OUT / "experiments_eval.png")
    panels.nth(2).screenshot(path=OUT / "experiments_training.png")
    page.screenshot(path=OUT / "experiments.png", full_page=True)


STEPS = {"screen": screen, "benchmark": benchmark, "experiments": experiments}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="screen,benchmark,experiments")
    parser.add_argument("--criteria", type=int, default=6, help="criteria kept from the example")
    parser.add_argument("--bench-tokens", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.goto(URL)
        page.wait_for_selector("text=Ready", timeout=600_000)
        for step in args.steps.split(","):
            STEPS[step](page, args)
            print("captured", step, flush=True)
        browser.close()


if __name__ == "__main__":
    main()
