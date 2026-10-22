"""Screenshots of every code cell of the Databricks export, plus a paginated PDF of it.

Databricks no longer publishes notebooks, and the certification asks for screenshots of the cell
outputs instead. This script renders `steam_project_databricks.html` in headless Chromium and
writes one PNG per code cell into `images/`, named `<cell>_<section>.png` after the cell number
shown in the export and the heading that precedes it. A cell whose result also has a chart is
captured twice: `_table.png` with every result on its Table tab, `_chart.png` with the
Visualization tabs open. The whole rendered notebook goes to `steam_project_databricks.pdf`.

Three things are fixed in the browser before capturing, because the export does not show them:
every result opens on its Table tab (the charts are one click away), long text outputs sit in a
400 px scroll box, and the notebook itself sits in a fixed-height box that print cannot paginate.
The export loads its scripts from the Databricks CDN, so the run needs network.

    python capture.py              # from the repository root
"""
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / "steam_project_databricks.html"
IMAGES = ROOT / "images"
PDF = ROOT / "steam_project_databricks.pdf"
N_CELLS = 82          # the export's cell count; the container with that many children is the notebook

CELLS_JS = """() => {
  const parent = [...document.querySelectorAll('div')].find(e => e.children.length === %d);
  return [...parent.children].map(c => {
    const r = c.getBoundingClientRect();
    const box = c.querySelector('.command-box');
    const heads = [...c.querySelectorAll('h1, h2, h3')].map(h => h.textContent.trim());
    return {x: r.x, y: r.y + window.scrollY, w: r.width, h: r.height,
            code: !!box && box.getBoundingClientRect().height > 0,
            heading: heads.length ? heads[heads.length - 1] : null,
            chart: [...c.querySelectorAll('[role=tab]')].some(t => !t.innerText.trim().startsWith('Table'))};
  });
}""" % N_CELLS

UNCLIP_JS = """() => {
  let e = [...document.querySelectorAll('div')].find(e => e.children.length === %d);
  for (; e; e = e.parentElement) {
    for (const [k, v] of [['height', 'auto'], ['max-height', 'none'], ['min-height', '0'],
                          ['overflow', 'visible'], ['position', 'static']])
      e.style.setProperty(k, v, 'important');
  }
  const btn = [...document.querySelectorAll('button')]
    .filter(b => b.innerText.trim() === 'Import Notebook' && b.getBoundingClientRect().width > 0)[0];
  if (btn) btn.parentElement.parentElement.remove();          // the top bar collapses once unclipped
  const logo = document.querySelector('img');
  if (logo) logo.closest('div').remove();
}""" % N_CELLS


def slug(heading):
    """'3.3 Price' -> '3.3_price'; '0. Environment' -> '0_environment'."""
    s = heading.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9.]+", "_", s).strip("_.")
    s = re.sub(r"\.(?=_|$)", "", s)                 # '0._environment' -> '0_environment'
    return s[:40].rstrip("_")


def shoot(page, cells, i, name):
    c = cells[i]
    page.screenshot(path=str(IMAGES / name), clip={"x": c["x"], "y": c["y"], "width": c["w"], "height": c["h"]})
    print(f"  {name}")


def main():
    IMAGES.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(EXPORT.as_uri(), wait_until="networkidle", timeout=120_000)
        time.sleep(5)
        page.set_viewport_size({"width": 1400, "height": 45_000})    # the whole notebook laid out at once
        time.sleep(3)
        page.add_style_tag(content=".css-1isht1x{max-height:none!important;height:auto!important;overflow:visible!important}")
        time.sleep(2)

        # pass 1: every result on its Table tab
        cells = page.evaluate(CELLS_JS)
        names, heading = {}, "untitled"
        for i, c in enumerate(cells):
            if c["heading"]:
                heading = c["heading"]
            if c["code"]:
                names[i] = f"{i + 1:02d}_{slug(heading)}"
        code_cells = sorted(names)
        charted = [i for i in code_cells if cells[i]["chart"]]
        print(f"{len(code_cells)} code cells, {len(charted)} with a chart")
        for i in code_cells:
            shoot(page, cells, i, names[i] + ("_table.png" if i in charted else ".png"))

        # pass 2: the Visualization tabs open, the charted cells again
        tabs = page.locator("[role=tab]").filter(has_not_text="Table")
        n_tabs = tabs.count()
        for k in range(n_tabs):
            tabs.nth(k).click(force=True)
        time.sleep(4)
        cells = page.evaluate(CELLS_JS)             # heights changed with the charts open
        print(f"{n_tabs} chart tabs opened")
        for i in charted:
            shoot(page, cells, i, names[i] + "_chart.png")

        # the PDF: back to a normal viewport (a 45 000 px one prints as blank pages), then unclip
        page.set_viewport_size({"width": 1400, "height": 900})
        time.sleep(2)
        page.evaluate(UNCLIP_JS)
        page.add_style_tag(content="[class*=css-1b8mo0r]{break-inside:avoid}")
        page.pdf(path=str(PDF), format="A4", scale=0.62, print_background=True,
                 margin={"top": "8mm", "bottom": "8mm", "left": "4mm", "right": "4mm"})
        print(f"  {PDF.name}")
        browser.close()


if __name__ == "__main__":
    main()
