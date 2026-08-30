"""Turn build/nb_source.txt into steam_project.ipynb (+ a Databricks-importable .py).

Source format: cells separated by lines '#%% md' or '#%% code'.
"""
import json, sys, pathlib

src = pathlib.Path("build/nb_source.txt").read_text()
cells, kind, buf = [], None, []

def flush():
    if kind and buf:
        body = "\n".join(buf).strip("\n")
        if body.strip():
            cells.append((kind, body))

for line in src.split("\n"):
    if line.startswith("#%% "):
        flush(); kind, buf = line[4:].strip(), []
    else:
        buf.append(line)
flush()

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
for k, body in cells:
    lines = [l + "\n" for l in body.split("\n")]
    lines[-1] = lines[-1].rstrip("\n")
    if k == "md":
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})
    else:
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": lines})
pathlib.Path("steam_project.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

# Databricks source format, importable straight into a workspace
out = ["# Databricks notebook source"]
for i, (k, body) in enumerate(cells):
    if i: out.append("\n# COMMAND ----------\n")
    if k == "md":
        out.append("# MAGIC %md")
        out += ["# MAGIC " + l if l else "# MAGIC" for l in body.split("\n")]
    else:
        out.append(body)
pathlib.Path("steam_project_databricks.py").write_text("\n".join(out) + "\n")
print(f"OK: {len(cells)} cells ({sum(1 for k,_ in cells if k=='code')} code, "
      f"{sum(1 for k,_ in cells if k=='md')} md) -> steam_project.ipynb + steam_project_databricks.py")
