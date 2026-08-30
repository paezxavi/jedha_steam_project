"""Print the executed notebook's outputs as text, to check a run without opening Jupyter."""
import sys, json, nbformat
nb = nbformat.read("steam_project.ipynb", as_version=4)
only = sys.argv[1:] and set(int(a) for a in sys.argv[1:])
for i, c in enumerate(nb.cells):
    if c.cell_type != "code" or (only and i not in only):
        continue
    print(f"\n===== cell {i} =====")
    print("\n".join("  | " + l for l in c.source.split("\n")[:4]))
    for o in c.outputs:
        if o.output_type == "stream":
            print(o.text.rstrip()[:2500])
        elif o.output_type == "error":
            print("!! ERROR:", o.ename, o.evalue)
        else:
            d = o.get("data", {})
            print((d.get("text/plain") or "")[:3000])
