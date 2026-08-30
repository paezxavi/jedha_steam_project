"""Execute steam_project.ipynb in place, so the committed notebook carries its outputs."""
import os, sys, nbformat
os.environ.setdefault("JAVA_HOME", os.path.expanduser("~/.local/jdk/jdk-17.0.20.1+1"))
from nbclient import NotebookClient
nb = nbformat.read("steam_project.ipynb", as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "."}})
client.execute()
nbformat.write(nb, "steam_project.ipynb")
print(f"EXECUTED_OK {len(nb.cells)} cells")
