"""Run every notebook in Project/Notebooks/ in order.

`python Project/run_all.py` executes the six notebooks via
`jupyter nbconvert --execute --inplace`, so opening the .ipynb files
afterward shows fresh outputs and plots. The candidate-pair dataset
under Project/Data/output/ is pre-computed.
"""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent / "Notebooks"

for notebook in sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*.ipynb")):
    print(f"-> {notebook.name}")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert",
         "--to", "notebook", "--execute", "--inplace", str(notebook)],
        check=True,
    )

print("All notebooks executed successfully.")
