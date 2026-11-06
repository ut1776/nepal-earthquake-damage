#!/usr/bin/env python3
"""Fast, dependency-free repository integrity checks for RICHTER."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NOTEBOOK = ROOT / "Notebook" / "richter-earthquake-damage-prediction.ipynb"

def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)

def main() -> None:
    if not README.is_file():
        fail("README.md is missing")
    if not NOTEBOOK.is_file():
        fail("project notebook is missing")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if not cells or cells[0].get("cell_type") != "markdown":
        fail("notebook must begin with a Markdown introduction")

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if not code_cells:
        fail("notebook contains no code cells")

    errors = []
    for index, cell in enumerate(cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append((index, output.get("ename", "unknown error")))
    if errors:
        fail(f"notebook contains recorded execution errors: {errors}")

    required_paths = [
        "Dataset/train_values.csv",
        "Dataset/train_labels.csv",
        "Dataset/test_values.csv",
        "Dataset/submission_format.csv",
        "assets/retrospective-banner.svg",
        "assets/model-results.svg",
        "assets/workflow.svg",
        "requirements.txt",
        "DATASET_NOTICE.md",
    ]
    missing = [path for path in required_paths if not (ROOT / path).is_file()]
    if missing:
        fail(f"required project files are missing: {missing}")

    readme = README.read_text(encoding="utf-8")
    referenced_assets = re.findall(r'src="(assets/[^"]+)"', readme)
    missing_assets = [path for path in referenced_assets if not (ROOT / path).is_file()]
    if missing_assets:
        fail(f"README references missing image assets: {missing_assets}")

    if "Notebook/richter-earthquake-damage-prediction.ipynb" not in readme:
        fail("README does not link to the project notebook")

    print("RICHTER repository validation passed")
    print(f"Notebook cells: {len(cells)}")
    print(f"Code cells: {len(code_cells)}")
    print(f"Recorded notebook errors: {len(errors)}")
    print(f"README image references checked: {len(referenced_assets)}")
    print(f"Required project files checked: {len(required_paths)}")

if __name__ == "__main__":
    main()
