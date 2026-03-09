"""Script de CI – verifica que todos los notebooks sean válidos.

No tengan errores en las salidas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_notebook(path: Path) -> list[str]:
    issues = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                issues.append(f"  Celda {i + 1}: {output.get('ename')} – {output.get('evalue')}")
    return issues


def main() -> None:
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    if not notebooks:
        print("No se encontraron notebooks – se omite la validación.")
        return

    failed = False
    for nb_path in notebooks:
        issues = check_notebook(nb_path)
        if issues:
            print(f"[FALLO] {nb_path.name}")
            for issue in issues:
                print(issue)
            failed = True
        else:
            print(f"[OK]    {nb_path.name}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
