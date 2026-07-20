#!/usr/bin/env python3
"""Validate the self-contained public release without recomputing the scans."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

EXPECTED_COUNTS = {
    "phase5": 4_165,
    "phase6": 24_990,
    "phase7": 24_990,
    "phase8": 2_400,
    "phase9": 3_200,
    "phase10": 5_760,
}

FIGURE_STEMS = {
    "cm_population_distance_symmetry",
    "equal_distance_cm_controls",
    "adversarial_local_search",
    "passive_gate_retention",
    "blind_bounded_global_search",
    "headline_numerical_evidence",
}

LOCAL_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def check_consolidated_results() -> None:
    results = json.loads((ROOT / "data/consolidated_results.json").read_text())
    assert results["phase5"]["cm_candidate_count"] == EXPECTED_COUNTS["phase5"]
    assert results["phase6"]["generic_control_count"] == EXPECTED_COUNTS["phase6"]
    assert results["phase7"]["equal_distance_control_count"] == EXPECTED_COUNTS["phase7"]
    assert results["phase8"]["objective_evaluations"] == EXPECTED_COUNTS["phase8"]
    assert results["phase9"]["objective_evaluations"] == EXPECTED_COUNTS["phase9"]
    assert results["phase10"]["objective_evaluations"] == EXPECTED_COUNTS["phase10"]

    strict = {
        row["polarization_type"]: row["blind_to_strongest_known_cm_ratio"]
        for row in results["phase10"]["largest_radius_results"]
    }
    assert set(strict) == {"(1,3)", "(1,5)", "(1,1,2)", "(1,1,3)", "(1,2,2)"}
    assert all(0.0 < float(value) < 1.0 for value in strict.values())


def check_figures() -> None:
    for stem in FIGURE_STEMS:
        for suffix in (".png", ".pdf"):
            path = ROOT / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 1_000, path


def check_notebooks() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    assert len(notebooks) == 10
    for path in notebooks:
        notebook = json.loads(path.read_text())
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells and all(cell["execution_count"] is not None for cell in code_cells), path
        for cell in code_cells:
            assert all(output.get("output_type") != "error" for output in cell.get("outputs", [])), path


def check_release_paths() -> None:
    text_files = [REPO / "README.md", ROOT / "README.md"]
    text_files += sorted((ROOT / "docs").glob("*.md"))
    text_files += sorted((ROOT / "notebooks").glob("*.ipynb"))
    forbidden = ("/Users/", "cm-passive-cliffords")
    for path in text_files:
        content = path.read_text()
        assert not any(token in content for token in forbidden), path


def check_local_markdown_links() -> None:
    markdown_files = [REPO / "README.md", ROOT / "README.md", ROOT / "data/README.md"]
    markdown_files += sorted((ROOT / "docs").glob("*.md"))
    markdown_files += [ROOT / "notebooks/README.md"]
    missing: list[tuple[Path, str]] = []
    for source in markdown_files:
        for raw_target in LOCAL_LINK.findall(source.read_text()):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if target and not (source.parent / target).resolve().exists():
                missing.append((source.relative_to(REPO), raw_target))
    assert not missing, missing


def main() -> None:
    check_consolidated_results()
    check_figures()
    check_notebooks()
    check_release_paths()
    check_local_markdown_links()
    print("Release integrity checks passed.")


if __name__ == "__main__":
    main()
