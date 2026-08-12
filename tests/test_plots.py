"""The figures must regenerate from committed results with no manual steps."""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUTS = ("sweep.json", "sweep_curves.npz", "gate_sweep.json")
FIGURES = (
    "1_headline_decoupling",
    "2_the_fix",
    "3_nees_vs_time",
    "4_gate_effect",
)


def _run(results_dir: pathlib.Path):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot.py"), "--results", str(results_dir)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


@pytest.mark.skipif(
    not all((RESULTS / f).exists() for f in INPUTS),
    reason="results not generated; run scripts/sweep.py and scripts/gate_sweep.py",
)
def test_all_figures_regenerate(tmp_path):
    for f in INPUTS:
        shutil.copy(RESULTS / f, tmp_path / f)

    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "skip:" not in proc.stdout, f"a figure was skipped:\n{proc.stdout}"

    figdir = tmp_path / "figures"
    for name in FIGURES:
        png = figdir / f"{name}.png"
        txt = figdir / f"{name}.txt"
        assert png.exists() and png.stat().st_size > 5000, f"{name}.png missing or tiny"
        # Every figure carries a caption, and no figure exists without a stated N.
        assert txt.exists(), f"{name}.txt caption missing"
        caption = txt.read_text()
        # Every figure must state how many runs it is built from. The wording is
        # free; the number is not optional.
        assert re.search(r"\b\d{2,}\s+runs\b|\bN\s*=\s*\d+", caption), (
            f"{name} caption does not state its sample size"
        )
        assert len(caption) > 150, f"{name} caption is too thin to be useful"


def test_missing_inputs_are_skipped_not_crashed(tmp_path):
    """A partial results directory should produce what it can and say what is
    missing, rather than failing the whole run."""
    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "skip:" in proc.stdout
    assert "scripts/sweep.py" in proc.stdout, "should name the command that fixes it"
