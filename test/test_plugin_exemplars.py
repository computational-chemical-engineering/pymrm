"""Run the agent plugin's exemplar models and require every built-in check to pass.

The exemplars under ``plugins/pymrm/skills/conventions/exemplars`` are what coding
agents copy. Each exposes ``run_checks()`` returning ``(results, passed)``.
"""

import importlib.util
from pathlib import Path

import pytest

EXEMPLAR_DIR = Path(__file__).resolve().parents[1] / "plugins" / "pymrm" / "skills" / "conventions" / "exemplars"
PATTERN_DIR = EXEMPLAR_DIR.parents[1] / "model-patterns" / "exemplars"
EXEMPLARS = sorted(EXEMPLAR_DIR.glob("*.py")) + sorted(PATTERN_DIR.glob("*.py"))


def test_exemplars_found():
    assert len(EXEMPLARS) >= 7


@pytest.mark.parametrize("path", EXEMPLARS, ids=lambda p: p.stem)
def test_exemplar_checks_pass(path):
    spec = importlib.util.spec_from_file_location(f"exemplar_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results, passed = module.run_checks(verbose=False)
    failed = {key: results[key] for key, ok in passed.items() if not ok}
    assert not failed, f"{path.name}: failed checks {failed}"


CHECKER = EXEMPLAR_DIR.parent / "scripts" / "check_model.py"

BAD_MODEL = """
import numpy as np
from scipy.sparse import csc_array
from pymrm import NumJac, newton
NumJac((150,))(lambda c: -c**2, np.ones(150))
s = 1e-20
newton(lambda x: (np.array([x[0]**2 - s]), csc_array(np.array([[2 * x[0]]]))), np.array([1e-8]))
"""


def test_checker_flags_pitfalls(tmp_path):
    import subprocess
    import sys
    bad = tmp_path / "bad_model.py"
    bad.write_text(BAD_MODEL)
    out = subprocess.run([sys.executable, str(CHECKER), str(bad)], capture_output=True, text=True)
    assert out.returncode == 1
    assert "[P3]" in out.stdout and "[P7]" in out.stdout


def test_checker_silent_on_exemplar():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-W", "ignore", str(CHECKER), str(EXEMPLAR_DIR / "steady_pellet.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "no pitfall symptoms" in out.stdout


def test_plugin_copy_of_checks_is_identical():
    """The plugin ships a copy of pymrm.checks for environments with an older pymrm."""
    library = Path(__file__).resolve().parents[1] / "src" / "pymrm" / "checks.py"
    copy = EXEMPLAR_DIR.parent / "scripts" / "pymrm_checks.py"
    assert library.read_bytes() == copy.read_bytes(), "run: cp src/pymrm/checks.py " + str(copy)


@pytest.mark.parametrize("notebook", sorted(EXEMPLAR_DIR.glob("*.ipynb")) + sorted(PATTERN_DIR.glob("*.ipynb")),
                         ids=lambda p: p.stem)
def test_exemplar_notebooks_execute(notebook):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(notebook, as_version=4)
    nbclient.NotebookClient(nb, timeout=300, kernel_name="python3").execute()
