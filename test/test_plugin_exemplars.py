"""Run the agent plugin's exemplar models and require every built-in check to pass.

The exemplars under ``plugins/pymrm/skills/conventions/exemplars`` are what coding
agents copy. Each exposes ``run_checks()`` returning ``(results, passed)``.
"""

import importlib.util
from pathlib import Path

import pytest

EXEMPLAR_DIR = Path(__file__).resolve().parents[1] / "plugins" / "pymrm" / "skills" / "conventions" / "exemplars"
EXEMPLARS = sorted(EXEMPLAR_DIR.glob("*.py"))


def test_exemplars_found():
    assert len(EXEMPLARS) >= 3


@pytest.mark.parametrize("path", EXEMPLARS, ids=lambda p: p.stem)
def test_exemplar_checks_pass(path):
    spec = importlib.util.spec_from_file_location(f"exemplar_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results, passed = module.run_checks(verbose=False)
    failed = {key: results[key] for key, ok in passed.items() if not ok}
    assert not failed, f"{path.name}: failed checks {failed}"
