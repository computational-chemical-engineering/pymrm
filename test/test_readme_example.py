"""The README's minimal model must run and give a converged, correct answer."""
import re
from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_minimal_model_runs():
    text = README.read_text()
    section = text.split("## Conventions for users and coding agents", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, flags=re.S).group(1)
    namespace = {}
    exec(compile(code, "README.md", "exec"), namespace)
    c = namespace["c"]
    assert namespace["result"].success
    assert c.shape == (100, 1) and 0.0 < c[0, 0] < c[-1, 0] < 1.0
    from pymrm.checks import residual_check
    assert residual_check(namespace["residual"], namespace["result"].x)["ok"]
