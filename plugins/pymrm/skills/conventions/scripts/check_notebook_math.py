"""Check that the equations in notebook Markdown cells render everywhere.

Usage:
    python check_notebook_math.py notebook.ipynb [more.ipynb ...]

Notebooks are read in JupyterLab (MathJax), VS Code (KaTeX), the GitHub preview
(Markdown is processed before the maths) and Colab. This script checks every
Markdown cell against a portable subset (style guide section 3.1):

- display maths as $$ ... $$ with the $$ lines separated from text by blank lines;
- inline maths $...$ without a space just inside the dollars and without \\ ;
- environments (aligned, cases, array, ...) only inside $$, never a bare
  \\begin{align} or \\begin{equation};
- no cross-references (\\label, \\ref, \\eqref); at most one \\tag per display
  block and none in inline maths;
- no macros (\\newcommand, \\renewcommand, \\def) and no \\require;
- no maths in headings; no | inside maths in a table row (use \\vert).

Then every expression is parsed with KaTeX, the strictest of the engines, when
Node.js and KaTeX are available (KaTeX is looked for in $PYMRM_KATEX_DIR,
default ~/.cache/pymrm-katex; install with
`npm install --prefix ~/.cache/pymrm-katex katex@0.16`). Without them the
parse step is skipped and the report says so.

Exit status 1 when anything is flagged.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FORBIDDEN = {r"\label": "equation labels do not render in KaTeX (VS Code)",
             r"\eqref": "cross-references do not render in KaTeX (VS Code)",
             r"\ref": "cross-references do not render in KaTeX (VS Code)",
             r"\newcommand": "macros do not carry over between cells in every renderer",
             r"\renewcommand": "macros do not carry over between cells in every renderer",
             r"\def": "macros do not carry over between cells in every renderer",
             r"\require": "MathJax-only extension loading",
             r"\bm": "not in KaTeX; use \\boldsymbol"}
INLINE = re.compile(r"(?<![\\$])\$(?!\$)((?:[^$\\]|\\.)+?)(?<![\\$])\$(?!\$)", re.S)
BARE_ENV = re.compile(r"^\s*\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?)\}", re.M)


def _strip_code(text):
    text = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def _cell_source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def extract(markdown):
    """Return (expressions, problems) for one Markdown cell."""
    text = _strip_code(markdown)
    problems, expressions = [], []
    lines = text.split("\n")

    # display maths: $$ ... $$
    for match in re.finditer(r"\$\$(.+?)\$\$", text, flags=re.S):
        body = match.group(1)
        expressions.append((body.strip(), True))
        start, end = match.start(), match.end()
        before = text[:start].rstrip(" ")
        after = text[end:].lstrip(" ")
        if before and not before.endswith("\n\n") and not before.endswith("\n"):
            problems.append(("display", body, "$$ does not start on its own line"))
        elif before.endswith("\n") and not before.endswith("\n\n") and before.strip():
            problems.append(("display", body, "no blank line before $$ (GitHub may not render it)"))
        if after and not after.startswith("\n"):
            problems.append(("display", body, "text directly after the closing $$"))
        elif after.startswith("\n") and not after.startswith("\n\n") and after.strip():
            problems.append(("display", body, "no blank line after $$ (GitHub may not render it)"))
    without_display = re.sub(r"\$\$(.+?)\$\$", " ", text, flags=re.S)

    for m in BARE_ENV.finditer(without_display):
        problems.append(("environment", m.group(0).strip(), "environment outside $$; use aligned/cases inside $$"))

    # inline maths: $...$, may wrap over a line break but not over a blank line
    for paragraph in re.split(r"\n\s*\n", without_display):
        stars = 0
        for m in INLINE.finditer(paragraph):
            body = m.group(1)
            expressions.append((body, False))
            if body != body.strip():
                problems.append(("inline", body, "space just inside the $ delimiters"))
            if "\\\\" in body:
                problems.append(("inline", body, "line break \\\\ in inline maths"))
            stars += body.count("*")
            if re.search(r"\\tag(?![A-Za-z])", body):
                problems.append(("inline", body, "\\tag in inline maths"))
        if stars >= 2:
            problems.append(("inline", paragraph.strip()[:80],
                             "two or more * in inline maths in one paragraph can become italics on GitHub; "
                             "use \\cdot or \\ast"))

    for line in lines:
        if re.match(r"\s*#", line) and "$" in line:
            problems.append(("heading", line.strip(), "maths in a heading"))
        if line.lstrip().startswith("|"):
            for m in INLINE.finditer(line):
                if "|" in m.group(1):
                    problems.append(("table", line.strip(), "| inside maths in a table row; use \\vert"))

    for body, display in expressions:
        if display and len(re.findall(r"\\tag(?![A-Za-z])", body)) > 1:
            problems.append(("command", body, "more than one \\tag in a display block"))
        for command, why in FORBIDDEN.items():
            if re.search(re.escape(command) + r"(?![A-Za-z])", body):
                problems.append(("command", body, f"{command}: {why}"))
    return expressions, problems


def katex_errors(expressions):
    katex_dir = Path(os.environ.get("PYMRM_KATEX_DIR", Path.home() / ".cache" / "pymrm-katex"))
    module = katex_dir / "node_modules" / "katex"
    node = shutil.which("node")
    if node is None or not module.is_dir():
        return None
    script = (
        "const k=require(process.argv[1]);let d='';process.stdin.on('data',c=>d+=c);"
        "process.stdin.on('end',()=>{const out=JSON.parse(d).map(([t,disp])=>{"
        "try{k.renderToString(t,{displayMode:disp,throwOnError:true,strict:'ignore'});return null;}"
        "catch(e){return e.message.split('\\n')[0];}});console.log(JSON.stringify(out));});"
    )
    run = subprocess.run([node, "-e", script, str(module)], input=json.dumps(expressions),
                         capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def check(path):
    notebook = json.loads(Path(path).read_text())
    findings, all_expressions, owners = [], [], []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        expressions, problems = extract(_cell_source(cell))
        findings += [(index, kind, snippet, why) for kind, snippet, why in problems]
        all_expressions += [[body, display] for body, display in expressions]
        owners += [index] * len(expressions)
    errors = katex_errors(all_expressions)
    if errors is None:
        note = "KaTeX parse skipped (node or KaTeX not found; see the module docstring)"
    else:
        note = f"KaTeX parsed {len(all_expressions)} expressions"
        for owner, (body, _), error in zip(owners, all_expressions, errors):
            if error:
                findings.append((owner, "katex", body, error))
    return findings, note


def main(paths):
    status = 0
    for path in paths:
        findings, note = check(path)
        print(f"{path}: {len(findings)} finding(s); {note}")
        for cell, kind, snippet, why in findings:
            short = " ".join(snippet.split())
            print(f"  cell {cell} [{kind}] {why}: {short[:90]}")
        status |= bool(findings)
    return int(status)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
