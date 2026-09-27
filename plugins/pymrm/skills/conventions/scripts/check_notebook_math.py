"""Check that the equations in notebook Markdown cells render everywhere.

Usage:
    python check_notebook_math.py notebook.ipynb [more.ipynb ...]

Notebooks are read in JupyterLab (MathJax), VS Code (KaTeX), the GitHub preview
(Markdown is processed before the maths) and Colab. No single engine is the
strictest (KaTeX accepts \\bm, JupyterLab's MathJax does not), so this script
combines a KaTeX parse with rules for what the others need. It checks every
Markdown cell against a portable subset (style guide section 3.1):

- display maths as $$ ... $$;
- inline maths $...$ without a space just inside the dollars and without \\ ;
- environments (aligned, cases, array, ...) only inside $$, never a bare
  \\begin{align} or \\begin{equation};
- no cross-references (\\label, \\ref, \\eqref); at most one \\tag per display
  block and none in inline maths;
- no macros (\\newcommand, \\renewcommand, \\def) and no \\require;
- no maths in headings; no | inside maths in a table row (use \\vert).

Then every expression is parsed with KaTeX (the engine of VS Code) when
Node.js and KaTeX are available (KaTeX is looked for in $PYMRM_KATEX_DIR,
default ~/.cache/pymrm-katex; install with
`npm install --prefix ~/.cache/pymrm-katex katex@0.16`). Without them the
parse step is skipped and the report says so.

Findings are errors or warnings. Errors are confirmed failures (\\label in VS
Code, \\eqref on GitHub, KaTeX parse errors, \\bm in JupyterLab). Warnings are
advisable but rendered in the test notebook, or not covered by it.
Exit status 1 when an error is flagged.
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
             r"\bm": "not in MathJax 3 as configured in JupyterLab; use \\boldsymbol"}
INLINE = re.compile(r"(?<![\\$])\$(?!\$)((?:[^$\\]|\\.)+?)(?<![\\$])\$(?!\$)", re.S)
BARE_ENV = re.compile(r"^\s*\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?)\}", re.M)


# Confirmed by rendering docs/math-render-test.ipynb (2026-09-27): on GitHub only
# \eqref failed, in VS Code only \label; blank lines around $$, a bare align
# environment, \\ and * in inline maths and | in tables rendered. Those are
# warnings (advisable), the confirmed failures and KaTeX parse errors are errors.
WARNINGS = ("environment outside $$", "line break", "two or more *", "| inside maths",
            "maths in a heading", "\\newcommand", "\\renewcommand", "\\def")


def _blank_out(match):
    return "\n" * match.group(0).count("\n")


def _strip_code(text):
    """Remove everything that is not Markdown prose: code, comments, URLs, quote markers."""
    text = re.sub(r"^(```|~~~).*?^\1[^\n]*$", _blank_out, text, flags=re.S | re.M)   # fenced code
    text = re.sub(r"<!--.*?-->", _blank_out, text, flags=re.S)                         # HTML comments
    text = re.sub(r"(?m)^[ \t]*>[ \t]?", "", text)                                   # blockquote markers
    text = re.sub(r"\]\([^)]*\)", "]", text)                                           # link targets
    text = re.sub(r"(`+)(.+?)\1", "", text, flags=re.S)                                # code spans
    blocks = re.split(r"(\n[ \t]*\n)", text)                                           # indented code blocks
    for i, block in enumerate(blocks):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if lines and all(re.match(r"( {4}|\t)", ln) for ln in lines) and not re.match(r"\s*([-*+]|\d+\.)\s", lines[0]):
            blocks[i] = "\n" * block.count("\n")
    return "".join(blocks)


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
        expressions.append((match.group(1).strip(), True))
    without_display = re.sub(r"\$\$(.+?)\$\$", " ", text, flags=re.S)

    prose = INLINE.sub(" ", without_display)
    for m in re.finditer(r"\\(eqref|ref)\{[^}]*\}", prose):
        problems.append(("command", m.group(0), f"\\{m.group(1)} in the text does not render on GitHub"))
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
            starred = [m.group(1) for m in INLINE.finditer(paragraph) if "*" in m.group(1)]
            problems.append(("inline", "  ".join(f"${b}$" for b in starred),
                             "two or more * in inline maths in one paragraph may become italics in some "
                             "Markdown renderers; \\ast or ^{\\ast} is safer"))

    for line in lines:
        if re.match(r"\s*#", line) and re.search(r"(?<!\\)\$", line):
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


def is_warning(why):
    return why.startswith(WARNINGS)


def main(paths):
    status = 0
    for path in paths:
        findings, note = check(path)
        errors = [f for f in findings if not is_warning(f[3])]
        print(f"{path}: {len(errors)} error(s), {len(findings) - len(errors)} warning(s); {note}")
        for cell, kind, snippet, why in findings:
            short = " ".join(snippet.split())
            level = "warning" if is_warning(why) else "error"
            print(f"  cell {cell} {level} [{kind}] {why}: {short[:90]}")
        status |= bool(errors)
    return int(status)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
