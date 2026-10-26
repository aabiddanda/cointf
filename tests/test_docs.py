"""Run the Python examples in the README and docs/ so they cannot go stale.

Blocks that only show a signature (no executable statements) are skipped.
"""

import re
from pathlib import Path

import pytest

root = Path(__file__).parent.parent
pages = [
    root / "README.md",
    *sorted(p for p in (root / "docs").rglob("*.md") if "_build" not in p.parts),
]
signature_only = re.compile(r"^\s*\w+\(.*\)\s*->|^\s*loglik_fast\(data, L, nu, p, \*")


def python_blocks(path):
    """Return the fenced Python code blocks of a Markdown file."""
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
    return [b for b in blocks if not signature_only.match(b)]


@pytest.mark.parametrize("path", pages, ids=lambda p: p.name)
def test_examples_run(path):
    # Blocks on a page build on each other, so run them in one namespace.
    namespace = {}
    for block in python_blocks(path):
        exec(compile(block, str(path), "exec"), namespace)
