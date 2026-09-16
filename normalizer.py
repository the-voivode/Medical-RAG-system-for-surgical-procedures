"""
normalizer.py
-------------
Reads every .md file in the docs/ folder, normalizes the Persian text using hazm,
and writes cleaned versions to docs_normalized/ (originals are preserved).

The script is structure-aware: it keeps your RAG delimiters (DOC:, ===PARENT===,
===CHILD===) byte-identical and only normalizes the content text.

Install:  pip install hazm
Run:      python normalizer.py
"""

import re
from pathlib import Path
from hazm import Normalizer

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR  = Path("docs")
OUTPUT_DIR = Path("docs_normalized")   # ⚠️ change to Path("docs") to overwrite in place

# ============================================================
# HAZM SETUP
# Defaults perform: character refinement (ي→ی, ك→ک, ة→ه, Arabic digits→Persian),
# punctuation spacing, and affix/ZWNJ (نیم‌فاصله) fixing for می‌/ها/تر etc.
# ============================================================
normalizer = Normalizer()

# Structural markers (must survive normalization unchanged)
DOC_RE    = re.compile(r"^\s*(DOC:)\s*(.*)$")
PARENT_RE = re.compile(r"^\s*(===PARENT===)\s*(.*)$")
CHILD_RE  = re.compile(r"^\s*(===CHILD===)\s*$")
IMG_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")


def normalize_text(text: str) -> str:
    """Normalize a plain chunk of Persian text. Reusable for queries later."""
    return normalizer.normalize(text).strip()


def normalize_line(line: str) -> str:
    """Normalize one line while preserving RAG structure markers."""
    # DOC: <name>  -> keep marker, normalize the name
    m = DOC_RE.match(line)
    if m:
        return f"{m.group(1)} {normalize_text(m.group(2))}"

    # ===PARENT=== <title>  -> keep marker, normalize the title
    m = PARENT_RE.match(line)
    if m:
        return f"{m.group(1)} {normalize_text(m.group(2))}"

    # ===CHILD===  -> keep exactly as-is
    if CHILD_RE.match(line):
        return "===CHILD==="
    
    # 🖼️ Markdown image line — preserve EXACTLY as-is
    if IMG_RE.match(line):
        return line
    
    # Empty / whitespace-only line
    if not line.strip():
        return ""

    # Normal content line -> preserve leading indent, normalize the rest
    indent  = line[: len(line) - len(line.lstrip())]
    content = normalizer.normalize(line.strip())
    return indent + content


def normalize_file(src: Path, dst: Path) -> tuple[int, int]:
    """Normalize one file. Returns (total_lines, changed_lines)."""
    text  = src.read_text(encoding="utf-8-sig")   # utf-8-sig handles BOM safely
    lines = text.split("\n")

    normalized = [normalize_line(ln) for ln in lines]
    changed    = sum(1 for a, b in zip(lines, normalized) if a != b)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(normalized), encoding="utf-8")
    return len(lines), changed


def main():
    if not INPUT_DIR.exists():
        print(f"❌ Input folder not found: {INPUT_DIR.resolve()}")
        return

    md_files = sorted(INPUT_DIR.glob("*.md"))
    if not md_files:
        print(f"⚠️  No .md files found in {INPUT_DIR.resolve()}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Normalizing {len(md_files)} files: '{INPUT_DIR}' → '{OUTPUT_DIR}'\n")

    total_changed = 0
    for md in md_files:
        dst = OUTPUT_DIR / md.name
        n_lines, n_changed = normalize_file(md, dst)
        total_changed += n_changed
        print(f"✅ {md.name:<45} {n_lines:>4} lines, {n_changed:>3} changed")

    print(f"\n🎉 Done. {total_changed} lines modified across {len(md_files)} files.")
    print(f"👉 Review the output in '{OUTPUT_DIR}/', then point your chunker at this folder.")


if __name__ == "__main__":
    main()