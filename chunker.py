# pip install langchain-core

import re
import json
from pathlib import Path
from langchain_core.documents import Document

PARENT_RE = re.compile(r"^===PARENT===\s*(.*?)\s*$", re.M)
DOC_RE    = re.compile(r"^DOC:\s*(.+)$", re.M)
IMG_RE = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+)[^)]*\)")
CHILD_DELIM = "===CHILD==="


def parse_file(path: Path):
    """Parse one markdown file -> (parents list, children list of Documents)."""
    raw = path.read_text(encoding="utf-8")

    if not DOC_RE.search(raw) or not PARENT_RE.search(raw):
        raise ValueError(f"{path.name}: no 'DOC:' line or '===PARENT===' markers found")

    doc_id   = path.stem
    doc_name = DOC_RE.search(raw).group(1).strip()

    parts = PARENT_RE.split(raw)  # [header, title1, body1, title2, body2, ...]
    parents, children = [], []

    for i in range(1, len(parts), 2):
        title = parts[i].strip().rstrip(":")
        body  = parts[i + 1]
        parent_id = f"{doc_id}_p{i // 2}"

        # ---- parent: full section text (delimiters removed) ----
        parents.append({
            "parent_id": parent_id,
            "doc_id":    doc_id,
            "doc_name":  doc_name,
            "title":     title,
            "text":      body.replace(CHILD_DELIM, "\n").strip(),
        })

        # ---- children: one per delimited block ----
        chunks = [c.strip() for c in body.split(CHILD_DELIM) if c.strip()]
        if not chunks:
            print(f"  WARNING: parent '{title}' in {path.name} has no children")

        for j, text in enumerate(chunks):
            images = [{"alt": a.strip(), "path": p.strip()}
                      for a, p in IMG_RE.findall(text)]
            clean_text = IMG_RE.sub("", text).strip()
            embed_text = f"سند: {doc_name} | بخش: {title} | {clean_text}"
            children.append(Document(
                page_content=embed_text,
                metadata={
                    "child_id":     f"{parent_id}_c{j}",
                    "parent_id":    parent_id,
                    "doc_id":       doc_id,
                    "doc_name":     doc_name,
                    "section":      title,
                    "chunk_index":  j,
                    "display_text": clean_text,
                    "images":       images,          # ← NEW
                },
            ))

    return parents, children


def chunk_directory(folder):
    folder = Path(folder)
    all_parents, all_children = {}, []

    for md in sorted(folder.glob("*.md")):
        try:
            parents, children = parse_file(md)
        except ValueError as e:
            print(f"SKIPPED -> {e}")
            continue
        for p in parents:
            all_parents[p["parent_id"]] = p
        all_children.extend(children)
        print(f"OK: {md.name:<40} {len(parents)} parents, {len(children)} children")

    return all_parents, all_children


def save(parents, children, out_dir="chunked"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    (out / "parents.json").write_text(
        json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "children.json").write_text(
        json.dumps(
            [{"page_content": c.page_content, "metadata": c.metadata} for c in children],
            ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"Saved -> {out/'parents.json'} , {out/'children.json'}")


def report(children):
    lengths = [len(c.metadata["display_text"]) for c in children]
    print(f"\nTotal children : {len(children)}")
    print(f"Unique parents : {len({c.metadata['parent_id'] for c in children})}")
    print(f"Unique docs    : {len({c.metadata['doc_id'] for c in children})}")
    print(f"Child size     : min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)} chars")


if __name__ == "__main__":
    parents, children = chunk_directory("docs")   # put your .md files in ./docs
    save(parents, children)
    report(children)

    # sanity check: look at one child
    print("\n--- sample child ---")
    print(children[0].page_content[:300])