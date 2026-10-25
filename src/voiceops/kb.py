"""Load the knowledge base into section-level chunks.

Every KB file has a front-matter header (doc_id, doc_type, status, effective_from, ...)
and `## [SECTION-ID] Title` headings. One chunk = one section, so the section ID that
retrieval is scored on travels with the text. Product catalog rows are indexed too,
with IDs like `products.csv:ELE-0012`, because some answers need a catalog row.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

# KB-00.1: policy > agent guide > help > FAQ > marketing.
PRECEDENCE = {"policy": 0, "agent_guide": 1, "help": 2, "faq": 3, "catalog": 3, "marketing": 4, "meta": 1}

SECTION_RE = re.compile(r"^(#{2,4})\s*\[([^\]]+)\]\s*(.*)$")


@dataclass
class Chunk:
    id: str                      # section ID, e.g. RET-2, or products.csv:ELE-0012
    doc: str                     # file name
    title: str                   # section title
    text: str                    # section body (markdown)
    meta: dict = field(default_factory=dict)

    @property
    def doc_type(self) -> str:
        return self.meta.get("doc_type", "")

    @property
    def superseded(self) -> bool:
        return self.meta.get("status", "current") != "current"

    @property
    def rank(self) -> int:
        return PRECEDENCE.get(self.doc_type, 5)

    def header(self) -> str:
        m = self.meta
        bits = [f"[{self.id}] {self.title}", f"source: {self.doc}", f"type: {self.doc_type}"]
        if m.get("status"):
            bits.append(f"status: {m['status']}")
        if m.get("effective_from"):
            bits.append(f"effective_from: {m['effective_from']}")
        if m.get("superseded_by"):
            bits.append(f"superseded_by: {m['superseded_by']}")
        return " | ".join(bits)

    def render(self) -> str:
        return f"{self.header()}\n{self.text.strip()}"

    def search_text(self) -> str:
        return f"{self.meta.get('title', '')} - {self.title}\n{self.text}"


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    meta = {}
    for line in raw[3:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[end + 4:]


def load_kb_file(path: Path) -> list[Chunk]:
    meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    preamble: list[str] = []
    current: Chunk | None = None
    for line in body.splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = Chunk(id=m.group(2).strip(), doc=path.name, title=m.group(3).strip(), text="", meta=meta)
            chunks.append(current)
        elif current is not None:
            current.text += line + "\n"
        elif not line.startswith("# "):
            preamble.append(line)
    # A note above the first section (e.g. "this version was replaced ...") matters for every section.
    note = "\n".join(preamble).strip()
    if note:
        for c in chunks:
            c.text = note + "\n\n" + c.text
    return chunks


def load_catalog_chunks() -> list[Chunk]:
    path = config.DATA_DIR / "products.csv"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = "; ".join(f"{k}: {v}" for k, v in row.items() if v != "")
            out.append(Chunk(id=f"products.csv:{row['sku']}", doc="products.csv", title=row.get("title", row["sku"]),
                             text=text, meta={"doc_type": "catalog", "status": "current", "title": "Product catalog"}))
    return out


def load_chunks() -> list[Chunk]:
    if not config.KB_DIR.is_dir():
        raise FileNotFoundError(
            f"Knowledge base not found at {config.KB_DIR}. Extract the capstone ZIP into the project "
            "folder or set DATASET_DIR in .env.")
    chunks = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        chunks.extend(load_kb_file(path))
    chunks.extend(load_catalog_chunks())
    return chunks


def always_on_rules(chunks: list[Chunk]) -> list[Chunk]:
    """Agent guides and the KB meta doc are operating rules for every turn, not lookups."""
    return [c for c in chunks if c.doc_type in ("agent_guide", "meta")]
