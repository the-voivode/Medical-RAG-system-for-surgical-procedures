"""
bm25builder.py
---------------
Builds a Persian BM25 keyword index over your chunked children using
hazm (normalization + tokenization) and rank_bm25 (scoring).

Install:   pip install hazm rank_bm25
Test:      python bm25builder.py
Import:    from bm25builder import PersianBM25
"""

import json
import pickle
from pathlib import Path
from typing import List, Any, Tuple

from hazm import Normalizer, word_tokenize
from rank_bm25 import BM25Okapi

# Stopwords list (API differs across hazm versions -> load defensively)
try:
    from hazm import stopwords_list
    _STOPWORDS = set(stopwords_list())
except Exception:
    _STOPWORDS = set()


class PersianBM25:
    def __init__(self, use_stopwords: bool = True):
        self.normalizer = Normalizer()
        self.use_stopwords = use_stopwords
        self.stopwords = _STOPWORDS if use_stopwords else set()
        self.bm25: BM25Okapi | None = None
        self.docs: List[Any] = []            # LangChain Documents, same order as built

    # ---------- tokenization ----------
    def tokenize(self, text: str) -> List[str]:
        """Normalize (ZWNJ, ي/ک, digits) then tokenize; drop stopwords & punctuation."""
        text = self.normalizer.normalize(text)
        tokens = word_tokenize(text)
        out = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            if self.use_stopwords and t in self.stopwords:
                continue
            if not any(ch.isalnum() for ch in t):   # skip pure punctuation
                continue
            out.append(t)
        return out

    # ---------- build / search ----------
    def build(self, documents: List[Any], text_key: str = "display_text"):
        """Build the BM25 index from LangChain Documents (uses metadata[text_key])."""
        self.docs = documents
        corpus = []
        for d in documents:
            toks = self.tokenize(d.metadata.get(text_key, ""))
            corpus.append(toks if toks else ["<empty>"])   # guard empty docs
        self.bm25 = BM25Okapi(corpus)
        return self

    def search(self, query: str, k: int = 5) -> List[Tuple[Any, float]]:
        """Return top-k (Document, bm25_score) ranked by keyword relevance."""
        if self.bm25 is None:
            raise ValueError("BM25 not built. Call build() or load() first.")
        q_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # drop zero-overlap results
        return [(self.docs[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    # ---------- persistence ----------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "use_stopwords": self.use_stopwords}, f)

    def load(self, path: str, documents: List[Any]):
        """Load index. `documents` must be the SAME list/order used at build time."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.use_stopwords = data["use_stopwords"]
        self.docs = documents
        return self


# ---------- standalone test ----------
if __name__ == "__main__":
    from langchain_core.documents import Document

    with open("chunked/children.json", "r", encoding="utf-8") as f:
        children_data = json.load(f)
    docs = [Document(page_content=c["page_content"], metadata=c["metadata"])
            for c in children_data]

    bm25 = PersianBM25().build(docs)

    for q in [
        "کاتانوئید چیست و چرا در کرانیوتومی مهم است؟",
        "ابزار مرحله ایجاد تونل فمورال",
        "مد کوتر SWIFT COAG چه کاربردی دارد؟",
    ]:
        print(f"\n🔍 BM25 Query: {q}")
        for i, (doc, score) in enumerate(bm25.search(q, k=3), 1):
            print(f"  {i}. [{score:.3f}] {doc.metadata['doc_name']} | "
                  f"{doc.metadata['display_text'][:80]}...")