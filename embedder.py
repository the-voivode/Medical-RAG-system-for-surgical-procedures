import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from config import settings
from bm25builder import PersianBM25
from normalizer import normalize_text
from difflib import SequenceMatcher

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

LOG_FILE = Path("rag_eval_log.txt")


class ParentChildRAG:
    def __init__(self):
        self.children_path = Path(settings.children_path)
        self.parents_path = Path(settings.parents_path)
        self.index_dir = Path(settings.index_dir)
        self.model_name = settings.embedding_model
        self.device = settings.device

        self.documents: List[Document] = []
        self.parents: Dict[str, Dict[str, Any]] = {}
        self.embeddings = None
        self.vectorstore = None

    # ------------------------------------------------------------------ data
    def load_data(self):
        print(f"📂 Loading data from {self.children_path} and {self.parents_path}...")
        with open(self.parents_path, "r", encoding="utf-8") as f:
            self.parents = json.load(f)
        with open(self.children_path, "r", encoding="utf-8") as f:
            children_data = json.load(f)
        self.documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in children_data
        ]
        print(f"✅ Loaded {len(self.documents)} children and {len(self.parents)} parents.")

    # ------------------------------------------------------------ embeddings
    def setup_embeddings(self):
        print(f"\n🧠 Initializing embedding model: {self.model_name}")
        print("   (If this is the first time, the model weights will download.)")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ Embedding model ready.")

    # --------------------------------------------------------------- index
    def build_or_load_index(self, batch_size: int = 32):
        if self.index_dir.exists() and (self.index_dir / "index.faiss").exists():
            print(f"\n💾 Found existing FAISS index at {self.index_dir}. Loading...")
            self.vectorstore = FAISS.load_local(
                str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True
            )
            print("✅ Index loaded successfully.")
        else:
            print(f"\n🏗️  Building new FAISS index for {len(self.documents)} documents...")
            self.index_dir.mkdir(parents=True, exist_ok=True)
            texts = [doc.page_content for doc in self.documents]
            metadatas = [doc.metadata for doc in self.documents]
            embeddings_list = []
            for i in tqdm(range(0, len(texts), batch_size), desc="🔄 Embedding batches", unit="batch"):
                batch = texts[i : i + batch_size]
                embeddings_list.extend(self.embeddings.embed_documents(batch))
            text_embedding_pairs = list(zip(texts, embeddings_list))
            print("📦 Constructing FAISS vector index...")
            self.vectorstore = FAISS.from_embeddings(text_embedding_pairs, self.embeddings, metadatas=metadatas)
            self.vectorstore.save_local(str(self.index_dir))
            print(f"✅ Index built and saved to {self.index_dir}/.")

    # ---------------------------------------------------------------- bm25
    def build_or_load_bm25(self):
        self.bm25_retriever = PersianBM25(use_stopwords=True)
        bm25_path = Path(settings.bm25_index_path)
        if bm25_path.exists():
            self.bm25_retriever.load(str(bm25_path), self.documents)
            print(f"💾 Loaded BM25 index from {bm25_path}")
        else:
            self.bm25_retriever.build(self.documents, text_key="page_content")  # ← FIX
            self.bm25_retriever.save(str(bm25_path))
            print(f"🏗️  Built & saved BM25 index to {bm25_path}")
    # def build_or_load_bm25(self):
    #     self.bm25_retriever = PersianBM25(use_stopwords=True)
    #     bm25_path = Path(settings.bm25_index_path)
    #     if bm25_path.exists():
    #         self.bm25_retriever.load(str(bm25_path), self.documents)
    #         print(f"💾 Loaded BM25 index from {bm25_path}")
    #     else:
    #         self.bm25_retriever.build(self.documents)
    #         self.bm25_retriever.save(str(bm25_path))
    #         print(f"🏗️  Built & saved BM25 index to {bm25_path}")

    # -------------------------------------------------------- hybrid search
    def hybrid_search(self, query: str, k: int = None):
        k = k or settings.top_k
        cand = settings.hybrid_candidate_k
        dense_hits = self.vectorstore.similarity_search_with_score(query, k=cand)
        bm25_hits = self.bm25_retriever.search(query, k=cand)

        print("Dense hits:", len(dense_hits))
        print("BM25 hits:", len(bm25_hits))

        fused = {}
        legs = {}                                   # ← NEW: which leg found each candidate

        for rank, (doc, _) in enumerate(dense_hits, start=1):
            cid = doc.metadata["child_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank)
            legs.setdefault(cid, set()).add("dense")
        for rank, (doc, _) in enumerate(bm25_hits, start=1):
            cid = doc.metadata["child_id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank)
            legs.setdefault(cid, set()).add("bm25")
        # ---- consensus gate: only candidates found by BOTH legs ----
        # both = [cid for cid in fused if len(legs[cid]) == 2]
        # ranked_ids = sorted(both, key=fused.get, reverse=True)[:k]
                # prefer candidates found by BOTH legs, but never drop single-leg candidates
        ranked_ids = sorted(
            fused,
            key=lambda c: (len(legs[c]) == 2, fused[c]),
            reverse=True,
        )[:k]
        # if not ranked_ids:                          # safety net: never return empty
        #     ranked_ids = sorted(fused, key=fused.get, reverse=True)[:k]

        id_to_doc = {d.metadata["child_id"]: d for d in self.documents}
        return [(id_to_doc[cid], fused[cid]) for cid in ranked_ids]


    # # ----------------------------------------------------- retrieve+expand
    # def retrieve_and_expand(self, query: str, k: int = None):
    #     query = self.preprocess_query(query)  
    #     k = k or settings.top_k
    #     if settings.use_hybrid:
    #         results = self.hybrid_search(query, k=k)
    #     else:
    #         results = self.vectorstore.similarity_search_with_score(query, k=k)
    #     expanded = []
    #     for doc, score in results:
    #         parent_id = doc.metadata.get("parent_id")
    #         parent_data = self.parents.get(parent_id, {})
    #         expanded.append({
    #             "score": score,
    #             "child_id": doc.metadata.get("child_id"),
    #             "parent_id": parent_id,
    #             "doc_name": doc.metadata.get("doc_name"),
    #             "section": doc.metadata.get("section"),
    #             "child_text": doc.metadata.get("display_text"),
    #             "parent_text": parent_data.get("text", ""),
    #             "images": doc.metadata.get("images", []),   # ← NEW
    #         })
    #     return expanded

    # ----------------------------------------------------- retrieve+expand
    def retrieve_and_expand(self, query: str, k: int = None):
        query = normalize_text(query)                      # hazm preprocessing
        k = k or settings.top_k
        fetch_k = getattr(settings, "dedup_fetch_k", 15)   # over-fetch so k unique parents survive

        if settings.use_hybrid:
            results = self.hybrid_search(query, k=fetch_k)
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=fetch_k)

        # ---- step 1: group children by parent_id → each parent enters context ONCE ----
        per_parent, order = {}, []
        for doc, score in results:
            pid = doc.metadata.get("parent_id")
            if pid not in per_parent:
                per_parent[pid] = {"doc": doc, "score": score, "images": []}
                order.append(pid)
            e = per_parent[pid]
            if score > e["score"]:                         # keep best-scoring child as representative
                e["doc"], e["score"] = doc, score
            for im in doc.metadata.get("images", []):      # union images of ALL matched children
                if im["path"] not in [x["path"] for x in e["images"]]:
                    e["images"].append(im)

        expanded = []
        for pid in order:
            e = per_parent[pid]
            doc = e["doc"]
            parent_data = self.parents.get(pid, {})
            expanded.append({
                "score": e["score"],
                "child_id": doc.metadata.get("child_id"),
                "parent_id": pid,
                "doc_name": doc.metadata.get("doc_name"),
                "section": doc.metadata.get("section"),
                "child_text": doc.metadata.get("display_text"),
                "parent_text": parent_data.get("text", ""),
                "images": e["images"],
                "doc_names": [doc.metadata.get("doc_name")],
            })

        expanded.sort(key=lambda r: r["score"], reverse=True)

        # ---- step 2: merge the same section copied across different docs ----
        expanded = self._dedup(expanded, getattr(settings, "dedup_threshold", 0.9))
        final = expanded[:k]
        self._log_query(query, final, 0.0)   # ← every app query now logged
        return final

    @staticmethod
    def _dedup(expanded, threshold: float = 0.9):
        """Merge near-identical chunks; record every document they appear in."""
        kept = []
        for res in expanded:
            for k_ in kept:
                if SequenceMatcher(None, res["child_text"], k_["child_text"]).ratio() >= threshold:
                    # same section seen in another doc → merge, don't add a duplicate
                    if res["doc_name"] not in k_["doc_names"]:
                        k_["doc_names"].append(res["doc_name"])
                    for im in res.get("images", []):
                        if im["path"] not in [x["path"] for x in k_["images"]]:
                            k_["images"].append(im)
                    break
            else:
                res["doc_names"] = [res["doc_name"]]
                kept.append(res)
        return kept
    
    # embedder.py — add this method to ParentChildRAG
    # ------------------------------------------------------- query prep
    @staticmethod
    def preprocess_query(query: str) -> str:
        """Hazm-normalize the query so it matches the normalized docs."""
        return normalize_text(query)

    @staticmethod
    def _log_query(query: str, results: List[Dict[str, Any]], elapsed: float):
        """Append one query session to the eval log file."""
        lines = [
            f"\n{'='*70}",
            f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🔍 Query: {query}",
            f"⏱️  Retrieval took {elapsed:.3f} seconds. Found {len(results)} results.",
            "",
        ]
        for i, res in enumerate(results, 1):
            child_snippet = res["child_text"][:400] + "..." if len(res["child_text"]) > 400 else res["child_text"]
            parent_snippet = res["parent_text"][:600] + "..." if len(res["parent_text"]) > 600 else res["parent_text"]
            lines += [
                f"--- 🏆 Result {i} (Score: {res['score']:.4f}) ---",
                f"📄 Document : {res['doc_name']}",
                f"📑 Section  : {res['section']}",
                f"🧩 Child ID : {res['child_id']}",
                "-" * 50,
                "👉 Matched Child Context:",
                child_snippet,
                "-" * 50,
                "📚 Expanded Parent Context:",
                parent_snippet,
                "=" * 70,
                "",
            ]
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------- interactive loop
    def interactive_loop(self, k: int = settings.top_k):
        print("\n" + "=" * 70)
        print("🤖 RAG System Ready! Ask your questions.")
        print(f"📝 All queries & results are logged to: {LOG_FILE.resolve()}")
        print("Type 'quit', 'exit', or 'q' to stop.")
        print("=" * 70 + "\n")

        while True:
            try:
                query = input("🔍 Your Query: ").strip()
                if not query:
                    continue
                if query.lower() in ["quit", "exit", "q"]:
                    print("👋 Exiting RAG system. Goodbye!")
                    break

                start_time = time.time()
                results = self.retrieve_and_expand(query, k=k)
                elapsed = time.time() - start_time

                # ✨ Log to file BEFORE printing to terminal
                self._log_query(query, results, elapsed)

                print(f"\n⏱️  Retrieval took {elapsed:.3f} seconds. Found {len(results)} results:\n")
                for i, res in enumerate(results, 1):
                    print(f"--- 🏆 Result {i} (Score: {res['score']:.4f}) ---")
                    print(f"📄 Document : {res['doc_name']}")
                    print(f"📑 Section  : {res['section']}")
                    print(f"🧩 Child ID : {res['child_id']}")
                    print("-" * 50)
                    print("👉 Matched Child Context:")
                    child_snippet = res["child_text"][:400] + "..." if len(res["child_text"]) > 400 else res["child_text"]
                    print(child_snippet)
                    print("-" * 50)
                    print("📚 Expanded Parent Context:")
                    parent_snippet = res["parent_text"][:600] + "..." if len(res["parent_text"]) > 600 else res["parent_text"]
                    print(parent_snippet)
                    print("\n" + "=" * 70 + "\n")

            except KeyboardInterrupt:
                print("\n👋 Exiting RAG system. Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    rag = ParentChildRAG()
    rag.load_data()
    rag.setup_embeddings()
    rag.build_or_load_index()
    rag.build_or_load_bm25()
    rag.interactive_loop()