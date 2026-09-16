# verify_bm25.py
import json
from pathlib import Path
from langchain_core.documents import Document
from bm25builder import PersianBM25
from config import settings
from normalizer import normalize_text

docs = [Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in json.loads(Path(settings.children_path).read_text(encoding="utf-8"))]

loaded  = PersianBM25(use_stopwords=True); loaded.load(settings.bm25_index_path, docs)
old_way = PersianBM25(use_stopwords=True).build(docs, text_key="display_text")
new_way = PersianBM25(use_stopwords=True).build(docs, text_key="page_content")

q = normalize_text("نکات ایمنی قبل از عمل آپاندکتومی چیست؟")
tops = lambda b: [(d.metadata["doc_name"][:25], round(s, 2)) for d, s in b.search(q, k=5)]
t_loaded, t_old, t_new = tops(loaded), tops(old_way), tops(new_way)

print("Loaded pickle:", t_loaded)
print("display_text :", t_old)
print("page_content :", t_new)
if t_loaded == t_new:
    print("✅ Pickle IS the page_content build — BM25 now sees doc names + section titles.")
elif t_loaded == t_old:
    print("❌ Pickle is STILL the old display_text build — delete bm25_index.pkl and restart!")
else:
    print("⚠️ Matches neither — delete bm25_index.pkl and restart to be safe.")

print("\n--- The 'effect' in plain sight: BM25 top-3 per query ---")
for query in ["نکات ایمنی قبل از عمل آپاندکتومی چیست؟",
              "ابزار عمل هرنی اینگویینال",
              "مد SWIFT COAG چه کاربردی دارد؟"]:
    qn = normalize_text(query)
    print("\n🔍", query)
    for d, s in new_way.search(qn, k=3):
        print(f"   [{s:6.2f}] {d.metadata['doc_name']} | {d.metadata['section']}")