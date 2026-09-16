# app.py
from pathlib import Path
import streamlit as st

from auth import AuthManager
from keyvault import KeyVault
from chatstore import ChatStore
from config import settings
from embedder import ParentChildRAG
from generator import RAGGenerator
import time

st.set_page_config(page_title="Persian Medical RAG", layout="wide")

# app.py — top of the file, after set_page_config
# RTL_CSS = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap');

# /* ---- Font change ONLY (scoped so Streamlit icons/avatars stay intact) ---- */
# html, body,
# .stMarkdown,
# [data-testid="stMarkdownContainer"],
# [data-testid="stMarkdownContainer"] p,
# [data-testid="stMarkdownContainer"] li,
# [data-testid="stMarkdownContainer"] h1,
# [data-testid="stMarkdownContainer"] h2,
# [data-testid="stMarkdownContainer"] h3,
# [data-testid="stMarkdownContainer"] h4,
# [data-testid="stSidebar"] p,
# [data-testid="stSidebar"] button p,
# [data-testid="stTextInput"] input,
# [data-testid="stChatInputTextArea"] {
#     font-family: "Vazirmatn", "Segoe UI", Tahoma, sans-serif !important;
# }

# /* Persian / RTL rendering inside chat bubbles */
# [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
#     direction: rtl;
#     text-align: right;
# }
# /* Keep code blocks and the thinking expander (usually English) LTR */
# [data-testid="stChatMessage"] pre,
# [data-testid="stChatMessage"] code,
# [data-testid="stChatMessage"] [data-testid="stExpander"] {
#     direction: ltr;
#     text-align: left;
# }
# </style>
# """
RTL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap');

/* Font */
html, body,
.stMarkdown,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] button p,
[data-testid="stTextInput"] input,
[data-testid="stChatInputTextArea"] {
    font-family: "Vazirmatn", "Segoe UI", Tahoma, sans-serif !important;
}

/* =========================
   Persian chat messages
   ========================= */

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    direction: rtl;
    text-align: right;
    width: 100%;
}

/* Paragraphs */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    direction: rtl;
    text-align: right;
    margin: 0.35rem 0;
    line-height: 1.9;
}

/* Lists */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
    direction: rtl;
    text-align: right;

    /* This is the important part */
    padding-right: 1.8rem;
    padding-left: 0;

    margin-top: 0.5rem;
    margin-bottom: 0.7rem;
}

/* List items */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    direction: rtl;
    text-align: right;

    padding-right: 0.2rem;
    padding-left: 0;

    margin-bottom: 0.35rem;
    line-height: 1.9;
}

/* Nested lists */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li > ul,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li > ol {
    margin-top: 0.25rem;
    margin-bottom: 0.25rem;
    padding-right: 1.5rem;
}

/* Headings */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4 {
    direction: rtl;
    text-align: right;
    margin-top: 0.8rem;
    margin-bottom: 0.5rem;
    line-height: 1.7;
}

/* Tables */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
    direction: rtl;
    text-align: right;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
    text-align: right;
}

/* =========================
   Keep code LTR
   ========================= */

[data-testid="stChatMessage"] pre,
[data-testid="stChatMessage"] code {
    direction: ltr;
    text-align: left;
    font-family: Consolas, "Courier New", monospace !important;
}

/* =========================
   Chat input
   ========================= */

[data-testid="stChatInputTextArea"] {
    direction: rtl !important;
    text-align: right !important;
}

</style>
"""

st.markdown(RTL_CSS, unsafe_allow_html=True)

# st.markdown(RTL_CSS, unsafe_allow_html=True)
# st.markdown(RTL_CSS, unsafe_allow_html=True)
# st.markdown(RTL_CSS, unsafe_allow_html=True)
# st.markdown(RTL_CSS, unsafe_allow_html=True)
# st.markdown(RTL_CSS, unsafe_allow_html=True)

for key, val in {"authenticated": False, "user": None,
                 "active_conv": None, "messages": []}.items():
    if key not in st.session_state:
        st.session_state[key] = val

auth  = AuthManager()
store = ChatStore()
vault = KeyVault()

# app.py — replace the existing resolve_image function with:
PROJECT_ROOT = Path(__file__).parent.resolve()

def resolve_image(path: str) -> Path:
    """Resolve image path to an absolute path that Streamlit can open."""
    p = Path(path)
    
    # Already absolute and exists
    if p.is_absolute() and p.exists():
        return p
    
    # Try relative to project root first (most common case)
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    
    # Fallback: try docs/ and assets/ subdirectories
    for base in ("docs", "assets"):
        fallback = PROJECT_ROOT / base / path
        if fallback.exists():
            return fallback
    
    # Return absolute path even if not found (Streamlit will show a clear error)
    return PROJECT_ROOT / path

# def resolve_image(path: str) -> Path:
#     p = Path(path)
#     if p.exists():
#         return p
#     for base in (Path("."), Path("docs"), Path("assets")):
#         q = base / path
#         if q.exists():
#             return q
#     return p

# # ══════════════════ LOGIN PAGE (before loading anything heavy) ══════════════════
# if not st.session_state.authenticated:
#     st.title("🔐 Login")
#     tab_login, tab_reg = st.tabs(["Login", "Create account"])
#     with tab_login:
#         with st.form("login_form"):
#             u = st.text_input("Username")
#             p = st.text_input("Password", type="password")
#             if st.form_submit_button("Sign in"):
#                 profile = auth.verify(u, p)
#                 if profile:
#                     st.session_state.authenticated = True
#                     st.session_state.user = profile
#                     st.rerun()
#                 else:
#                     st.error("Wrong username or password.")
#     with tab_reg:   # optional self-registration; prefer create_user.py for key control
#         with st.form("reg_form"):
#             nu = st.text_input("Username")
#             npw = st.text_input("Password", type="password")
#             nk  = st.text_input("Your OpenRouter API key", type="password")
#             nm  = st.text_input("Default model (OpenRouter ID)", value="meta-llama/llama-3.1-8b-instruct")
#             if st.form_submit_button("Register"):
#                 try:
#                     auth.add_user(nu, npw, nk, nm)
#                     st.success("Account created — please log in.")
#                 except ValueError as e:
#                     st.error(str(e))
#     st.stop()

# username = st.session_state.user["username"]

# ══════════════════ LOGIN PAGE ══════════════════

if not st.session_state.authenticated:
    st.title("🔐 Login")

    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.form_submit_button("Sign in"):
            profile = auth.verify(u, p)

            if profile:
                st.session_state.authenticated = True
                st.session_state.user = profile
                st.rerun()
            else:
                st.error("Wrong username or password.")

    st.stop()


username = st.session_state.user["username"]
key_alias = st.session_state.user.get("key_alias", username)

openrouter_api_key = vault.get_key(key_alias)

if not openrouter_api_key:
    st.error("No OpenRouter API key is assigned to this user. Contact the administrator.")
    st.stop()

# ══════════════════ Shared heavy RAG pipeline (cached once) ══════════════════
@st.cache_resource(show_spinner="Loading RAG pipeline & indexes…")
def load_rag():
    rag = ParentChildRAG()
    rag.load_data()
    rag.setup_embeddings()
    rag.build_or_load_index()
    rag.build_or_load_bm25()
    return rag

# ══════════════════ SIDEBAR: model, memory, per-user history ══════════════════
with st.sidebar:
    st.markdown(f"### 👤 {username}")

    # model = st.text_input("OpenRouter model ID",
    #                       value=st.session_state.user["default_model"],
    #                       key="model_id")
    # use_memory   = st.toggle("🧠 Enable chat memory", value=True)
    # memory_turns = st.slider("Memory window (turns)", 1, 20, 3,
    #                          disabled=not use_memory)
    st.caption(f"🤖 Model: `{settings.openrouter_model}`")
    st.caption(f"🧠 Memory: {'on' if settings.use_chat_memory else 'off'} "
               f"(last {settings.memory_messages} messages)")
    st.divider()

    if st.button("＋ New chat", use_container_width=True):
        st.session_state.active_conv = store.new_conversation(username)
        st.session_state.messages = []
        st.rerun()

    convs = store.load_all(username)
    st.caption("Your conversations")
    for cid, conv in sorted(convs.items(),
                            key=lambda kv: kv[1]["updated"], reverse=True):
        c1, c2 = st.columns([6, 1])
        label = ("● " if cid == st.session_state.active_conv else "") + conv["title"]
        if c1.button(label, key=f"open_{cid}", use_container_width=True):
            st.session_state.active_conv = cid
            st.session_state.messages = conv["messages"]
            st.rerun()
        if c2.button("🗑", key=f"del_{cid}"):
            store.delete_conversation(username, cid)
            if st.session_state.active_conv == cid:
                st.session_state.active_conv = None
                st.session_state.messages = []
            st.rerun()

    st.divider()
    if st.button("Logout", use_container_width=True):
        for k in ("authenticated", "user", "active_conv", "messages"):
            st.session_state[k] = False if k == "authenticated" else (
                None if k in ("user", "active_conv") else [])
        st.rerun()

# Ensure an active conversation exists for this user
if st.session_state.active_conv is None:
    convs = store.load_all(username)
    if convs:
        cid = max(convs, key=lambda c: convs[c]["updated"])
        st.session_state.active_conv = cid
        st.session_state.messages = convs[cid]["messages"]
    else:
        st.session_state.active_conv = store.new_conversation(username)
        st.session_state.messages = []


# ══════════════════ MAIN CHAT ══════════════════
st.title("🏥 Persian Surgical RAG Assistant")

rag = load_rag()
# generator = RAGGenerator(api_key=st.session_state.user["openrouter_api_key"],
#                          model=settings.openrouter_model)

if st.session_state.get("generator_user") != username:
    st.session_state.generator = RAGGenerator(
        api_key=openrouter_api_key,
        model=settings.openrouter_model
    )
    st.session_state.generator_user = username

generator = st.session_state.generator

# --- 1. Render existing history FIRST ---
def render_message(m: dict):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for im in m.get("images", []):
            st.image(resolve_image(im["path"]), caption=im.get("alt") or im["path"])

for m in st.session_state.messages:
    render_message(m)

# --- 2. THEN handle new input ---
if prompt := st.chat_input("Ask your question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    store.append_message(username, st.session_state.active_conv,
                         {"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.time()
        with st.spinner("🔍 Retrieving context…"):
            results = rag.retrieve_and_expand(prompt)
        t_retrieval = time.time() - t0

        # Collect images now, but DON'T render them yet
        seen, imgs = set(), []
        for r in results[:1]:                      # ← was: for r in results:
            for im in r.get("images", []):
                if im["path"] not in seen:
                    seen.add(im["path"])
                    imgs.append(im)

        history = st.session_state.messages[:-1][-settings.memory_messages:] \
                  if settings.use_chat_memory else []
        response = st.write_stream(
            generator.stream_answer(prompt, results, history=history))

        # 🖼️ Images appear only AFTER generation finishes
        if imgs:
            st.divider()
            st.markdown("**🖼️ From the retrieved context:**")
            for im in imgs:
                st.image(resolve_image(im["path"]), caption=im.get("alt") or im["path"])

        st.caption(f"⏱️ Retrieval: {t_retrieval:.1f}s · Total: {time.time() - t0:.1f}s")

    assistant_msg = {"role": "assistant", "content": response, "images": imgs}
    st.session_state.messages.append(assistant_msg)
    store.append_message(username, st.session_state.active_conv, assistant_msg)

# if prompt := st.chat_input("Ask your question…"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     store.append_message(username, st.session_state.active_conv,
#                          {"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("🔍 Retrieving context (hazm-normalized query)…"):
#             results = rag.retrieve_and_expand(prompt)   # query preprocessed inside

#         sources = [{"doc_name": r["doc_name"], "section": r["section"]} for r in results]
#         if sources:
#             with st.expander("📚 Sources"):
#                 for s in sources:
#                     st.markdown(f"- *{s['doc_name']}* — {s['section']}")

#         # Images attached to the retrieved context
#         seen, imgs = set(), []
#         for r in results:
#             for im in r.get("images", []):
#                 if im["path"] not in seen:
#                     seen.add(im["path"])
#                     imgs.append(im)
#         if imgs:
#             st.markdown("**🖼️ From the retrieved context:**")
#             for im in imgs:
#                 st.image(resolve_image(im["path"]), caption=im.get("alt") or im["path"])

#         history = st.session_state.messages[:-1] if use_memory else []
#         response = st.write_stream(
#             generator.stream_answer(prompt, results,
#                                     history=history, max_turns=memory_turns))

#     assistant_msg = {"role": "assistant", "content": response,
#                      "sources": sources, "images": imgs}
#     st.session_state.messages.append(assistant_msg)
#     store.append_message(username, st.session_state.active_conv, assistant_msg)