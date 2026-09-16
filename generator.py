# generator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import datetime


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PROMPT_TEMPLATE = """You are an expert surgical and medical assistant, specializing in operating room protocols, surgical instruments, and perioperative care.

INSTRUCTIONS:
1. STRICT GROUNDING: Answer based STRICTLY and ONLY on the provided context. Do not use outside medical knowledge to fill in gaps. If the context does not contain the answer, reply exactly with: "اطلاعات کافی در متن ارائه‌شده برای پاسخ به این سوال وجود ندارد."
2. TERMINOLOGY: Answer in Persian. However, preserve exact English medical terms, instrument names, and device modes (e.g., SWIFT COAG, LigaSure, Pure Cut Mode, ORIF) in English, or include the English term in parentheses next to the Persian translation.
4. COMPLETENESS: When the context describes a multi-stage procedure, cover EVERY stage in order. NEVER compress or skip stages with summary sentences.
5. CUSTOM BEHAVIOR FOR TOOLS (ابزار): If the user asks about surgical tools or instruments (ابزار), BE SURE to structure your answer to include the sequential steps of the surgery (مراحل عمل), the specific tools used in each step, and relevant safety tips (نکات ایمنی).

{history_block}

Context:
{context}

Question: {question}

Answer:"""

class RAGGenerator:
    def __init__(self, api_key: str, model: str, temperature: float = 0.2):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,                      # ← per-user key
            base_url=OPENROUTER_BASE_URL,
            streaming=True,
            temperature=temperature,
            max_tokens=4096,
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    # @staticmethod
    # def format_context(retrieved_docs) -> str:
    #     if not retrieved_docs:
    #         return "No relevant context found in the documents."
    #     parts = [f"[Source {i}: {d['doc_name']} - {d['section']}]\n{d['parent_text']}"
    #              for i, d in enumerate(retrieved_docs, 1)]
    #     return "\n\n---\n\n".join(parts)

    @staticmethod
    def format_context(docs) -> str:
        if not docs:
            return "No relevant context found in the documents."
        parts = []
        for i, d in enumerate(docs, 1):
            names = d.get("doc_names", [d["doc_name"]])
            if len(names) > 1:
                shown = "، ".join(names[:5]) + ("…" if len(names) > 5 else "")
                header = (f"[Source {i}: SHARED/GENERAL note — appears in {len(names)} documents "
                          f"({shown}) | section: {d['section']}]")
            else:
                header = f"[Source {i}: {names[0]} - {d['section']}]"
            parts.append(f"{header}\n{d['parent_text']}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def format_history(messages) -> str:
        if not messages:
            return ""
        lines = [("User" if m["role"] == "user" else "Assistant") + ": " + m["content"]
                 for m in messages]
        return ("Previous conversation (use it to resolve pronouns and follow-ups):\n"
                + "\n".join(lines))

    def stream_answer(self, query, retrieved_docs, history=None):  # ← removed max_turns
        context = self.format_context(retrieved_docs)
        history_block = self.format_history(history or []) \
                        or "(Start of conversation - no previous turns.)"

        # ---------------------------------------------------------
        # 🕵️‍♂️ DEBUG: Construct the full prompt to inspect it
        # ---------------------------------------------------------
        full_prompt = self.prompt.format(
            context=context,
            question=query,
            history_block=history_block
        )
        
        # 1. Print to Console (for immediate checking)
        print("\n" + "="*80)
        print("📤 FULL PROMPT SENT TO LLM:")
        print("="*80)
        print(full_prompt)
        print("="*80 + "\n")

        # 2. Append to rag_eval_log.txt
        log_file = Path("rag_eval_log.txt")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'#'*80}\n")
            f.write(f"# 🤖 LLM INPUT LOG - {timestamp}\n")
            f.write(f"{'#'*80}\n")
            f.write(full_prompt)
            f.write(f"\n{'#'*80}\n")
        
        return self.chain.stream({"context": context, "question": query,
                                  "history_block": history_block})

