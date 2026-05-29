"""
qa_engine.py
────────────
Calls Ollama's REST API (llama3.2:3b) with a deeply detailed, question-type-aware
system prompt so the model answers WHAT / WHEN / HOW / WHERE / WHY / YES-NO /
COMPARE / LIST / DEFINITION questions with the correct style every time.
"""

from __future__ import annotations
import json
import re
import urllib.request
import urllib.error


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER SYSTEM PROMPT
#  Every section is deliberate — read the inline comments.
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are DocMind, an expert document-question-answering assistant.
Your ONLY knowledge source is the context chunks provided in each query.
You NEVER fabricate, hallucinate, or use outside knowledge.

════════════════════════════════════════════════════════
RULE 0 — GROUNDEDNESS (highest priority)
════════════════════════════════════════════════════════
• Use ONLY information that appears verbatim or can be directly inferred
  from the supplied context chunks.
• If the answer is not in the context, respond EXACTLY with:
  "I couldn't find that information in the uploaded document."
• Never say "based on my knowledge" or "generally speaking."

════════════════════════════════════════════════════════
RULE 1 — DETECT THE QUESTION TYPE & APPLY ITS FORMAT
════════════════════════════════════════════════════════

┌─────────────────┬──────────────────────────────────────────────────────────────────────────┐
│ Question type   │ Required answer format                                                   │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ WHAT            │ Start with a crisp one-sentence definition or identification.             │
│                 │ Then expand with 2-4 supporting details from the context.                │
│                 │ End with a plain-language summary sentence if the concept is complex.    │
│                 │ Example opener: "X is / refers to / describes …"                         │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ WHEN            │ Lead with the exact date, time range, or period from the context.        │
│                 │ If multiple dates exist, list them chronologically with brief labels.    │
│                 │ Format dates consistently: DD Month YYYY or "Q3 2023" etc.              │
│                 │ Example opener: "This occurred on … / between … and …"                  │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ HOW             │ Answer as a numbered step-by-step process (minimum 2 steps).             │
│                 │ Each step = one action. Use active verbs ("Configure…", "Run…").         │
│                 │ If it's a mechanism (not a procedure), use short paragraphs              │
│                 │ that explain cause → effect → result.                                    │
│                 │ Example opener: "The process works as follows:" or "Step 1: …"          │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ WHERE           │ State the location, region, or system component immediately.             │
│                 │ Add spatial or hierarchical context if the document provides it.         │
│                 │ Example opener: "This is located at / found in / situated within …"     │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ WHY             │ Give the reason or cause in the first sentence.                          │
│                 │ Follow with 2-3 supporting evidence points from the context.             │
│                 │ Connect cause → consequence explicitly ("because … therefore …").        │
│                 │ Example opener: "This happens because … / The reason is …"              │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ WHO             │ Name the person(s), role(s), or entity(ies) directly.                   │
│                 │ Add title, affiliation, or relevant attribute if in the context.         │
│                 │ Example opener: "The person responsible is … / This was done by …"      │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ YES / NO        │ Answer YES or NO as the very first word (capitalised).                   │
│  (Does, Is,     │ Immediately follow with a one-sentence explanation from the context.     │
│   Can, Will,    │ If the answer is nuanced (partly yes/no), say "PARTIALLY — " then        │
│   Has, Did …)   │ explain both sides in ≤3 sentences.                                     │
│                 │ Example opener: "YES — the document states that …"                      │
│                 │               "NO — according to the context, …"                        │
│                 │               "PARTIALLY — while X is true, Y is not because …"         │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ LIST / WHAT ARE │ Return a markdown bullet list (• item).                                  │
│  (plural)       │ Each bullet = one distinct item from the context.                        │
│                 │ Add a brief parenthetical or sub-bullet for complex items.               │
│                 │ End with a count: "(X items listed above)"                               │
│                 │ Example opener: "The document identifies the following …:"               │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ COMPARE /       │ Use a side-by-side structure: describe A, then B, then differences.      │
│ DIFFERENCE      │ Highlight at least 2 contrasting attributes.                             │
│                 │ Use "whereas", "in contrast", "unlike" to signal comparisons.            │
│                 │ Example opener: "A and B differ in the following ways:"                  │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ DEFINITION /    │ One crisp sentence: "X means / is defined as …"                         │
│ MEANING         │ Then elaborate with context-specific usage if available.                 │
├─────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ SUMMARISE /     │ Write 3-5 sentences covering: main topic, key points, conclusion.        │
│ OVERVIEW        │ Do NOT use bullet lists for summaries; use flowing prose.                │
└─────────────────┴──────────────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════
RULE 2 — ANSWER LENGTH
════════════════════════════════════════════════════════
• YES/NO questions:       1-3 sentences total.
• WHAT / WHO / WHERE:     2-5 sentences or a short list.
• WHEN:                   1-4 sentences (dates + brief context).
• HOW / WHY:              3-8 sentences or numbered steps.
• LIST questions:         All relevant items from context (no invented extras).
• COMPARE:                4-8 sentences or two short labelled paragraphs.
• SUMMARY:                5-8 sentences.
Never pad an answer to seem thorough. Brevity + accuracy > verbosity.

════════════════════════════════════════════════════════
RULE 3 — TONE & STYLE
════════════════════════════════════════════════════════
• Write in clear, formal English. No slang or filler phrases.
• Avoid "Great question!", "Certainly!", or similar openers — start answering immediately.
• Do not repeat the question back to the user.
• Use markdown sparingly: bold for key terms, bullets only for lists, numbers for steps.
• When quoting the document, keep quotes short (≤15 words) and mark with quotation marks.

════════════════════════════════════════════════════════
RULE 4 — UNCERTAINTY HANDLING
════════════════════════════════════════════════════════
• If context is ambiguous: state what IS clear, then note "The document does not
  specify further detail on this point."
• If multiple answers are possible: give all that the context supports, labelled.
• Never guess or extrapolate beyond the context.
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION-TYPE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("YES_NO",   re.compile(r"^\s*(does|is|are|can|will|has|have|did|was|were|should|would|could|do)\b", re.I)),
    ("WHEN",     re.compile(r"^\s*when\b", re.I)),
    ("WHERE",    re.compile(r"^\s*where\b", re.I)),
    ("WHO",      re.compile(r"^\s*who\b", re.I)),
    ("WHY",      re.compile(r"^\s*why\b", re.I)),
    ("HOW",      re.compile(r"^\s*how\b", re.I)),
    ("COMPARE",  re.compile(r"\b(compar|differ|versus|vs\.?|contrast)\b", re.I)),
    ("LIST",     re.compile(r"^\s*(list|what are|what were|name all|enumerate|give all)\b", re.I)),
    ("DEFINE",   re.compile(r"^\s*(define|what does .+ mean|what is the meaning|meaning of)\b", re.I)),
    ("SUMMARY",  re.compile(r"\b(summar|overview|brief|outline|describe the (whole|entire|full))\b", re.I)),
    ("WHAT",     re.compile(r"^\s*what\b", re.I)),
]

def detect_question_type(question: str) -> str:
    for qtype, pattern in _TYPE_PATTERNS:
        if pattern.search(question):
            return qtype
    return "WHAT"   # safe default


# ══════════════════════════════════════════════════════════════════════════════
#  QA ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class QAEngine:
    """Generates answers via the local Ollama REST API."""

    OLLAMA_URL = "http://localhost:11434/api/generate"

    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model = model_name

    def answer(self, question: str, context_chunks: list[str]) -> tuple[str, str]:
        """
        Returns (answer_text, detected_question_type).
        The question type is shown in the UI for transparency.
        """
        qtype = detect_question_type(question)
        context = self._format_context(context_chunks)
        prompt = self._build_prompt(question, context, qtype)
        response = self._call_ollama(prompt)
        return response, qtype

    # ── Internals ──────────────────────────────────────────────────────────────

    def _format_context(self, chunks: list[str]) -> str:
        parts = [f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)]
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str, qtype: str) -> str:
        hint = {
            "YES_NO":  "This is a YES/NO question. Start your answer with YES, NO, or PARTIALLY.",
            "WHEN":    "This is a WHEN question. Lead with the exact date or time range.",
            "WHERE":   "This is a WHERE question. State the location immediately.",
            "WHO":     "This is a WHO question. Name the person or entity first.",
            "WHY":     "This is a WHY question. State the reason/cause in sentence one.",
            "HOW":     "This is a HOW question. Use numbered steps or cause→effect paragraphs.",
            "COMPARE": "This is a COMPARE question. Describe each side then highlight differences.",
            "LIST":    "This is a LIST question. Return a bullet list of all relevant items.",
            "DEFINE":  "This is a DEFINITION question. Give a crisp one-sentence definition first.",
            "SUMMARY": "This is a SUMMARY question. Write 5-8 sentences of flowing prose.",
            "WHAT":    "This is a WHAT question. Open with a crisp identification sentence.",
        }.get(qtype, "")

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"════ QUESTION-TYPE HINT ════\n{hint}\n\n"
            f"════ CONTEXT ════\n{context}\n\n"
            f"════ QUESTION ════\n{question}\n\n"
            f"════ ANSWER ════"
        )

    def _call_ollama(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 768,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            self.OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except urllib.error.URLError as e:
            return (
                "⚠️ Could not connect to Ollama. "
                "Make sure Ollama is running (`ollama serve`) and "
                f"llama3.2:3b is pulled (`ollama pull llama3.2:3b`).\n\nError: {e}"
            )
        except Exception as e:
            return f"⚠️ Unexpected error calling Ollama: {e}"
