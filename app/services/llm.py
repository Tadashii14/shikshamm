from __future__ import annotations

import os
from typing import List

from openai import OpenAI


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    # Use env var; set timeouts per-request via with_options()
    return OpenAI()


def _clean_json_like(content: str) -> str:
    # Strip common code fences ```json ... ``` or ``` ... ```
    text = content.strip()
    if text.startswith("```"):
        # Remove first fence line
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        # Remove trailing fence if present
        if text.endswith("```"):
            text = text[:-3]
    # Try to extract first JSON array substring if present
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

def _local_flashcards_from_text(note_text: str, max_cards: int) -> List[dict]:
    import re
    text = (note_text or "").strip()
    if not text:
        return []
    # Split by lines/sentences, pick distinct informative chunks
    parts = [p.strip() for p in re.split(r"[\n\r\.]+", text) if len(p.strip()) > 5]
    seen = set()
    cards: List[dict] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        if len(cards) >= max_cards:
            break
        if ":" in p:
            term, desc = p.split(":", 1)
            front = term.strip()
            back = desc.strip()
        else:
            front = p.split(" ")[:8]
            front = " ".join(front) + "?"
            back = p
        cards.append({"front": front, "back": back})
    if not cards:
        cards = [{"front": "Summary", "back": text[:500]}]
    return cards[:max_cards]


def generate_flashcards_from_text(note_text: str, max_cards: int = 12) -> List[dict]:
    try:
        client = _get_client().with_options(timeout=30.0)
        prompt = (
            "Create concise Q/A flashcards from the following study notes. "
            "Return JSON array with objects {front, back}. Use at most "
            f"{max_cards} high-value cards. Notes:\n\n{note_text[:6000]}"
        )
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = rsp.choices[0].message.content or "[]"
        content = _clean_json_like(content)
        # Be permissive with JSON parsing
        import json
        data = json.loads(content)
        items = []
        for item in data:
            front = str(item.get("front", "")).strip()
            back = str(item.get("back", "")).strip()
            if front and back:
                items.append({"front": front, "back": back})
        if items:
            return items[:max_cards]
        # If parsing produced empty, fallback locally
        return _local_flashcards_from_text(note_text, max_cards)
    except Exception:
        return _local_flashcards_from_text(note_text, max_cards)



def _local_quiz_from_text(note_text: str, num_questions: int) -> list[dict]:
    import re
    text = (note_text or "").strip()
    if not text:
        return []
    lines = [l.strip() for l in re.split(r"[\n\r]+", text) if len(l.strip()) > 5]
    questions: list[dict] = []
    for l in lines:
        if len(questions) >= num_questions:
            break
        words = l.split()
        if len(words) < 4:
            continue
        stem = " ".join(words[:10])
        correct = l
        options = [correct, f"Not {words[0]}", f"Unrelated {words[1] if len(words)>1 else 'option'}", f"None of the above"]
        questions.append({"question": f"Complete: {stem} ...", "options": options, "answer": correct})
    if not questions:
        questions = [{"question": "No content available for quiz.", "options": ["OK", "-"], "answer": "OK"}]
    return questions[:num_questions]


def generate_quiz_from_text(note_text: str, num_questions: int = 8) -> list[dict]:
    try:
        client = _get_client().with_options(timeout=30.0)
        prompt = (
            "Create a multiple-choice quiz from the following study notes. "
            "Return JSON array with objects {question, options, answer}. "
            f"Generate {num_questions} questions. Notes:\n\n{note_text[:6000]}"
        )
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = rsp.choices[0].message.content or "[]"
        content = _clean_json_like(content)
        import json
        data = json.loads(content)
        items = []
        for item in data:
            q = str(item.get("question", "")).strip()
            opts = item.get("options", [])
            ans = str(item.get("answer", "")).strip()
            if q and isinstance(opts, list) and len(opts) >= 2 and ans:
                items.append({"question": q, "options": opts[:6], "answer": ans})
        if items:
            return items[:num_questions]
        return _local_quiz_from_text(note_text, num_questions)
    except Exception:
        return _local_quiz_from_text(note_text, num_questions)
