"""
Conflict resolver for RAG.

Contradiction detection uses two signals:
  1. Sentiment polarity flip between chunks sharing a person/topic
  2. Negation/conflict keyword presence in at least one chunk

Why not an NLI model?
  DeBERTa-mnli costs ~180MB. For personal diary text with named entities,
  sentiment polarity flip is a reliable enough proxy. This is an explicit
  engineering tradeoff, documented in the README.
"""

import re
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

NEGATION_PATTERNS = re.compile(
    r"\b(not|never|no longer|didn't|don't|doesn't|can't|won't|wasn't|isn't|"
    r"argument|conflict|fight|distant|snapped|angry|upset|struggling)\b",
    re.IGNORECASE
)


def _sentiment_polarity(text: str) -> str:
    compound = _vader.polarity_scores(text)["compound"]
    if compound >= 0.15:
        return "positive"
    elif compound <= -0.15:
        return "negative"
    return "neutral"


def _topics_overlap(t1: list, t2: list) -> float:
    s1, s2 = set(t1), set(t2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def detect_contradictions(chunks: list) -> list:
    """
    Returns pairs of chunks that appear contradictory.
    Requires: opposing polarity + shared entity/topic + negation signal.
    """
    contradictions = []
    polarities = {c["id"]: _sentiment_polarity(c["text"]) for c in chunks}

    for i, c1 in enumerate(chunks):
        for c2 in chunks[i + 1:]:
            p1 = polarities[c1["id"]]
            p2 = polarities[c2["id"]]

            if p1 == p2 or p1 == "neutral" or p2 == "neutral":
                continue

            topic_sim = _topics_overlap(c1.get("topics", []), c2.get("topics", []))
            shared_people = set(c1.get("people", [])) & set(c2.get("people", []))

            if topic_sim < 0.15 and not shared_people:
                continue

            has_conflict = (
                NEGATION_PATTERNS.search(c1["text"]) or
                NEGATION_PATTERNS.search(c2["text"])
            )
            if not has_conflict:
                continue

            contradictions.append({
                "chunk_a": {"id": c1["id"], "day": c1["day"], "text": c1["text"], "polarity": p1},
                "chunk_b": {"id": c2["id"], "day": c2["day"], "text": c2["text"], "polarity": p2},
                "shared_topics": list(set(c1.get("topics", [])) & set(c2.get("topics", []))),
                "shared_people": list(shared_people),
                "explanation": (
                    f"Day {c1['day']} ({p1}) and Day {c2['day']} ({p2}) "
                    f"give conflicting context about "
                    f"{', '.join(shared_people) if shared_people else 'overlapping topics'}"
                ),
            })

    return contradictions


def resolve_conflicts(query: str, chunks: list, contradictions: list) -> dict:
    """
    Merge retrieved chunks into a coherent response.

    Strategy: sort chronologically, tag contradicted chunks as superseded,
    prefer most recent as current state, surface full history in narrative.
    No LLM used — template-based merge. Offline constraint respected.
    """
    query_lower = query.lower()
    relevant = []
    for c in chunks:
        text_lower = c["text"].lower()
        people_match = any(
            p.lower() in query_lower or p.lower() in text_lower
            for p in c.get("people", [])
        )
        topic_match = any(t in query_lower for t in c.get("topics", []))
        keyword_match = any(w in text_lower for w in query_lower.split() if len(w) > 3)

        if people_match or topic_match or keyword_match:
            relevant.append(c)

    if not relevant:
        relevant = chunks[:3]

    relevant_sorted = sorted(relevant, key=lambda x: (x["day"], x.get("timestamp", "")))

    contradicted_ids = set()
    for contradiction in contradictions:
        a, b = contradiction["chunk_a"], contradiction["chunk_b"]
        older = a if a["day"] <= b["day"] else b
        contradicted_ids.add(older["id"])

    tagged_chunks = []
    for c in relevant_sorted:
        polarity = _sentiment_polarity(c["text"])
        tag = "contradicted" if c["id"] in contradicted_ids else "current"
        tagged_chunks.append({
            "id": c["id"],
            "day": c["day"],
            "date": c.get("date", ""),
            "text": c["text"],
            "polarity": polarity,
            "status": tag,
            "scores": {
                "final":   c.get("score_final", 0),
                "recency": c.get("score_recency", 0),
                "emotion": c.get("score_emotion", 0),
            }
        })

    current_chunks = [c for c in tagged_chunks if c["status"] == "current"]
    superseded_chunks = [c for c in tagged_chunks if c["status"] == "contradicted"]

    narrative_parts = []
    if superseded_chunks:
        day_refs = ", ".join(f"Day {c['day']}" for c in superseded_chunks)
        earlier_texts = " / ".join(c["text"][:100] for c in superseded_chunks)
        narrative_parts.append(f"Earlier ({day_refs}): {earlier_texts}")

    if current_chunks:
        most_recent = current_chunks[-1]
        narrative_parts.append(
            f"Most recently (Day {most_recent['day']}): {most_recent['text'][:150]}"
        )

    has_contradiction = len(contradictions) > 0
    contradiction_note = contradictions[0]["explanation"] if has_contradiction else ""

    return {
        "query": query,
        "has_contradiction": has_contradiction,
        "contradiction_note": contradiction_note,
        "merged_narrative": " | ".join(narrative_parts),
        "chunks": tagged_chunks,
        "contradiction_pairs": [
            {
                "day_a": c["chunk_a"]["day"],
                "text_a": c["chunk_a"]["text"][:100],
                "polarity_a": c["chunk_a"]["polarity"],
                "day_b": c["chunk_b"]["day"],
                "text_b": c["chunk_b"]["text"][:100],
                "polarity_b": c["chunk_b"]["polarity"],
                "explanation": c["explanation"],
            }
            for c in contradictions
        ],
        "source_trace": [
            {"id": c["id"], "day": c["day"], "status": c["status"]}
            for c in tagged_chunks
        ],
    }


if __name__ == "__main__":
    import json
    from rag.memory_store import MemoryStore
    from rag.retrieval_engine import load_persona_into_store, query as retrieve

    store = MemoryStore()
    load_persona_into_store(store)

    q = "Did I mention anything about my sister?"
    chunks = retrieve(store, q)
    contradictions = detect_contradictions(chunks)
    result = resolve_conflicts(q, chunks, contradictions)

    print("\n── Conflict Resolver Output ──\n")
    print(f"Contradiction detected: {result['has_contradiction']}")
    print(f"Note: {result['contradiction_note']}")
    print(f"\nMerged narrative:\n  {result['merged_narrative'][:300]}")
    print(f"\nSource trace:")
    for s in result["source_trace"]:
        print(f"  Day {s['day']} [{s['status']}] id={s['id']}")