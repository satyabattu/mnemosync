"""
Retrieval engine.

Ranking formula:
    score = 0.5 * semantic_similarity
          + 0.3 * recency_weight  (exponential decay, half-life=3 days)
          + 0.2 * emotional_salience  (|VADER compound|)

Recency half-life of 3 days: Day 7 scores ~4x higher than Day 1 on
recency alone. Linear decay would be simpler but doesn't model how
quickly older context becomes irrelevant.
"""

import json
import math
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rag.memory_store import MemoryStore
from drift.embedder import get_embedder

DATA_PATH = Path(__file__).parent.parent / "data" / "persona_fixture.json"

_vader = SentimentIntensityAnalyzer()

RECENCY_HALF_LIFE_DAYS = 3.0
SEMANTIC_WEIGHT = 0.50
RECENCY_WEIGHT  = 0.30
EMOTION_WEIGHT  = 0.20


def _recency_score(day: int, max_day: int) -> float:
    age = max_day - day
    return math.exp(-age * math.log(2) / RECENCY_HALF_LIFE_DAYS)


def _emotion_score(sentiment_compound: float) -> float:
    return abs(sentiment_compound)


def load_persona_into_store(store: MemoryStore, data_path: Path = DATA_PATH) -> int:
    with open(data_path) as f:
        data = json.load(f)

    embedder = get_embedder()
    count = 0

    for session in data["sessions"]:
        texts = [m["text"] for m in session["messages"]]
        embeddings = embedder.embed(texts)

        for msg, emb in zip(session["messages"], embeddings):
            sentiment = _vader.polarity_scores(msg["text"])["compound"]
            store.upsert_message(
                msg_id    = msg["id"],
                day       = session["day"],
                date      = session["date"],
                timestamp = msg["timestamp"],
                text      = msg["text"],
                topics    = msg.get("topics", []),
                people    = msg.get("people_mentioned", []),
                sentiment = sentiment,
                embedding = emb,
            )
            count += 1

    print(f"[retrieval] Loaded {count} messages into memory store")
    return count


def query(store: MemoryStore, query_text: str, top_k: int = 8) -> list:
    embedder = get_embedder()
    q_emb = embedder.embed_mean([query_text])

    all_msgs = store.get_all()
    if not all_msgs:
        return []
    max_day = max(m["day"] for m in all_msgs)

    semantic_results = store.search_semantic(q_emb, top_k=min(top_k * 2, 20))

    if not semantic_results:
        print("[retrieval] FAISS unavailable, using keyword search")
        semantic_results = store.search_keyword(query_text, top_k=min(top_k * 2, 20))
        for r in semantic_results:
            r["semantic_score"] = r.pop("score", 0.3)

    scored = []
    for msg in semantic_results:
        sem = msg.get("semantic_score", 0.0)
        rec = _recency_score(msg["day"], max_day)
        emo = _emotion_score(msg.get("sentiment", 0.0))
        final = SEMANTIC_WEIGHT * sem + RECENCY_WEIGHT * rec + EMOTION_WEIGHT * emo

        scored.append({
            **msg,
            "score_semantic": round(sem, 4),
            "score_recency":  round(rec, 4),
            "score_emotion":  round(emo, 4),
            "score_final":    round(final, 4),
        })

    scored.sort(key=lambda x: x["score_final"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    store = MemoryStore()
    load_persona_into_store(store)

    print("\n── Query: 'Did I mention anything about my sister?' ──\n")
    results = query(store, "Did I mention anything about my sister?")
    for r in results:
        print(f"Day {r['day']} | score={r['score_final']:.3f} | {r['text'][:80]}")