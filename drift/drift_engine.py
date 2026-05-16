"""
Persona drift engine.

Drift score = 1 - cosine similarity between consecutive day embeddings.
Mood via VADER compound score + keyword heuristics.
Tone via formality heuristic (contraction rate, word length, punctuation).

Why not train a tone classifier?
  Only 7 days of data. A trained model would overfit badly.
  The heuristic is less accurate but more honest about its limitations.

Drift score thresholds:
  0.00 - 0.15 : stable
  0.15 - 0.30 : mild
  0.30 - 0.50 : moderate
  0.50+       : significant
"""

import json
import re
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from drift.embedder import get_embedder

DATA_PATH = Path(__file__).parent.parent / "data" / "persona_fixture.json"

_vader = SentimentIntensityAnalyzer()

CONTRACTIONS = re.compile(
    r"\b(i'm|i've|i'd|i'll|can't|won't|don't|didn't|it's|that's|there's|"
    r"we're|they're|you're|isn't|wasn't|couldn't|shouldn't|wouldn't|"
    r"lol|tbh|idk|btw|omg|ngl|lmao|imo|fyi)\b",
    re.IGNORECASE
)


def _tone_features(texts: list) -> dict:
    joined = " ".join(texts)
    words = joined.split()
    if not words:
        return {"formality_score": 0.5, "tone_label": "neutral"}

    contraction_rate = len(CONTRACTIONS.findall(joined)) / max(len(words), 1)
    avg_word_len = np.mean([len(w) for w in words])
    exclamation_rate = joined.count("!") / max(len(texts), 1)

    formality_score = (
        min(avg_word_len / 8.0, 1.0) * 0.4
        + max(0, 0.3 - contraction_rate) * 0.4
        + max(0, 0.2 - exclamation_rate * 0.1) * 0.2
    )

    if formality_score >= 0.55:
        tone_label = "formal"
    elif formality_score >= 0.35:
        tone_label = "slightly_casual"
    else:
        tone_label = "casual"

    return {
        "formality_score": round(float(formality_score), 3),
        "tone_label": tone_label,
        "contraction_rate": round(float(contraction_rate), 3),
        "avg_word_len": round(float(avg_word_len), 2),
    }


def _describe_mood(compound: float, texts: list) -> str:
    joined = " ".join(texts).lower()

    playful_signals = ["lol", "lmao", "haha", "funny", "hear me out", "honestly"]
    anxious_signals = ["scared", "anxious", "worried", "panic", "dread"]
    reflective_signals = ["not sure", "something feels", "wondering", "thinking about"]

    if any(s in joined for s in playful_signals) and compound >= 0.0:
        return "playful"
    if any(s in joined for s in anxious_signals):
        return "anxious"
    if any(s in joined for s in reflective_signals):
        return "reflective"

    if compound >= 0.4:
        return "positive"
    elif compound >= 0.1:
        return "warm"
    elif compound >= -0.1:
        return "neutral"
    elif compound >= -0.3:
        return "stressed"
    else:
        return "frustrated"


def _extract_triggers(session: dict, prev_topics: set) -> str:
    current_topics = set()
    people = set()

    for msg in session["messages"]:
        current_topics.update(msg.get("topics", []))
        people.update(msg.get("people_mentioned", []))

    if session.get("drift_trigger"):
        return session["drift_trigger"]

    new_topics = current_topics - prev_topics
    if people:
        return f"mention of: {', '.join(sorted(people))}"
    if new_topics:
        return f"new topics: {', '.join(sorted(new_topics))}"
    return None


def compute_drift_timeline(data: dict = None) -> list:
    if data is None:
        with open(DATA_PATH) as f:
            data = json.load(f)

    sessions = sorted(data["sessions"], key=lambda s: s["day"])
    embedder = get_embedder()

    day_embeddings = []
    for session in sessions:
        texts = [m["text"] for m in session["messages"]]
        vec = embedder.embed_mean(texts)
        day_embeddings.append(vec)

    timeline = []
    prev_topics = set()
    prev_compound = None

    for i, session in enumerate(sessions):
        texts = [m["text"] for m in session["messages"]]
        joined = " ".join(texts)

        sentiment = _vader.polarity_scores(joined)
        compound = sentiment["compound"]
        mood = _describe_mood(compound, texts)
        tone_data = _tone_features(texts)

        if i == 0:
            drift_score = 0.0
        else:
            v1 = day_embeddings[i - 1]
            v2 = day_embeddings[i]
            cosine_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            drift_score = round(1.0 - cosine_sim, 4)

        mood_delta = round(compound - prev_compound, 3) if prev_compound is not None else 0.0
        prev_compound = compound

        trigger = _extract_triggers(session, prev_topics)
        prev_topics = {t for m in session["messages"] for t in m.get("topics", [])}

        if drift_score < 0.15:
            drift_severity = "stable"
        elif drift_score < 0.30:
            drift_severity = "mild"
        elif drift_score < 0.50:
            drift_severity = "moderate"
        else:
            drift_severity = "significant"

        timeline.append({
            "day": session["day"],
            "date": session["date"],
            "mood": mood,
            "tone": tone_data["tone_label"],
            "formality_score": tone_data["formality_score"],
            "sentiment_compound": round(compound, 3),
            "mood_delta": mood_delta,
            "drift_score": drift_score,
            "drift_severity": drift_severity,
            "trigger": trigger,
            "summary": f"Day {session['day']} -> {mood} & {tone_data['tone_label']}",
            "message_count": len(session["messages"]),
            "topics": list({t for m in session["messages"] for t in m.get("topics", [])}),
        })

    return timeline


if __name__ == "__main__":
    timeline = compute_drift_timeline()
    print("\n── Persona Drift Timeline ──\n")
    for entry in timeline:
        drift_str = f"drift={entry['drift_score']:.3f} ({entry['drift_severity']})"
        trigger_str = f"trigger: {entry['trigger']}" if entry['trigger'] else "no trigger"
        print(f"Day {entry['day']:>2} | {entry['mood']:<12} & {entry['tone']:<18} | {drift_str} | {trigger_str}")