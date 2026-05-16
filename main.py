"""
End-to-end integration test.
Run: python main.py
"""

def separator(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def test_classifier():
    separator("Part 2 — Intent Classifier")
    from classifier.intent_classifier import IntentClassifier
    clf = IntentClassifier()
    cases = [
        "remind me to submit the assignment by tonight",
        "i've been feeling really anxious all week",
        "push the hotfix before the standup",
        "lol nothing much just watching youtube",
        "asdfjkl;",
    ]
    for text in cases:
        r = clf.predict(text)
        print(f"  [{r['label']:<18}] {r['confidence']:.0%}  '{text[:50]}'")


def test_drift():
    separator("Part 1 — Drift Timeline")
    from drift.drift_engine import compute_drift_timeline
    timeline = compute_drift_timeline()
    for entry in timeline:
        print(
            f"  Day {entry['day']} | {entry['mood']:<12} & {entry['tone']:<18} | "
            f"drift={entry['drift_score']:.3f} ({entry['drift_severity']:<12}) | "
            f"trigger: {entry['trigger'] or 'none'}"
        )


def test_rag():
    separator("Part 3 — Memory Query + Conflict Resolution")
    from rag.memory_store import MemoryStore
    from rag.retrieval_engine import load_persona_into_store, query as retrieve
    from rag.conflict_resolver import detect_contradictions, resolve_conflicts

    store = MemoryStore()
    load_persona_into_store(store)

    q = "Did I mention anything about my sister?"
    print(f"\n  Query: '{q}'")

    chunks = retrieve(store, q, top_k=6)
    contradictions = detect_contradictions(chunks)
    result = resolve_conflicts(q, chunks, contradictions)

    print(f"\n  Contradiction detected: {result['has_contradiction']}")
    if result["contradiction_note"]:
        print(f"  Note: {result['contradiction_note']}")
    print(f"\n  Merged narrative:\n  {result['merged_narrative'][:250]}")
    print(f"\n  Source trace:")
    for s in result["source_trace"]:
        print(f"    Day {s['day']} [{s['status']}]")


if __name__ == "__main__":
    print("\nMnemoSync - Integration Test")
    try:
        test_classifier()
    except Exception as e:
        print(f"  ERROR: {e}")
    try:
        test_drift()
    except Exception as e:
        print(f"  ERROR: {e}")
    try:
        test_rag()
    except Exception as e:
        print(f"  ERROR: {e}")
    print("\nDone.\n")