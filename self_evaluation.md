# Self-Evaluation — MnemoSync

## Part 1: Adaptive Persona Engine — 7/10

What works: Drift score correctly identifies day-to-day semantic shift.
Mood labels (playful, anxious, reflective, frustrated) match the fixture
ground truth. Trigger detection correctly identifies family events and
work stress as primary drivers.

Limitation: Tone classifier is a heuristic. Labels most days as
"slightly_casual" which is technically correct but not very granular.
A proper approach would use 20-30 labeled examples per tone category.
VADER also struggles with compound emotions — Day 5 (mixed relief and
family conflict) outputs "reflective" which is right for wrong reasons.

## Part 2: Offline Intent Classifier — 8/10

What works: 504KB model, ~2ms inference, no external APIs, handles
informal text well via char n-grams.

Limitation: CV F1 is 0.65 on 125 examples. Works well on clear cases,
struggles with short or ambiguous messages like "I'm fine" or "fix this".
I chose LinearSVC over logistic regression because it generalises better
on small high-dimensional sparse feature sets. CalibratedClassifierCV is
necessary because LinearSVC does not natively output probabilities.

I did not use a transformer — DistilBERT gives ~10-15% better F1 but
costs 260MB and violates the constraint.

## Part 3: Conflict Resolution in RAG — 7/10

What works: Correctly retrieves sister mentions from Days 2, 5, 7.
Correctly identifies Day 2/Day 5 as contradictory (positive vs negative
sentiment about same person). Merged narrative is coherent and chronological.
Every chunk tagged with source and status.

Limitation: Contradiction detection is heuristic. Would miss:
- Factual corrections without sentiment difference
- Sarcastic text
- Contradictions across topic boundaries

An ONNX-quantised NLI model (~15MB) would fix this properly within budget.

## Part 4: System Design — 8/10

Honest 1-page design. Local-first architecture is coherent. SQLite as
canonical store, FAISS as derived artifact, metadata-only sync by default.
Sync conflict resolution (last-write-wins + manual override, no silent drops)
is practical and correct.

Missing: multi-device clock skew handling, specific Lambda sync function design.
SQLCipher not implemented.

## Overall

Strongest: Part 3 — conflict resolver produces genuinely useful output
that a naive RAG system wouldn't.

Weakest: Part 1 tone detection — heuristic is too coarse.

With 48 hours: 200+ more training examples, replace tone heuristic with
small labeled dataset, test quantised ONNX NLI model.

With a week: learn retrieval weights from user signals, implement cloud
sync layer, add proper unit tests.