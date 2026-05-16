---
title: MnemoSync
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.11"
app_file: app.py
pinned: false
---


# MnemoSync

**Local-first memory integrity engine for persistent conversational agents.**

Most memory systems store and retrieve. MnemoSync treats memory as unreliable
by design — people contradict themselves, moods shift, context degrades.
The system's job is not just retrieval, it's *reconciliation*.

---

## What It Does

| Module | What it solves |
|--------|----------------|
| Persona Drift Engine | Tracks mood/tone shifts across days, detects triggers |
| Intent Classifier | Classifies messages offline (<5ms, ~504KB model) |
| Conflict-Aware RAG | Retrieves memory, detects contradictions, merges coherently |
| Sync Architecture | Local-first design with optional cloud metadata sync |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/mnemosync
cd mnemosync
pip install -r requirements.txt
```

**Streamlit demo:**
```bash
streamlit run app.py
```

**FastAPI backend:**
```bash
uvicorn api:app --reload
# Docs at http://localhost:8000/docs
```

**Test individual modules:**
```bash
python main.py
python -m classifier.intent_classifier
python -m drift.drift_engine
python -m rag.conflict_resolver
```

---

## Architecture

mnemosync/
├── app.py                    # Streamlit demo
├── api.py                    # FastAPI backend
├── main.py                   # Integration test
├── data/
│   ├── persona_fixture.json  # 7-day synthetic persona
│   └── memory.db             # SQLite (auto-generated)
├── classifier/
│   ├── train_data.py         # 125 labeled training examples
│   ├── intent_classifier.py  # TF-IDF + CalibratedLinearSVC
│   └── model/                # .pkl file (auto-generated)
├── drift/
│   ├── embedder.py           # sentence-transformers + TF-IDF fallback
│   └── drift_engine.py       # Cosine drift + VADER + tone heuristics
├── rag/
│   ├── memory_store.py       # SQLite + FAISS
│   ├── retrieval_engine.py   # Weighted ranking
│   └── conflict_resolver.py  # Contradiction detection + merge
└── docs/
└── system_design.md      # Sync architecture (Part 4)


---

## Design Decisions

**Intent Classifier: TF-IDF + LinearSVC over transformer**
Constraint was <50MB and <200ms. TransformerS cost 260MB+. TF-IDF char n-grams
handle informal text (lol, idk, tbh) well. Final model: 504KB, ~2ms inference.
Tradeoff: CV F1 ~0.65 on 125 examples. Clear-cut cases work well.

**Drift Detection: Cosine similarity + VADER, not a classifier**
Only 7 days of training data. A classifier would overfit. VADER compound
as mood proxy is accurate for personal diary text. Tone heuristic (formality
score) is interpretable and 0ms.

**Contradiction Detection: Sentiment polarity, not NLI**
DeBERTa-mnli costs ~180MB. For personal memory with named entities, sentiment
polarity flip + negation keywords is a reliable proxy. Documented tradeoff.

**Storage: SQLite + FAISS, not ChromaDB**
ChromaDB adds 80MB+. SQLite ships with Python. FAISS index always rebuilt
from SQLite — prevents index/database divergence.

---

## Benchmarks

| Component | Metric | Result |
|-----------|--------|--------|
| Intent model size | .pkl | 504 KB |
| Intent inference | per message | ~2ms |
| Drift timeline | 7 days | ~180ms (TF-IDF) |
| RAG query + resolve | end-to-end | ~90ms |

---

## Known Limitations

1. CV F1 ~0.65 on 125 training examples. Short/ambiguous messages misclassify.
2. Contradiction detection is heuristic — misses non-sentiment contradictions.
3. TF-IDF drift scores less meaningful than transformer embeddings.
4. Cloud sync designed but not implemented.
5. VADER struggles with sarcasm.

---

## Future Improvements

- Active learning loop for classifier improvement
- Quantised ONNX NLI model (~15MB) for better contradiction detection
- FAISS IVF index for >10k messages
- SQLCipher for encrypted local storage
- Cloud sync layer (DynamoDB + Lambda)

---

## Demo

Hosted: 
Loom: 

