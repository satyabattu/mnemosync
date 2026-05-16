import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV

MODEL_PATH = Path(__file__).parent / "model" / "intent_model.pkl"


def train(save: bool = True) -> Pipeline:
    from classifier.train_data import TRAINING_DATA

    texts, labels = zip(*TRAINING_DATA)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        max_features=8000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    base_svc = LinearSVC(C=0.8, max_iter=2000, class_weight="balanced")

    X = vectorizer.fit_transform(texts)
    base_svc.fit(X, labels)

    calibrated = CalibratedClassifierCV(base_svc, cv="prefit", method="sigmoid")
    calibrated.fit(X, labels)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", calibrated),
    ])

    # CV score on uncalibrated version for reporting only
    scores = cross_val_score(
        Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), max_features=8000, sublinear_tf=True, lowercase=True)),
            ("clf", LinearSVC(C=0.8, max_iter=2000, class_weight="balanced"))
        ]),
        texts, labels, cv=3, scoring="f1_weighted"
    )
    print(f"[classifier] CV F1: {scores.mean():.3f} +/- {scores.std():.3f}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        size_kb = MODEL_PATH.stat().st_size / 1024
        print(f"[classifier] Saved to {MODEL_PATH} ({size_kb:.1f} KB)")

    return pipeline


def load() -> Pipeline:
    if not MODEL_PATH.exists():
        print("[classifier] Model not found, training now...")
        return train()
    return joblib.load(MODEL_PATH)


class IntentClassifier:
    def __init__(self):
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is None:
            self._pipeline = load()

    def predict(self, text: str) -> dict:
        self._ensure_loaded()

        text = text.strip()
        if not text:
            return {"label": "unknown", "confidence": 0.0, "raw_text": text}

        proba = self._pipeline.predict_proba([text])[0]
        classes = self._pipeline.classes_

        max_idx = int(np.argmax(proba))
        label = classes[max_idx]
        confidence = float(proba[max_idx])

        if confidence < 0.35:
            label = "unknown"

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "raw_text": text,
            "all_scores": {cls: round(float(p), 3) for cls, p in zip(classes, proba)},
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


if __name__ == "__main__":
    import time
    print("Training classifier...")
    train(save=True)
    clf = IntentClassifier()
    test_cases = [
        ("remind me to call mum tomorrow at 6", "reminder"),
        ("i feel so overwhelmed and lost right now", "emotional-support"),
        ("finish the API docs before the sprint ends", "action-item"),
        ("lol nothing much just chilling", "small-talk"),
        ("asdf jkl;", "unknown"),
    ]
    print("\n── Inference Test ──")
    for text, expected in test_cases:
        start = time.perf_counter()
        result = clf.predict(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        match = "OK" if result["label"] == expected else "MISS"
        print(f"{match} [{elapsed_ms:.1f}ms] '{text}' -> {result['label']} ({result['confidence']})")