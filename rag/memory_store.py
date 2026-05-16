"""
Memory store: SQLite for structured data, FAISS for vector search.

SQLite is the canonical store. FAISS index is derived — rebuilt from
SQLite on startup. This prevents index/database divergence bugs.

Why not ChromaDB? It adds 80MB+ of dependencies. SQLite ships with
Python, faiss-cpu is a pure C extension. Simpler, lighter, more transparent.
"""

import json
import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"

_faiss = None

def _get_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            _faiss = False
    return _faiss if _faiss is not False else None


def _init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            day         INTEGER NOT NULL,
            date        TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            text        TEXT NOT NULL,
            topics      TEXT NOT NULL,
            people      TEXT NOT NULL,
            sentiment   REAL,
            embedding   BLOB
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_day ON messages(day)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON messages(date)")
    conn.commit()


class MemoryStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _init_db(self.conn)

        self._index = None
        self._index_ids = []
        self._dim = None

    def upsert_message(self, msg_id, day, date, timestamp, text, topics, people, sentiment, embedding):
        emb_bytes = embedding.astype(np.float32).tobytes()
        self.conn.execute("""
            INSERT OR REPLACE INTO messages
            (id, day, date, timestamp, text, topics, people, sentiment, embedding)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (msg_id, day, date, timestamp, text,
              json.dumps(topics), json.dumps(people), sentiment, emb_bytes))
        self.conn.commit()
        self._add_to_index(msg_id, embedding)

    def _add_to_index(self, msg_id: str, embedding: np.ndarray):
        faiss = _get_faiss()
        if faiss is None:
            return

        vec = embedding.astype(np.float32).reshape(1, -1)
        dim = vec.shape[1]

        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(vec)
        self._index_ids.append(msg_id)

    def get_all(self) -> list:
        rows = self.conn.execute(
            "SELECT id, day, date, timestamp, text, topics, people, sentiment FROM messages ORDER BY day, timestamp"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_day(self, day: int) -> list:
        rows = self.conn.execute(
            "SELECT id, day, date, timestamp, text, topics, people, sentiment FROM messages WHERE day=? ORDER BY timestamp",
            (day,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search_keyword(self, query: str, top_k: int = 10) -> list:
        terms = query.lower().split()
        rows = self.conn.execute(
            "SELECT id, day, date, timestamp, text, topics, people, sentiment FROM messages"
        ).fetchall()
        results = []
        for r in rows:
            text_lower = r[4].lower()
            score = sum(1 for t in terms if t in text_lower) / len(terms)
            if score > 0:
                d = self._row_to_dict(r)
                d["score"] = score
                results.append(d)
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def search_semantic(self, query_embedding: np.ndarray, top_k: int = 10) -> list:
        faiss = _get_faiss()
        if faiss is None or self._index is None or self._index.ntotal == 0:
            return []

        vec = query_embedding.astype(np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            msg_id = self._index_ids[idx]
            rows = self.conn.execute(
                "SELECT id, day, date, timestamp, text, topics, people, sentiment FROM messages WHERE id=?",
                (msg_id,)
            ).fetchall()
            if rows:
                d = self._row_to_dict(rows[0])
                d["semantic_score"] = float(score)
                results.append(d)

        return results

    def rebuild_index(self):
        faiss = _get_faiss()
        if faiss is None:
            return

        rows = self.conn.execute(
            "SELECT id, embedding FROM messages ORDER BY day, timestamp"
        ).fetchall()
        if not rows:
            return

        self._index = None
        self._index_ids = []
        for msg_id, emb_bytes in rows:
            if emb_bytes:
                vec = np.frombuffer(emb_bytes, dtype=np.float32)
                self._add_to_index(msg_id, vec)

        print(f"[memory_store] Rebuilt FAISS index with {len(self._index_ids)} vectors")

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "id": row[0],
            "day": row[1],
            "date": row[2],
            "timestamp": row[3],
            "text": row[4],
            "topics": json.loads(row[5]),
            "people": json.loads(row[6]),
            "sentiment": row[7],
        }