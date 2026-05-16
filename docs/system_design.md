# MnemoSync — Sync Architecture Design

## Overview

MnemoSync is designed local-first. The device is the source of truth.
Cloud sync is additive — the system works fully offline.

---

## Storage Architecture

### On-Device Storage

| Layer      | Technology   | What lives here                                 |
|------------|--------------|-------------------------------------------------|
| Structured | SQLite       | Messages, timestamps, topics, people, sentiment |
| Vector     | FAISS (flat) | Embeddings for semantic search                  |
| Models     | .pkl files   | Intent classifier (~504 KB)                     |
| Config     | JSON / env   | User preferences, sync settings                 |

SQLite is the canonical store. The FAISS index is rebuilt from SQLite on
startup — never treated as primary. This prevents index/database divergence.

### What Stays Local Only

- Raw message text (privacy — user controls this)
- Full conversation history
- Embedding vectors (regenerated from local text on each device)

### What Syncs to Cloud

- Message metadata: id, day, timestamp, topics, people, sentiment
- Drift timeline summaries
- User-confirmed conflict resolutions

Raw text never leaves the device unless the user explicitly opts into
full cloud backup. Privacy-first default.

---

## Sync Architecture

Device A                  Cloud (S3 + DynamoDB)         Device B

SQLite --- metadata ---> DynamoDB                <--- SQLite
FAISS      (no raw text)                               FAISS
<-- pull on open --------------------------->


Sync triggers: on app open + every 15 minutes when connected.

---

## Conflict Resolution Strategy

### 1. Temporal Conflicts (same entity, different times)

Not true conflicts — state transitions. Resolution: chronological ordering
with recency preference. Sister being "doing well" on Day 2 and "having an
argument" on Day 5 are both true — at different points in time.

### 2. Sync Conflicts (same timestamp, different devices)

Resolution: timestamp-based last-write-wins with manual override.

1. Detect: same (user_id, approximate_timestamp), different content
2. Auto-resolve: keep message with later device_write_timestamp
3. Surface conflict in UI with both versions
4. User chooses which to keep

The system never silently drops conflicting versions. Both stored with
conflict_flag=true until the user resolves.

---

## Architecture Diagram

+----------------------------------------------------------+
|                        DEVICE                            |
|                                                          |
|  Messages  --embed-->  FAISS Vector Index                |
|  (SQLite)              (in-memory, rebuilt on startup)   |
|      |                        |                          |
|      |                  semantic search                  |
|      v                        v                          |
|  +------------------------------------------------+      |
|  |           Retrieval Engine                     |      |
|  |  0.5sem + 0.3recency + 0.2*emotion           |      |
|  +------------------------------------------------+      |
|                        |                                 |
|                        v                                 |
|  +------------------------------------------------+      |
|  |         Conflict Resolver                      |      |
|  |  detect -> tag -> merge narrative              |      |
|  +------------------------------------------------+      |
|                                                          |
|  Drift Engine          Intent Classifier                 |
|  (VADER + embeddings)  (TF-IDF + LinearSVC)              |
+----------------------------+-----------------------------+
|  metadata only
v
+----------------------------------------------------------+
|               CLOUD (optional)                           |
|  DynamoDB: message metadata, drift summaries             |
|  S3: encrypted raw text backup (opt-in only)             |
+----------------------------------------------------------+

---

## Security Considerations

- Raw text local-only by default
- Embeddings not synced by default
- SQLite unencrypted (production: SQLCipher)
- Cloud sync over HTTPS with per-user encryption keys
- No third-party API calls in the critical path

---

## Known Simplifications

1. FAISS flat index — fine for <10k messages. Production: IVF index.
2. Cloud sync designed, not implemented.
3. SQLite unencrypted in this implementation.

