"""M5 — Q-Ledger: hash-chained, Merkle-anchored key-spend ledger (replay guard).

Each record:   H( session_nonce || key_block_id || message_hash || verdict )
Records are chained with SHA3-256 (prev_hash included) and batched into
Merkle roots every ``batch_size`` records.  Backed by SQLite (``:memory:`` by
default) so the demo needs no external database; the schema is identical for
PostgreSQL.

API
---
    ledger.is_spent(key_block_id, message_bit) -> bool          (D5 lookup, O(1))
    ledger.append(...)  -> LedgerRecord
    ledger.verify_chain() -> bool
    ledger.merkle_proof(index) -> list[(hash, side)]
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass

GENESIS = "0" * 64


def h(*parts: str) -> str:
    m = hashlib.sha3_256()
    for p in parts:
        m.update(p.encode())
        m.update(b"|")
    return m.hexdigest()


@dataclass
class LedgerRecord:
    index: int
    timestamp: float
    session_nonce: str
    key_block_id: str
    message_bit: int
    message_hash: str
    verdict: str
    attack_class: str
    threat_index: float
    record_hash: str
    prev_hash: str
    chain_hash: str
    merkle_root: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return GENESIS
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [h(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: list[str], idx: int) -> list[tuple[str, str]]:
    proof = []
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sib = idx ^ 1
        proof.append((level[sib], "L" if sib < idx else "R"))
        level = [h(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2
    return proof


def verify_merkle_proof(leaf: str, proof: list[tuple[str, str]], root: str) -> bool:
    cur = leaf
    for sib, side in proof:
        cur = h(sib, cur) if side == "L" else h(cur, sib)
    return cur == root


class QLedger:
    def __init__(self, path: str = ":memory:", batch_size: int = 8):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.batch_size = batch_size
        self._init()

    def _init(self) -> None:
        c = self.conn
        c.execute("""CREATE TABLE IF NOT EXISTS records (
            idx INTEGER PRIMARY KEY, ts REAL, session_nonce TEXT, key_block_id TEXT,
            message_bit INTEGER, message_hash TEXT, verdict TEXT, attack_class TEXT,
            threat_index REAL, record_hash TEXT, prev_hash TEXT, chain_hash TEXT, merkle_root TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS spent (
            key_block_id TEXT, message_bit INTEGER, idx INTEGER,
            PRIMARY KEY (key_block_id, message_bit))""")
        c.execute("""CREATE TABLE IF NOT EXISTS merkle_batches (
            batch INTEGER PRIMARY KEY, first_idx INTEGER, last_idx INTEGER, root TEXT)""")
        c.commit()

    # -- queries -------------------------------------------------------------
    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def head_hash(self) -> str:
        row = self.conn.execute("SELECT chain_hash FROM records ORDER BY idx DESC LIMIT 1").fetchone()
        return row[0] if row else GENESIS

    def is_spent(self, key_block_id: str, message_bit: int) -> bool:
        return self.conn.execute("SELECT 1 FROM spent WHERE key_block_id=? AND message_bit=?",
                                 (key_block_id, message_bit)).fetchone() is not None

    def spent_record(self, key_block_id: str, message_bit: int) -> LedgerRecord | None:
        row = self.conn.execute("SELECT idx FROM spent WHERE key_block_id=? AND message_bit=?",
                                (key_block_id, message_bit)).fetchone()
        return self.get(row[0]) if row else None

    def get(self, idx: int) -> LedgerRecord | None:
        row = self.conn.execute("SELECT * FROM records WHERE idx=?", (idx,)).fetchone()
        return LedgerRecord(*row) if row else None

    def all(self, limit: int = 200) -> list[LedgerRecord]:
        rows = self.conn.execute("SELECT * FROM records ORDER BY idx DESC LIMIT ?", (limit,)).fetchall()
        return [LedgerRecord(*r) for r in rows]

    def batches(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM merkle_batches ORDER BY batch").fetchall()
        return [{"batch": b, "first_idx": f, "last_idx": l, "root": r} for b, f, l, r in rows]

    # -- mutation ------------------------------------------------------------
    def append(self, session_nonce: str, key_block_id: str, message_bit: int, message_hash: str,
               verdict: str, attack_class: str = "NONE", threat_index: float = 0.0,
               mark_spent: bool = True) -> LedgerRecord:
        idx = len(self)
        prev = self.head_hash()
        rec_hash = h(session_nonce, key_block_id, str(message_bit), message_hash, verdict)
        chain = h(prev, rec_hash)
        rec = LedgerRecord(idx, time.time(), session_nonce, key_block_id, message_bit, message_hash,
                           verdict, attack_class, float(threat_index), rec_hash, prev, chain)
        self.conn.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (rec.index, rec.timestamp, rec.session_nonce, rec.key_block_id,
                           rec.message_bit, rec.message_hash, rec.verdict, rec.attack_class,
                           rec.threat_index, rec.record_hash, rec.prev_hash, rec.chain_hash, None))
        if mark_spent and verdict in ("ACCEPT", "TRANSFERABLE_UNCERTAIN") and \
                not self.is_spent(key_block_id, message_bit):
            self.conn.execute("INSERT INTO spent VALUES (?,?,?)", (key_block_id, message_bit, idx))
        self.conn.commit()
        if (idx + 1) % self.batch_size == 0:
            self._anchor_batch(idx + 1 - self.batch_size, idx)
        return self.get(idx)  # type: ignore[return-value]

    def _anchor_batch(self, first: int, last: int) -> None:
        leaves = [r[0] for r in self.conn.execute(
            "SELECT chain_hash FROM records WHERE idx BETWEEN ? AND ? ORDER BY idx", (first, last))]
        root = merkle_root(leaves)
        batch_no = first // self.batch_size
        self.conn.execute("INSERT OR REPLACE INTO merkle_batches VALUES (?,?,?,?)",
                          (batch_no, first, last, root))
        self.conn.execute("UPDATE records SET merkle_root=? WHERE idx BETWEEN ? AND ?", (root, first, last))
        self.conn.commit()

    # -- integrity -----------------------------------------------------------
    def verify_chain(self) -> bool:
        prev = GENESIS
        for r in self.conn.execute("SELECT * FROM records ORDER BY idx").fetchall():
            rec = LedgerRecord(*r)
            if rec.prev_hash != prev:
                return False
            if rec.record_hash != h(rec.session_nonce, rec.key_block_id, str(rec.message_bit),
                                    rec.message_hash, rec.verdict):
                return False
            if rec.chain_hash != h(prev, rec.record_hash):
                return False
            prev = rec.chain_hash
        return True

    def merkle_proof(self, idx: int) -> dict:
        batch_no = idx // self.batch_size
        row = self.conn.execute("SELECT first_idx,last_idx,root FROM merkle_batches WHERE batch=?",
                                (batch_no,)).fetchone()
        if not row:
            return {"anchored": False}
        first, last, root = row
        leaves = [r[0] for r in self.conn.execute(
            "SELECT chain_hash FROM records WHERE idx BETWEEN ? AND ? ORDER BY idx", (first, last))]
        proof = merkle_proof(leaves, idx - first)
        return {"anchored": True, "root": root, "leaf": leaves[idx - first], "proof": proof,
                "valid": verify_merkle_proof(leaves[idx - first], proof, root)}

    def export_json(self) -> str:
        return json.dumps([r.as_dict() for r in self.all(10_000)], indent=1)
