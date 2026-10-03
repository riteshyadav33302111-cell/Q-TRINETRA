"""Wegman–Carter information-theoretic MAC (M0).

Polynomial universal hashing over the prime field GF(p) with p = 2^61 - 1
(a Mersenne prime, so reduction is cheap and exact in Python ints), followed
by a one-time pad.  For a message of ``n`` blocks the forgery probability of
an adversary without the key is at most ``n / p`` (~ n * 4e-19) regardless of
computing power — this is what authenticates the classical correction-bit
stream (m1, m2) in the teleportation step.

Key material: ``(r, s)`` fresh per message; ``r`` is the hash key, ``s`` the
one-time pad.  In the prototype keys are drawn from an in-memory
pre-shared pool (``MacKeyPool``) which models the output of QKD.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field

P = (1 << 61) - 1          # Mersenne prime 2^61 - 1
BLOCK_BYTES = 7            # 56-bit blocks < P


def _blocks(msg: bytes) -> list[int]:
    # Length-prefix to make the encoding injective.
    data = len(msg).to_bytes(8, "big") + msg
    pad = (-len(data)) % BLOCK_BYTES
    data += b"\x00" * pad
    return [int.from_bytes(data[i:i + BLOCK_BYTES], "big") for i in range(0, len(data), BLOCK_BYTES)]


def poly_hash(msg: bytes, r: int) -> int:
    """h_r(m) = sum_i m_i * r^(n-i) mod P  (Horner form)."""
    h = 0
    for b in _blocks(msg):
        h = (h * r + b) % P
    return h


def wc_tag(msg: bytes, r: int, s: int) -> int:
    """Wegman–Carter tag: (h_r(m) + s) mod P."""
    return (poly_hash(msg, r) + s) % P


def wc_verify(msg: bytes, tag: int, r: int, s: int) -> bool:
    return wc_tag(msg, r, s) == tag


@dataclass
class MacKeyPool:
    """Pre-shared one-time MAC keys (models QKD-derived key material)."""
    _keys: list[tuple[int, int]] = field(default_factory=list)
    consumed: int = 0

    @classmethod
    def generate(cls, n: int, seed: int | None = None) -> "MacKeyPool":
        if seed is None:
            keys = [(secrets.randbelow(P), secrets.randbelow(P)) for _ in range(n)]
        else:
            import numpy as np
            rng = np.random.default_rng(seed)
            keys = [(int(rng.integers(0, P)), int(rng.integers(0, P))) for _ in range(n)]
        return cls(keys)

    def next_key(self) -> tuple[int, int]:
        if self.consumed >= len(self._keys):
            raise RuntimeError("MAC key pool exhausted")
        k = self._keys[self.consumed]
        self.consumed += 1
        return k

    def key_at(self, idx: int) -> tuple[int, int]:
        return self._keys[idx]

    def __len__(self) -> int:
        return len(self._keys)


@dataclass(frozen=True)
class AuthenticatedMessage:
    payload: bytes
    key_index: int
    tag: int


def authenticate(pool_sender: MacKeyPool, payload: bytes) -> AuthenticatedMessage:
    idx = pool_sender.consumed
    r, s = pool_sender.next_key()
    return AuthenticatedMessage(payload, idx, wc_tag(payload, r, s))


def check(pool_receiver: MacKeyPool, msg: AuthenticatedMessage) -> bool:
    r, s = pool_receiver.key_at(msg.key_index)
    return wc_verify(msg.payload, msg.tag, r, s)
