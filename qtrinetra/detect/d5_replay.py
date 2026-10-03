"""D5 — Freshness & Replay Guard (Q-Ledger lookup).

Every key block may be spent once.  Re-submitting a spent (key_block_id,
message_bit) is caught deterministically (p = 0).  Replaying an old revealed
key against a *fresh* block yields ~50 % mismatches and is caught by D1.
"""
from __future__ import annotations

from qtrinetra.detect.base import DetectorResult
from qtrinetra.ledger.qledger import QLedger
from qtrinetra.protocol.qds import Signature


def detect_d5(sig: Signature, ledger: QLedger, session_nonce: str) -> DetectorResult:
    spent = ledger.is_spent(sig.key_block_id, sig.message_bit)
    prev = ledger.spent_record(sig.key_block_id, sig.message_bit) if spent else None
    stale_nonce = prev is not None and prev.session_nonce == session_nonce
    same_msg = prev is not None and prev.message_hash == sig.message_hash()
    if spent:
        expl = (f"Key block '{sig.key_block_id}' bit {sig.message_bit} was already spent at ledger "
                f"index {prev.index} ({'same' if same_msg else 'DIFFERENT'} message hash) → REPLAY.")
        return DetectorResult("D5", "Freshness & Replay Guard", 1e-300, True, "alarm",
                              stats={"spent": True, "prev_index": prev.index,
                                     "prev_message_hash": prev.message_hash,
                                     "same_message": same_msg, "stale_nonce": stale_nonce,
                                     "ledger_len": len(ledger), "chain_valid": ledger.verify_chain()},
                              explanation=expl)
    return DetectorResult("D5", "Freshness & Replay Guard", 1.0, False, "ok",
                          stats={"spent": False, "ledger_len": len(ledger),
                                 "chain_valid": ledger.verify_chain()},
                          explanation=f"Key block '{sig.key_block_id}' is fresh; nonce {session_nonce} bound.")
