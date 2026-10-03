"""FastAPI orchestrator — REST + WebSocket front for the Q-Trinetra engine.

    uvicorn backend.app:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /                      dashboard (frontend/index.html)
GET  /api/scenarios         list of attack scenarios (digital twin)
GET  /api/decision-table    the auditable fusion rule table
POST /api/run               {"scenario": "...", "seed": 1, "L": 4800, "backend": "fast"} → SessionResult
GET  /api/demo              run the 5-step demo, returns list of SessionResults
GET  /api/ledger            Q-Ledger records + Merkle batches + chain validity
GET  /api/ledger/{idx}/proof  Merkle inclusion proof
GET  /api/bench/quick       light benchmark numbers for the dashboard
WS   /ws/live               streams SPRT LLR trace step-by-step for a scenario
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qtrinetra.attacks.adversaries import SCENARIO_NAMES, build_scenario
from qtrinetra.detect.fusion import decision_table_export
from qtrinetra.ledger.qledger import QLedger
from qtrinetra.pipeline import run_session
from qtrinetra.protocol.qds import ProtocolParams

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Q-Trinetra API", version="0.1.0",
              description="Pauli-spectrum threat forensics for teleportation-based QDS (SIH 2026 PS 26141)")
LEDGER = QLedger(os.environ.get("QT_LEDGER_PATH", ":memory:"))


class RunRequest(BaseModel):
    scenario: str = "honest"
    seed: int | None = None
    L: int = Field(4800, ge=300, le=200_000)
    s_a: float = 0.08
    s_v: float = 0.14
    e0: float = 0.01
    p_min: float = 1 / 6
    backend: str = "fast"
    param: float | None = None
    message: str = "TRANSFER 1,00,000 INR -> ACC 4471"


def _params(req: RunRequest) -> ProtocolParams:
    if req.backend not in ("fast", "stim"):
        raise HTTPException(400, "backend must be 'fast' or 'stim'")
    return ProtocolParams(L=req.L, s_a=req.s_a, s_v=req.s_v, e0=req.e0, p_min=req.p_min, backend=req.backend)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="32" cy="32" r="30" fill="#0b1220"/>'
           '<circle cx="32" cy="32" r="16" fill="#38bdf8"/><circle cx="32" cy="32" r="7" fill="#fff"/></svg>')
    return Response(svg, media_type="image/svg+xml")


@app.get("/api/scenarios")
def scenarios():
    return [build_scenario(n).as_dict() for n in SCENARIO_NAMES]


@app.get("/api/decision-table")
def decision_table():
    return decision_table_export()


@app.post("/api/run")
def run(req: RunRequest):
    if req.scenario not in SCENARIO_NAMES:
        raise HTTPException(404, f"unknown scenario; choose from {SCENARIO_NAMES}")
    sc = build_scenario(req.scenario, req.param)
    r = run_session(sc, params=_params(req), seed=req.seed, ledger=LEDGER, message=req.message.encode())
    return JSONResponse(json.loads(json.dumps(r.as_dict(), default=str)))


@app.get("/api/demo")
def demo(seed: int = 42):
    steps = ["honest", "measure_guess_forger", "intercept_resend_Z", "fake_source", "replay_same_block"]
    out = [json.loads(json.dumps(run_session(s, seed=seed, ledger=LEDGER).as_dict(), default=str)) for s in steps]
    return out


@app.get("/api/ledger")
def ledger(limit: int = 100):
    return {"length": len(LEDGER), "chain_valid": LEDGER.verify_chain(), "head": LEDGER.head_hash(),
            "batches": LEDGER.batches(), "records": [r.as_dict() for r in LEDGER.all(limit)]}


@app.get("/api/ledger/{idx}/proof")
def ledger_proof(idx: int):
    if LEDGER.get(idx) is None:
        raise HTTPException(404, "no such record")
    return LEDGER.merkle_proof(idx)


@app.get("/api/bench/quick")
def bench_quick():
    from qtrinetra.analytics.evaluate import forgery_probability_vs_L, sprt_savings, throughput
    return {"forgery_probability_vs_L": forgery_probability_vs_L(trials=2000),
            "sprt_savings": sprt_savings(trials=100), "throughput": throughput(n=50_000)}


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Client sends {"scenario": ..., "seed": ...}; server streams the SPRT LLR trace
    one step at a time (so the dashboard curve animates), then the final report."""
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            scenario = msg.get("scenario", "honest")
            if scenario not in SCENARIO_NAMES:
                await ws.send_json({"type": "error", "detail": "unknown scenario"})
                continue
            r = run_session(scenario, seed=msg.get("seed"), ledger=LEDGER)
            payload = json.loads(json.dumps(r.as_dict(), default=str))
            trace = payload["sprt"]["llr_trace"] if payload.get("sprt") else []
            await ws.send_json({"type": "start", "scenario": payload["scenario"],
                                "A": payload["sprt"]["A"] if trace else None,
                                "B": payload["sprt"]["B"] if trace else None, "n": len(trace)})
            step = max(1, len(trace) // 60)
            for i in range(0, len(trace), step):
                await ws.send_json({"type": "llr", "i": i + 1, "llr": trace[i]})
                await asyncio.sleep(0.02)
            await ws.send_json({"type": "result", "session": payload})
    except WebSocketDisconnect:
        return


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
