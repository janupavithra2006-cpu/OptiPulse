#!/usr/bin/env python3
"""
api_server.py — OptiPulse-Pay Asynchronous Payment Gateway
FastAPI server on 127.0.0.1:8000 with in-memory ledger and 1:N cosine vector matching.
"""

import json
import time
import hmac
import hashlib
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ── Configuration ──────────────────────────────────────────────────────────────
ENROLLED_DB_PATH   = Path("kyc_database/enrolled_users.json")
COSINE_THRESHOLD   = 0.65          # τ minimum for identity match (calibrated for live webcam)
TRANSACTION_AMOUNT = 150.0         # ₹ per transaction
SECRET_KEY         = b"optipulse-secret-hmac-key-v1"  # HMAC signing key
LOG_PATH           = Path("logs/transactions.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("optipulse.api")

# ── In-Memory Ledger ───────────────────────────────────────────────────────────
# Structure: { user_id: { ...profile, "vector": np.ndarray(512,) } }
_ledger: dict[str, dict] = {}
_ledger_lock = asyncio.Lock()
_last_db_mtime: float = 0.0


# ── Pydantic Schemas ───────────────────────────────────────────────────────────
class AuthenticateRequest(BaseModel):
    """Payload sent by optipulse_engine.py for payment authentication."""
    query_vector: list[float] = Field(..., min_length=500, max_length=512,
                                      description="512D (or 500D legacy) L2-normalized biometric query vector")
    transaction_amount: float = Field(default=TRANSACTION_AMOUNT, gt=0.0,
                                      description="Amount in INR to debit")
    terminal_id: str          = Field(default="TERM-001", max_length=64)
    timestamp: str            = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("query_vector")
    @classmethod
    def validate_unit_vector(cls, v: list[float]) -> list[float]:
        arr = np.array(v, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if not (0.90 <= norm <= 1.10):
            raise ValueError(
                f"Query vector must be L2-normalized (got norm={norm:.4f}). "
                "Re-normalize before sending."
            )
        return v


class AuthenticateResponse(BaseModel):
    """Signed JSON response returned to the POS terminal."""
    status: str                    # "AUTHORIZED" | "DECLINED" | "REJECTED"
    reason: str
    user_id: Optional[str]         = None
    display_name: Optional[str]    = None
    similarity_score: Optional[float] = None
    remaining_balance: Optional[float] = None
    transaction_id: str
    terminal_id: str
    server_timestamp: str
    hmac_signature: str


# ── Helper Functions ───────────────────────────────────────────────────────────
def _align_vector(v: np.ndarray, target_dim: int = 512) -> np.ndarray:
    """Ensure vector is exactly target_dim (pad or truncate) and unit-normalized."""
    if len(v) == target_dim:
        norm = np.linalg.norm(v)
        return (v / norm if norm > 1e-9 else v).astype(np.float32)
    if len(v) < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:len(v)] = v
        norm = np.linalg.norm(padded)
        return (padded / norm if norm > 1e-9 else padded).astype(np.float32)
    truncated = v[:target_dim].astype(np.float32)
    norm = np.linalg.norm(truncated)
    return (truncated / norm if norm > 1e-9 else truncated).astype(np.float32)


def _ensure_ledger_synced():
    """Hot-sync in-memory ledger if enrolled_users.json was modified on disk."""
    global _last_db_mtime, _ledger
    if not ENROLLED_DB_PATH.exists():
        return
    try:
        current_mtime = ENROLLED_DB_PATH.stat().st_mtime
        if current_mtime > _last_db_mtime or len(_ledger) == 0:
            _load_ledger()
            _last_db_mtime = current_mtime
    except Exception as e:
        logger.warning(f"Failed to check ledger mtime: {e}")


def _cosine_similarity_batch(query: np.ndarray, enrolled_vecs: np.ndarray) -> np.ndarray:
    """
    O(N) vectorized cosine similarity: since all vectors are L2-normalized,
    cosine similarity = dot product.  Returns similarity scores for all N users.

    Args:
        query:         (512,) float32 L2-normalized query vector.
        enrolled_vecs: (N, 512) float32 matrix of enrolled vectors.
    Returns:
        (N,) float32 similarity scores in range [-1, 1].
    """
    return enrolled_vecs @ query  # (N, 512) · (512,) → (N,)


def _generate_txn_id() -> str:
    """Generate a time-based unique transaction ID."""
    ts_ns = time.time_ns()
    return f"TXN-{ts_ns:016X}"


def _sign_response(payload: str) -> str:
    """Compute HMAC-SHA256 signature over response payload."""
    return hmac.new(SECRET_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _load_ledger() -> None:
    """Load enrolled users from disk and build the in-memory ledger with numpy vectors."""
    global _ledger
    if not ENROLLED_DB_PATH.exists():
        logger.warning("enrolled_users.json not found — running with empty ledger. "
                        "Run enroll_kyc.py first.")
        _ledger = {}
        return

    with open(ENROLLED_DB_PATH, "r") as f:
        db = json.load(f)

    loaded = 0
    for uid, profile in db.items():
        vec_path = profile.get("vector_path", "")
        if not vec_path or not Path(vec_path).exists():
            logger.warning(f"Vector file missing for {uid}: {vec_path} — skipping.")
            continue
        vec = np.load(vec_path).astype(np.float32)
        vec = _align_vector(vec, 512)
        _ledger[uid] = {**profile, "vector": vec}
        loaded += 1

    logger.info(f"Ledger loaded: {loaded}/{len(db)} users active.")


async def _persist_ledger() -> None:
    """Persist the current balance state back to enrolled_users.json (balance only)."""
    if not ENROLLED_DB_PATH.exists():
        return
    with open(ENROLLED_DB_PATH, "r") as f:
        db = json.load(f)
    for uid, entry in _ledger.items():
        if uid in db:
            db[uid]["balance"] = entry["balance"]
    with open(ENROLLED_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def _log_transaction(txn_id: str, status: str, user_id: Optional[str],
                     score: float, amount: float, terminal_id: str) -> None:
    """Append a transaction record to the audit log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "txn_id":     txn_id,
        "status":     status,
        "user_id":    user_id,
        "score":      round(score, 6),
        "amount":     amount,
        "terminal":   terminal_id,
        "ts":         datetime.now(timezone.utc).isoformat(),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── FastAPI Application ────────────────────────────────────────────────────────
app = FastAPI(
    title="OptiPulse-Pay Gateway",
    description="Autonomous edge facial micro-payment authentication API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Load ledger on server startup."""
    _load_ledger()
    logger.info("OptiPulse-Pay Gateway — READY on http://127.0.0.1:8000")


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe — returns server status and enrolled user count."""
    return {
        "status": "healthy",
        "enrolled_users": len(_ledger),
        "server_time": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.post(
    "/api/v1/verify_identity",
    response_model=AuthenticateResponse,
    tags=["Authentication"],
    summary="Pre-payment biometric verification without debiting funds",
)
async def verify_identity(req: AuthenticateRequest) -> AuthenticateResponse:
    """
    Verify biometric match against enrolled KYC database without debiting wallet.
    Returns matched user profile, balance, and similarity score for user confirmation.
    """
    _ensure_ledger_synced()
    txn_id = _generate_txn_id()

    async with _ledger_lock:
        if not _ledger:
            resp_payload = f"{txn_id}|REJECTED|NO_ENROLLED_USERS"
            logger.warning(f"[{txn_id}] Verification failed: no users in ledger")
            return AuthenticateResponse(
                status="REJECTED",
                reason="NO_ENROLLED_USERS",
                transaction_id=txn_id,
                terminal_id=req.terminal_id,
                server_timestamp=datetime.now(timezone.utc).isoformat(),
                hmac_signature=_sign_response(resp_payload),
            )

        query_vec = _align_vector(np.array(req.query_vector, dtype=np.float32), 512)
        user_ids = list(_ledger.keys())
        enrolled_matrix = np.stack([_ledger[uid]["vector"] for uid in user_ids], axis=0)

        similarities = _cosine_similarity_batch(query_vec, enrolled_matrix)
        best_idx  = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_uid  = user_ids[best_idx]
        user = _ledger[best_uid]

        logger.info(f"[{txn_id}] Verification: best={best_uid} ({user['display_name']}) | score={best_score:.4f}")

        if best_score < COSINE_THRESHOLD:
            logger.warning(f"[{txn_id}] Mismatch: best_score={best_score:.4f} < {COSINE_THRESHOLD}")
            _log_transaction(txn_id, "DECLINED", best_uid, best_score, req.transaction_amount, req.terminal_id)
            resp_payload = f"{txn_id}|DECLINED|BIOMETRIC_MISMATCH|{best_score:.6f}"
            return AuthenticateResponse(
                status="DECLINED",
                reason="BIOMETRIC_MISMATCH",
                similarity_score=round(best_score, 4),
                transaction_id=txn_id,
                terminal_id=req.terminal_id,
                server_timestamp=datetime.now(timezone.utc).isoformat(),
                hmac_signature=_sign_response(resp_payload),
            )

        if user["balance"] < req.transaction_amount:
            resp_payload = f"{txn_id}|DECLINED|INSUFFICIENT_FUNDS|{best_uid}"
            return AuthenticateResponse(
                status="DECLINED",
                reason="INSUFFICIENT_FUNDS",
                user_id=best_uid,
                display_name=user["display_name"],
                similarity_score=round(best_score, 4),
                remaining_balance=round(user["balance"], 2),
                transaction_id=txn_id,
                terminal_id=req.terminal_id,
                server_timestamp=datetime.now(timezone.utc).isoformat(),
                hmac_signature=_sign_response(resp_payload),
            )

        resp_payload = f"{txn_id}|VERIFIED|{best_uid}|{user['balance']:.2f}"
        return AuthenticateResponse(
            status="VERIFIED",
            reason="IDENTITY_VERIFIED",
            user_id=best_uid,
            display_name=user["display_name"],
            similarity_score=round(best_score, 4),
            remaining_balance=round(user["balance"], 2),
            transaction_id=txn_id,
            terminal_id=req.terminal_id,
            server_timestamp=datetime.now(timezone.utc).isoformat(),
            hmac_signature=_sign_response(resp_payload),
        )


@app.post(
    "/api/v1/authenticate_payment",
    response_model=AuthenticateResponse,
    tags=["Authentication"],
    summary="Biometric payment authentication with 1:N cosine matching",
)
async def authenticate_payment(req: AuthenticateRequest) -> AuthenticateResponse:
    """
    Authenticate a payment using a 512D biometric query vector.
    """
    _ensure_ledger_synced()
    t0 = time.perf_counter()
    txn_id = _generate_txn_id()

    async with _ledger_lock:
        if not _ledger:
            resp_payload = f"{txn_id}|REJECTED|NO_ENROLLED_USERS"
            return AuthenticateResponse(
                status="REJECTED",
                reason="NO_ENROLLED_USERS",
                transaction_id=txn_id,
                terminal_id=req.terminal_id,
                server_timestamp=datetime.now(timezone.utc).isoformat(),
                hmac_signature=_sign_response(resp_payload),
            )

        query_vec = _align_vector(np.array(req.query_vector, dtype=np.float32), 512)

        # Build enrolled matrix (N, 512)
        user_ids = list(_ledger.keys())
        enrolled_matrix = np.stack(
            [_ledger[uid]["vector"] for uid in user_ids], axis=0
        )  # (N, 512)

        # O(N) cosine similarities
        similarities = _cosine_similarity_batch(query_vec, enrolled_matrix)  # (N,)
        best_idx  = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_uid  = user_ids[best_idx]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[{txn_id}] Match: {best_uid} | score={best_score:.4f} | {elapsed_ms:.1f}ms")

        # Threshold check
        if best_score < COSINE_THRESHOLD:
            _log_transaction(txn_id, "DECLINED", None, best_score, req.transaction_amount, req.terminal_id)
            resp_payload = f"{txn_id}|DECLINED|BIOMETRIC_MISMATCH|{best_score:.6f}"
            return AuthenticateResponse(
                status="DECLINED",
                reason="BIOMETRIC_MISMATCH",
                similarity_score=round(best_score, 4),
                transaction_id=txn_id,
                terminal_id=req.terminal_id,
                server_timestamp=datetime.now(timezone.utc).isoformat(),
                hmac_signature=_sign_response(resp_payload),
            )

        user = _ledger[best_uid]

        # Balance check
        if user["balance"] < req.transaction_amount:
            _log_transaction(txn_id, "DECLINED", best_uid, best_score, req.transaction_amount, req.terminal_id)
            resp_payload = f"{txn_id}|DECLINED|INSUFFICIENT_FUNDS|{best_uid}"
            return AuthenticateResponse(
                status="DECLINED",
                reason="INSUFFICIENT_FUNDS",
                user_id=best_uid,
                display_name=user["display_name"],
                similarity_score=round(best_score, 4),
                remaining_balance=round(user["balance"], 2),
                transaction_id=txn_id,
                terminal_id=req.terminal_id,
                server_timestamp=datetime.now(timezone.utc).isoformat(),
                hmac_signature=_sign_response(resp_payload),
            )

        # Debit transaction
        user["balance"] = round(user["balance"] - req.transaction_amount, 2)
        await _persist_ledger()

        _log_transaction(txn_id, "AUTHORIZED", best_uid, best_score, req.transaction_amount, req.terminal_id)
        resp_payload = f"{txn_id}|AUTHORIZED|{best_uid}|{user['balance']:.2f}"
        logger.info(
            f"[{txn_id}] AUTHORIZED {user['display_name']} | "
            f"₹{req.transaction_amount:.0f} debited | balance=₹{user['balance']:.2f}"
        )
        return AuthenticateResponse(
            status="AUTHORIZED",
            reason="IDENTITY_VERIFIED",
            user_id=best_uid,
            display_name=user["display_name"],
            similarity_score=round(best_score, 4),
            remaining_balance=user["balance"],
            transaction_id=txn_id,
            terminal_id=req.terminal_id,
            server_timestamp=datetime.now(timezone.utc).isoformat(),
            hmac_signature=_sign_response(resp_payload),
        )


@app.get("/api/v1/ledger", tags=["Admin"])
async def get_ledger_summary():
    """Return sanitized ledger summary (no vectors) for admin inspection."""
    async with _ledger_lock:
        summary = [
            {
                "user_id":      uid,
                "display_name": entry["display_name"],
                "balance":      entry["balance"],
                "enrolled_at":  entry.get("enrolled_at", ""),
            }
            for uid, entry in _ledger.items()
        ]
    return {"users": summary, "total": len(summary)}


@app.post("/api/v1/reload_ledger", tags=["Admin"])
async def reload_ledger():
    """Hot-reload the enrolled users database without restarting the server."""
    async with _ledger_lock:
        _load_ledger()
    return {"status": "reloaded", "enrolled_users": len(_ledger)}


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
        workers=1,
    )
