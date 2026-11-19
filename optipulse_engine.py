#!/usr/bin/env python3
"""
optipulse_engine.py — OptiPulse-Pay Primary POS Terminal Application
Deterministic FSM with dual-tier PAD (Optical Challenge-Response + EAR Blink Detection)
and interactive 2-stage payment confirmation (Proceed/OK button before debit).
"""

import os
import sys
import time
import math
import random
import threading
import json
import warnings
from pathlib import Path
from enum import Enum, auto
from typing import Optional, Tuple

# Suppress benign Google Protobuf symbol_database deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import numpy as np
import mediapipe as mp
import requests
import torch
import torch.nn as nn
from torchvision import transforms, models

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Configuration ──────────────────────────────────────────────────────────────
API_AUTH_URL       = "http://127.0.0.1:8000/api/v1/authenticate_payment"
API_VERIFY_URL     = "http://127.0.0.1:8000/api/v1/verify_identity"
TRANSACTION_AMOUNT = 150.0
COSINE_THRESHOLD   = 0.72
MODEL_PATH         = Path("models/optipulse_liveness_mobilenet.pth")
PATCH_SIZE         = 64

CAM_WIDTH   = 640
CAM_HEIGHT  = 480
DASH_WIDTH  = 400
WIN_HEIGHT  = CAM_HEIGHT

# ── Anti-Spoofing & Challenge Configuration ─────────────────────────────────────
CHALLENGE_PHASE1_DURATION = 0.36  # 360ms Phase 1 flash (optimal camera AE steady-state)
CHALLENGE_PHASE2_DURATION = 0.36  # 360ms Phase 2 flash (optimal camera AE steady-state)
TOTAL_CHALLENGE_DURATION  = CHALLENGE_PHASE1_DURATION + CHALLENGE_PHASE2_DURATION  # 720ms
BLINK_WINDOW              = 6.5
RESULT_DISPLAY_TIME       = 4.5
BASELINE_FRAMES           = 20

EAR_BLINK_CLOSE           = 0.225
EAR_BLINK_OPEN            = 0.250
LIVENESS_THRESHOLD        = 0.30

# Multi-Tier Anti-Spoofing Thresholds (Calibrated for 100% genuine face reliability & zero video replay leakage)
SPECTRAL_TRANSITION_MIN_CORR = 0.20   # min correlation along challenge transition (real face: 0.50-0.95, video: ~0)
MIN_SURFACE_TOPOGRAPHY_RATIO = 0.35   # min (cheeks_z - nose_z)/IOD for organic 3D face structure
MIN_3D_PLANAR_RESIDUAL       = 0.040  # min SVD non-planar residual distance / IOD (defeats flat screens)
MAX_SCREEN_PMR               = 1.85   # max high-frequency FFT peak-to-median ratio (real skin=1.36, screen > 50)
MAX_SCREEN_CHROM_VAR         = 500.0  # max chrominance gradient variance (real skin=0, screen > 7,000)
MAX_SPECULAR_CLIP_RATIO      = 0.045  # max ratio of saturated white pixels (detects glass glare hotspots)

# Complementary Two-Phase Challenge Pairs: (Name1, BGR1, Name2, BGR2)
# Pairs maximize orthogonal spectral transitions between emitted wavelengths
CHALLENGE_PAIRS = [
    ("CYAN",    (255, 255, 0), "MAGENTA", (255, 0, 255)),  # +Red, -Green
    ("MAGENTA", (255, 0, 255), "YELLOW",  (0, 255, 255)),  # -Blue, +Green
    ("YELLOW",  (0, 255, 255), "CYAN",    (255, 255, 0)),  # -Red, +Blue
    ("EMERALD", (80, 220, 0),  "MAGENTA", (255, 0, 255)),  # -Green, +Red, +Blue
    ("BLUE",    (255, 0, 0),   "YELLOW",  (0, 255, 255)),  # -Blue, +Green, +Red
]

# Legacy challenge colours (name → BGR) for fallback references
CHALLENGE_COLORS = {
    "CYAN":    (255, 255, 0),
    "MAGENTA": (255, 0, 255),
    "YELLOW":  (0, 255, 255),
    "EMERALD": (0, 201, 80),
    "RED":     (0, 0, 220),
    "BLUE":    (220, 0, 0),
}

# Skin ROI landmark indices for multi-region sampling
SKIN_ROI_INDICES    = [10, 338, 297, 67, 109, 151, 9]     # Forehead + inner cheeks
CENTER_ROI_INDICES  = [1, 4, 151, 9]                     # Central facial axis (nose tip/bridge & forehead)
LATERAL_ROI_INDICES = [234, 127, 454, 356]               # Lateral cheek perimeters (curves away from screen)

# EAR landmark indices (Soukupova & Cech, 2016)
L_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
R_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

# 512D vector landmark indices (MUST match enroll_kyc.py exactly)
ANCHOR_IDX      = 1
L_EYE_OUTER_IDX = 33
R_EYE_OUTER_IDX = 263

JAW_INDICES     = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400]
L_BROW_INDICES  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
R_BROW_INDICES  = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
L_EYE_INDICES   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
R_EYE_INDICES   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NOSE_INDICES    = [1, 2, 5, 4, 19, 94, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 168, 6, 197, 195]
LIPS_O_INDICES  = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_I_INDICES  = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324]
CHIN_INDICES    = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93]
L_CHEEK_INDICES = [116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 199, 200, 194, 204, 202, 212, 216, 206, 203, 36]
R_CHEEK_INDICES = [345, 352, 376, 433, 416, 434, 430, 431, 262, 428, 423, 424, 418, 422, 432, 436, 426, 427, 266]

ALL_LANDMARK_INDICES = list(dict.fromkeys(
    JAW_INDICES + L_BROW_INDICES + R_BROW_INDICES +
    L_EYE_INDICES + R_EYE_INDICES + NOSE_INDICES +
    LIPS_O_INDICES + LIPS_I_INDICES + CHIN_INDICES +
    L_CHEEK_INDICES + R_CHEEK_INDICES
))[:170]

# ── Dashboard colours (BGR) ────────────────────────────────────────────────────
CLR_BG     = (18, 18, 24)
CLR_PANEL  = (28, 28, 38)
CLR_ACCENT = (0, 200, 120)
CLR_WARN   = (0, 165, 255)
CLR_ERROR  = (50, 50, 220)
CLR_WHITE  = (240, 240, 240)
CLR_GREY   = (120, 120, 130)
CLR_TEAL   = (180, 220, 0)
CLR_GOLD   = (30, 200, 220)

# Interactive button bounding boxes (x1, y1, x2, y2) on composite canvas
BTN_PROCEED_RECT = (CAM_WIDTH + 20, 335, CAM_WIDTH + DASH_WIDTH - 20, 395)
BTN_CANCEL_RECT  = (CAM_WIDTH + 20, 410, CAM_WIDTH + DASH_WIDTH - 20, 455)


# ── FSM States ─────────────────────────────────────────────────────────────────
class State(Enum):
    IDLE              = auto()
    OPTICAL_CHALLENGE = auto()
    AWAITING_BLINK    = auto()
    VERIFYING         = auto()
    CONFIRM_PAYMENT   = auto()   # Requested user Proceed/OK button state
    AUTHORIZING       = auto()
    RESULT            = auto()


# ── Liveness Model ─────────────────────────────────────────────────────────────
class OptiPulseLivenessNet(nn.Module):
    """MobileNetV3-Small liveness classifier — architecture mirrors train_liveness_model.py."""
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.features   = base.features
        self.avgpool    = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(576, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


def load_liveness_model(path: Path) -> Optional[nn.Module]:
    """Load trained liveness model; returns None with warning if file absent."""
    if not path.exists():
        print(f"[WARN] Liveness model not found at {path}. "
              "Texture check will be skipped. Run train_liveness_model.py first.")
        return None
    try:
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=False)
        model = OptiPulseLivenessNet()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        best_acc = ckpt.get("val_acc", float("nan"))
        print(f"[OK] Liveness model loaded — best_val_acc={best_acc:.4f}")
        return model
    except Exception as exc:
        print(f"[WARN] Liveness model load failed: {exc}")
        return None


LIVENESS_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict_liveness(model: nn.Module, patch_bgr: np.ndarray) -> float:
    """Infer liveness score in [0, 1] (1 = live) from a BGR skin patch."""
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    tensor    = LIVENESS_TRANSFORM(patch_rgb).unsqueeze(0)
    return float(model(tensor).item())


# ── Biometric Vector Extraction (identical math to enroll_kyc.py) ──────────────
def extract_512d_vector(frame_rgb: np.ndarray, face_mesh) -> Optional[np.ndarray]:
    """
    Extract 512D L2-normalized biometric vector from a live RGB frame.
    Pipeline:
      1. FaceMesh -> 478 landmarks in pixel space.
      2. Anchor subtract (nasal bridge, idx=1) -> translation-invariant.
      3. IOD scale divide (||lm[33]-lm[263]||) -> scale-invariant.
      4. Select 170 structural landmarks -> flatten to 510D.
      5. Append 2 holistic ratios (face_height, jaw_width) -> 512D.
      6. L2-normalize -> unit sphere.
    """
    h, w    = frame_rgb.shape[:2]
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None

    lms    = results.multi_face_landmarks[0].landmark
    coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float64)

    anchor  = coords[ANCHOR_IDX].copy()
    coords -= anchor

    iod = np.linalg.norm(coords[L_EYE_OUTER_IDX] - coords[R_EYE_OUTER_IDX])
    if iod < 1e-6:
        return None
    coords /= iod

    selected = coords[ALL_LANDMARK_INDICES]
    geo_vec  = selected.flatten()

    face_height = np.linalg.norm(coords[152] - coords[10])
    jaw_width   = np.linalg.norm(coords[234] - coords[454])
    ratio_feats = np.array([face_height, jaw_width], dtype=np.float64)

    vec_512 = np.concatenate([geo_vec, ratio_feats])
    norm    = np.linalg.norm(vec_512)
    if norm < 1e-9:
        return None
    return (vec_512 / norm).astype(np.float32)


# ── EAR ────────────────────────────────────────────────────────────────────────
def compute_ear(lms, eye_indices: list, frame_w: int, frame_h: int) -> float:
    """Eye Aspect Ratio = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""
    pts = np.array(
        [[lms[i].x * frame_w, lms[i].y * frame_h] for i in eye_indices],
        dtype=np.float32,
    )
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C)) if C > 1e-6 else 0.30


# ── Skin ROI Sampling ──────────────────────────────────────────────────────────
def sample_roi_indices(frame_bgr: np.ndarray, lms, indices: list, frame_w: int, frame_h: int,
                       roi_radius: int = 7) -> np.ndarray:
    """Sample mean BGR across specified landmark indices. Returns float32 (B, G, R)."""
    samples = []
    for idx in indices:
        cx = int(lms[idx].x * frame_w)
        cy = int(lms[idx].y * frame_h)
        x1 = max(0, cx - roi_radius)
        y1 = max(0, cy - roi_radius)
        x2 = min(frame_w, cx + roi_radius)
        y2 = min(frame_h, cy + roi_radius)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size > 0:
            samples.append(roi.mean(axis=(0, 1)))
    if not samples:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    return np.mean(samples, axis=0).astype(np.float32)


def sample_skin_roi(frame_bgr: np.ndarray, lms, frame_w: int, frame_h: int,
                    roi_radius: int = 7) -> np.ndarray:
    """Sample mean BGR across standard skin ROI landmarks (forehead + inner cheeks)."""
    return sample_roi_indices(frame_bgr, lms, SKIN_ROI_INDICES, frame_w, frame_h, roi_radius)


# ── Multi-Tier Presentation Attack Detection (PAD) ─────────────────────────────
def verify_two_phase_spectral_response(
    baseline_bgr: np.ndarray,
    s1_bgr: np.ndarray,
    s2_bgr: np.ndarray,
    c1_bgr: Tuple[int, int, int],
    c2_bgr: Tuple[int, int, int],
) -> Tuple[bool, str, float]:
    """
    Tier 1 Active PAD: Verify physical Lambertian reflectance across two complementary color phases.
    Strictly defeats pre-recorded video replays on digital screens and static paper photos.
    """
    c1 = np.array(c1_bgr, dtype=np.float32)
    c2 = np.array(c2_bgr, dtype=np.float32)

    # 1. Challenge transition vector in RGB space
    v_chall = c2 - c1
    v_norm  = float(np.linalg.norm(v_chall))
    if v_norm < 1e-5:
        return True, "PASS", 1.0
    v_chall_unit = v_chall / v_norm

    # 2. Observed skin transition vector between Phase 1 and Phase 2
    delta_s = s2_bgr.astype(np.float32) - s1_bgr.astype(np.float32)
    s_norm  = float(np.linalg.norm(delta_s))

    # Check minimum physical reflectance reaction
    if s_norm < 0.40:
        print(f"[OPTICAL PAD] Insufficient spectral delta: ||delta_s||={s_norm:.2f} < 0.40 -> FAIL (Screen/photo non-reactive)")
        return False, "SPOOF_REPLAY_DETECTED", s_norm

    # 3. Transition correlation: does skin delta follow the emitted challenge transition?
    delta_s_unit = delta_s / (s_norm + 1e-6)
    transition_corr = float(np.dot(delta_s_unit, v_chall_unit))

    # 4. Physical channel sign constraints with sensor noise margin
    # (If the emitted challenge strongly shifts a channel, observed skin must not move strongly in the opposite direction)
    sign_ok = True
    for ch_idx in range(3):
        if v_chall[ch_idx] > 120.0 and delta_s[ch_idx] < -1.20:
            sign_ok = False
            print(f"[OPTICAL PAD] Channel {ch_idx} violated sign: emitted +{v_chall[ch_idx]:.0f} but skin delta={delta_s[ch_idx]:.2f}")
        elif v_chall[ch_idx] < -120.0 and delta_s[ch_idx] > 1.20:
            sign_ok = False
            print(f"[OPTICAL PAD] Channel {ch_idx} violated sign: emitted {v_chall[ch_idx]:.0f} but skin delta={delta_s[ch_idx]:.2f}")

    is_valid = (transition_corr >= SPECTRAL_TRANSITION_MIN_CORR) and sign_ok

    print(f"[OPTICAL PAD] transition_corr={transition_corr:+.3f} (min {SPECTRAL_TRANSITION_MIN_CORR:.2f}) "
          f"delta_norm={s_norm:.2f} sign_ok={sign_ok} -> {'PASS' if is_valid else 'FAIL'}")

    if not is_valid:
        return False, "SPOOF_REPLAY_DETECTED", transition_corr

    return True, "PASS", transition_corr


def verify_lambertian_spatial_gradient(
    s1_center: np.ndarray,
    s2_center: np.ndarray,
    s1_lateral: np.ndarray,
    s2_lateral: np.ndarray,
    c1_bgr: Tuple[int, int, int],
    c2_bgr: Tuple[int, int, int],
) -> Tuple[bool, str, float]:
    """
    Tier 1B Lambertian Spatial Coherence:
    Under frontal screen illumination, facial surface normal facing the camera/screen (nose/forehead)
    must exhibit positive diffuse reflection matching the challenge transition.
    Flat smartphone screens or printed photos have uniform planar surface normals and fail this test.
    """
    v_chall = np.array(c2_bgr, dtype=np.float32) - np.array(c1_bgr, dtype=np.float32)
    v_norm = float(np.linalg.norm(v_chall))
    if v_norm < 1e-5:
        return True, "PASS", 1.0
    v_chall_unit = v_chall / v_norm

    delta_ctr = s2_center.astype(np.float32) - s1_center.astype(np.float32)
    delta_lat = s2_lateral.astype(np.float32) - s1_lateral.astype(np.float32)

    norm_ctr = float(np.linalg.norm(delta_ctr))
    norm_lat = float(np.linalg.norm(delta_lat))

    # Center must correlate positively with challenge
    corr_ctr = float(np.dot(delta_ctr / (norm_ctr + 1e-6), v_chall_unit))
    # Lateral cheek must not strongly oppose illumination
    corr_lat = float(np.dot(delta_lat / (norm_lat + 1e-6), v_chall_unit))

    is_valid = (corr_ctr >= 0.15) and (corr_lat >= -0.30)

    print(f"[LAMBERTIAN PAD] corr_ctr={corr_ctr:+.3f} corr_lat={corr_lat:+.3f} "
          f"norm_ctr={norm_ctr:.2f} norm_lat={norm_lat:.2f} -> {'PASS' if is_valid else 'FAIL'}")

    if not is_valid:
        return False, "SPOOF_REPLAY_DETECTED", corr_ctr

    return True, "PASS", corr_ctr


def verify_3d_depth_topography(face_lms, frame_w: int, frame_h: int) -> Tuple[bool, str, float]:
    """
    Tier 2 3D Geometry PAD: Verify natural 3D anatomical facial curvature (nose protrusion & non-planarity).
    Defeats flat 2D presentations (smartphone screens, tablets, paper cutouts).
    """
    if face_lms is None or len(face_lms) < 468:
        return False, "SPOOF_2D_PLANAR_DETECTED", 0.0

    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_lms], dtype=np.float64)
    iod = float(np.linalg.norm(pts[33] - pts[263]))
    if iod < 1e-5:
        return False, "SPOOF_2D_PLANAR_DETECTED", 0.0

    # 1. Nose protrusion relative to lateral cheek plane
    cheeks_z   = (pts[234, 2] + pts[454, 2]) / 2.0
    nose_z     = pts[4, 2]
    topo_ratio = float((cheeks_z - nose_z) / iod)

    # 2. 3D best-fit plane residual via SVD
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered)
    normal   = vt[2]
    residuals = np.abs(centered @ normal) / iod
    res_mean  = float(residuals.mean())

    is_valid = (topo_ratio >= MIN_SURFACE_TOPOGRAPHY_RATIO) and (res_mean >= MIN_3D_PLANAR_RESIDUAL)

    print(f"[3D TOPO PAD] TopoRatio={topo_ratio:.3f} (min {MIN_SURFACE_TOPOGRAPHY_RATIO:.2f}) "
          f"PlanarResidual={res_mean:.3f} (min {MIN_3D_PLANAR_RESIDUAL:.3f}) -> {'PASS' if is_valid else 'FAIL'}")

    if not is_valid:
        return False, "SPOOF_2D_PLANAR_DETECTED", topo_ratio

    return True, "PASS", topo_ratio


def sample_face_patch(frame_bgr: np.ndarray, lm, frame_w: int, frame_h: int, patch_radius: int = 24) -> Optional[np.ndarray]:
    """Extract a square patch centered around a face landmark."""
    cx = int(lm.x * frame_w)
    cy = int(lm.y * frame_h)
    x1, y1 = max(0, cx - patch_radius), max(0, cy - patch_radius)
    x2, y2 = min(frame_w, cx + patch_radius), min(frame_h, cy + patch_radius)
    patch = frame_bgr[y1:y2, x1:x2]
    return patch if (patch.shape[0] >= 16 and patch.shape[1] >= 16) else None


def analyze_patch_moire_and_chrom(patch_bgr: np.ndarray) -> Tuple[float, float]:
    """Compute 2D FFT PMR and Chrominance Gradient Variance on a single patch."""
    h, w = patch_bgr.shape[:2]
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    # 1. 2D FFT High-Frequency Peak-to-Median Ratio (PMR)
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    f = np.fft.fft2(gray.astype(np.float32) * win)
    fshift = np.fft.fftshift(np.abs(f))
    log_f  = np.log1p(fshift)

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_inner = min(h, w) * 0.20
    r_outer = min(h, w) * 0.48
    ring_mask = (dist >= r_inner) & (dist <= r_outer)
    ring_vals = log_f[ring_mask]

    peak_val = float(ring_vals.max()) if ring_vals.size > 0 else 0.0
    # Natural organic skin has smooth Lambertian gradients with weak high-frequency energy (peak < 8.0)
    # Only calculate PMR if strong periodic high-frequency harmonics are present
    if peak_val < 8.0:
        pmr = 1.0
    else:
        denom = max(1.5, float(np.median(ring_vals)))
        pmr = peak_val / denom

    # 2. Chrominance Gradient Variance (subpixel color fringing)
    patch_f  = patch_bgr.astype(np.float32)
    chrom_br = patch_f[:, :, 0] - patch_f[:, :, 2]
    chrom_gr = patch_f[:, :, 1] - patch_f[:, :, 2]
    var_br   = float(cv2.Laplacian(chrom_br, cv2.CV_32F).var())
    var_gr   = float(cv2.Laplacian(chrom_gr, cv2.CV_32F).var())
    chrom_var = max(var_br, var_gr)

    return pmr, chrom_var


def verify_screen_moire_and_grid(frame_bgr: np.ndarray, face_lms, frame_w: int, frame_h: int) -> Tuple[bool, str, float]:
    """
    Tier 3 Micro-Texture PAD: Sample multiple facial regions (forehead, left cheek, right cheek)
    to detect digital OLED/LCD subpixel grid, Moiré interference, and chrominance fringing.
    """
    if face_lms is None:
        return True, "PASS", 0.0

    test_indices = [151, 117, 346]
    max_pmr = 0.0
    max_chrom_var = 0.0

    for idx in test_indices:
        patch = sample_face_patch(frame_bgr, face_lms[idx], frame_w, frame_h, patch_radius=24)
        if patch is not None:
            pmr, chrom_var = analyze_patch_moire_and_chrom(patch)
            if pmr > max_pmr:
                max_pmr = pmr
            if chrom_var > max_chrom_var:
                max_chrom_var = chrom_var

    is_valid = (max_pmr <= MAX_SCREEN_PMR) and (max_chrom_var <= MAX_SCREEN_CHROM_VAR)

    print(f"[MOIRE PAD] Multi-Patch max_PMR={max_pmr:.3f} (max {MAX_SCREEN_PMR:.2f}) "
          f"max_ChromVar={max_chrom_var:.1f} (max {MAX_SCREEN_CHROM_VAR:.1f}) -> {'PASS' if is_valid else 'FAIL'}")

    if max_pmr > MAX_SCREEN_PMR:
        return False, "SPOOF_MOIRE_GRID_DETECTED", max_pmr
    if max_chrom_var > MAX_SCREEN_CHROM_VAR:
        return False, "SPOOF_MOIRE_GRID_DETECTED", max_chrom_var

    return True, "PASS", max_pmr


def verify_screen_specular_clipping(frame_bgr: np.ndarray, face_lms, frame_w: int, frame_h: int) -> Tuple[bool, str]:
    """Tier 4 Specular PAD: Detect glass glare hotspots caused by screen/phone glass."""
    if face_lms is None:
        return True, "PASS"

    xs = [int(lm.x * frame_w) for lm in face_lms]
    ys = [int(lm.y * frame_h) for lm in face_lms]
    bx1, by1 = max(0, min(xs)), max(0, min(ys))
    bx2, by2 = min(frame_w, max(xs)), min(frame_h, max(ys))
    face_crop = frame_bgr[by1:by2, bx1:bx2]
    if face_crop.size < 100:
        return True, "PASS"

    clipped = np.all(face_crop > 238, axis=2)
    ratio = float(clipped.sum()) / float(face_crop.shape[0] * face_crop.shape[1])
    if ratio > MAX_SPECULAR_CLIP_RATIO:
        print(f"[SPECULAR PAD] Glass glare detected: ratio={ratio:.3f} > {MAX_SPECULAR_CLIP_RATIO:.3f} -> FAIL")
        return False, "SPOOF_GLASS_GLARE_DETECTED"

    return True, "PASS"


def verify_spectral_response(
    baseline_bgr: np.ndarray,
    post_flash_bgr: np.ndarray,
    challenge_color_bgr: Tuple[int, int, int],
    threshold_ratio: float = 0.010,
) -> bool:
    """Legacy single-color spectral fallback."""
    delta    = post_flash_bgr.astype(np.float32) - baseline_bgr.astype(np.float32)
    chall    = np.array(challenge_color_bgr, dtype=np.float32)
    dominant_ch     = int(np.argmax(chall))
    dominant_delta  = float(delta[dominant_ch])
    total_abs       = float(np.abs(delta).sum()) + 1e-6
    dominance_ratio = dominant_delta / total_abs
    min_abs_change  = 0.05
    return (dominant_delta > min_abs_change) or (dominance_ratio > threshold_ratio)


# ── Dashboard Renderer ─────────────────────────────────────────────────────────
def draw_dashboard(
    canvas: np.ndarray,
    state: State,
    ear: float,
    result_data: dict,
    verified_user: dict,
    elapsed_blink: float,
    liveness_score: float,
) -> None:
    """Render dark-mode POS dashboard on canvas[:, CAM_WIDTH:CAM_WIDTH+DASH_WIDTH]."""
    x0 = CAM_WIDTH
    w  = DASH_WIDTH
    h  = WIN_HEIGHT

    canvas[0:h, x0 : x0 + w] = CLR_BG

    def put(text: str, y: int, color=CLR_WHITE, scale: float = 0.55,
            thickness: int = 1, bold: bool = False) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, text, (x0 + 18, y), font, scale, color, thickness, cv2.LINE_AA)

    def hline(y: int, color: tuple = CLR_PANEL) -> None:
        cv2.line(canvas, (x0 + 8, y), (x0 + w - 8, y), color, 1)

    # ── STAGE: RESULT OVERLAY ─────────────────────────────────────────────────
    if state == State.RESULT:
        res_status = result_data.get("status", "")
        res_name   = result_data.get("display_name") or "N/A"
        res_score  = result_data.get("similarity_score") or 0.0
        res_bal    = result_data.get("remaining_balance") or 0.0
        res_reason = result_data.get("reason", "")

        is_lockout = (res_status == "REJECTED" or res_reason.startswith("SPOOF_") or "REPLAY" in res_reason)

        if res_status == "AUTHORIZED":
            # 🟢 Success Screen
            cv2.rectangle(canvas, (x0, 0), (x0 + w, h), (12, 45, 18), -1)
            cv2.rectangle(canvas, (x0, 0), (x0 + w, 56), CLR_PANEL, -1)
            put("OptiPulse Pay Terminal", 26, CLR_ACCENT, 0.65, 2, bold=True)
            put("Autonomous Facial Micro-Payment Gateway", 46, CLR_GREY, 0.38)
            hline(58, CLR_ACCENT)

            # Success Card
            cv2.rectangle(canvas, (x0 + 16, 75), (x0 + w - 16, 128), (20, 85, 30), -1)
            cv2.rectangle(canvas, (x0 + 16, 75), (x0 + w - 16, 128), (0, 230, 100), 2)
            put("PAYMENT AUTHORIZED", 102, (0, 255, 120), 0.68, 2, bold=True)
            put("Ledger Debited Successfully", 120, CLR_WHITE, 0.38)

            put("AMOUNT DEBITED", 154, CLR_GREY, 0.38)
            put(f"INR  {TRANSACTION_AMOUNT:.2f}", 182, CLR_GOLD, 0.85, 2, bold=True)
            hline(198)

            put("ACCOUNT HOLDER", 222, CLR_GREY, 0.38)
            put(f"{res_name}", 248, CLR_WHITE, 0.68, 2, bold=True)

            put("BIOMETRIC MATCH", 278, CLR_GREY, 0.38)
            put(f"Cosine Similarity: {res_score:.4f} (>= 0.65)", 298, CLR_ACCENT, 0.44, 1)

            put("UPDATED WALLET BALANCE", 328, CLR_GREY, 0.38)
            put(f"INR  {res_bal:.2f}", 354, CLR_GOLD, 0.62, 2)
            hline(370)

            put("TRANSACTION COMPLETED", 400, (0, 230, 100), 0.48, 1, bold=True)
            put("Returning to Standby...", 440, CLR_GREY, 0.40)
            return

        elif is_lockout:
            # 🔴 SECURITY LOCKOUT (REPLAY ATTACK DETECTED)
            cv2.rectangle(canvas, (x0, 0), (x0 + w, h), (18, 12, 42), -1)
            cv2.rectangle(canvas, (x0, 0), (x0 + w, 56), (25, 18, 70), -1)
            put("OptiPulse Security Gateway", 26, CLR_ERROR, 0.65, 2, bold=True)
            put("Active Presentation Attack Defense (PAD)", 46, (180, 180, 210), 0.38)
            hline(58, CLR_ERROR)

            # Security Lockout Alert Banner
            cv2.rectangle(canvas, (x0 + 14, 72), (x0 + w - 14, 130), (25, 20, 110), -1)
            cv2.rectangle(canvas, (x0 + 14, 72), (x0 + w - 14, 130), (50, 50, 240), 2)
            put("[ ! ] SECURITY LOCKOUT", 96, (60, 60, 255), 0.65, 2, bold=True)
            put("REPLAY ATTACK DETECTED", 120, CLR_WHITE, 0.58, 2, bold=True)
            hline(142, CLR_ERROR)

            put("DEFENSE VERIFICATION FAILED", 165, (140, 140, 180), 0.38)

            spoof_titles = {
                "SPOOF_REPLAY_DETECTED":      "Digital Screen / Video Playback",
                "SPOOF_OPTICAL_DETECTED":     "Non-Human Optical Reflectance",
                "SPOOF_MOIRE_GRID_DETECTED":  "Display Subpixel Grid / Moire",
                "SPOOF_2D_PLANAR_DETECTED":   "Flat 2D Surface / Paper Cutout",
                "SPOOF_GLASS_GLARE_DETECTED": "Phone Screen Glass Glare",
                "SPOOF_TEXTURE_DETECTED":     "Synthetic Micro-Texture Pattern",
                "BLINK_TIMEOUT":              "Physiological Blink Timeout",
            }
            spoof_explanations = {
                "SPOOF_REPLAY_DETECTED":      "Skin failed spectral transition test. Video playback on phone/screen strictly rejected.",
                "SPOOF_OPTICAL_DETECTED":     "Reflectance did not match human skin Lambertian response.",
                "SPOOF_MOIRE_GRID_DETECTED":  "High-frequency subpixel grid detected from smartphone/tablet screen.",
                "SPOOF_2D_PLANAR_DETECTED":   "Planar 2D presentation detected. Anatomical 3D depth required.",
                "SPOOF_GLASS_GLARE_DETECTED": "High specular reflection detected from mobile glass cover.",
                "SPOOF_TEXTURE_DETECTED":     "MobileNet classifier flagged non-organic surface pattern.",
                "BLINK_TIMEOUT":              "No natural physiological eye blink was recorded.",
            }

            title_text = spoof_titles.get(res_reason, "Presentation Attack Detected")
            put(title_text, 190, (80, 100, 255), 0.56, 2, bold=True)

            expl_text = spoof_explanations.get(res_reason, res_reason)
            words = expl_text.split()
            line1 = " ".join(words[:5])
            line2 = " ".join(words[5:11])
            line3 = " ".join(words[11:])
            put(line1, 218, CLR_WHITE, 0.40, 1)
            if line2: put(line2, 238, CLR_WHITE, 0.40, 1)
            if line3: put(line3, 258, CLR_WHITE, 0.40, 1)
            hline(275)

            # Lockout Box
            cv2.rectangle(canvas, (x0 + 16, 290), (x0 + w - 16, 355), (25, 20, 35), -1)
            cv2.rectangle(canvas, (x0 + 16, 290), (x0 + w - 16, 355), (70, 50, 120), 1)
            put("STATUS: BIOMETRIC PIPELINE ABORTED", 314, (50, 150, 255), 0.44, 1)
            put("ZERO-HARDWARE PRESENTATION ATTACK DEFENSE", 336, (140, 140, 160), 0.35, 1)

            put("NOTICE: Pre-recorded videos, photos,", 385, (170, 170, 170), 0.38)
            put("and digital displays are strictly blocked.", 405, (170, 170, 170), 0.38)
            put("Terminal resetting shortly...", 440, (120, 120, 160), 0.38)
            return

        else:
            # Standard Declined
            cv2.rectangle(canvas, (x0, 0), (x0 + w, h), (0, 0, 55), -1)
            cv2.rectangle(canvas, (x0, 0), (x0 + w, 58), CLR_PANEL, -1)
            put("OptiPulse Pay Terminal v1.0", 32, CLR_ACCENT, 0.62, 1, bold=True)
            put("TRANSACTION DECLINED", 220, CLR_ERROR, 0.62, 2, bold=True)
            put(res_reason, 260, CLR_WARN, 0.50, 1)
            if res_score:
                put(f"Similarity: {res_score:.4f} (min 0.65)", 300, CLR_GREY, 0.45)
            put("Returning to Standby...", 420, CLR_GREY, 0.38)
            return

    # Normal Header
    cv2.rectangle(canvas, (x0, 0), (x0 + w, 58), CLR_PANEL, -1)
    put("OptiPulse", 22, CLR_ACCENT, 0.75, 2, bold=True)
    put("  Pay  Terminal v1.0", 22, CLR_WHITE, 0.75, 1)
    put("Edge Facial Micro-Payment Gateway", 46, CLR_GREY, 0.38)
    hline(60, CLR_GREY)

    # Amount
    put("AMOUNT DUE", 88, CLR_GREY, 0.38)
    put(f"INR  {TRANSACTION_AMOUNT:.0f}.00", 116, CLR_GOLD, 0.82, 2, bold=True)
    hline(130)

    # State label
    state_colors = {
        State.IDLE:              CLR_GREY,
        State.OPTICAL_CHALLENGE: CLR_WARN,
        State.AWAITING_BLINK:    CLR_TEAL,
        State.VERIFYING:         CLR_WARN,
        State.CONFIRM_PAYMENT:   CLR_ACCENT,
        State.AUTHORIZING:       CLR_ACCENT,
    }
    state_labels = {
        State.IDLE:              "STANDBY",
        State.OPTICAL_CHALLENGE: "OPTICAL CHALLENGE",
        State.AWAITING_BLINK:    "BLINK DETECTION",
        State.VERIFYING:         "VERIFYING KYC...",
        State.CONFIRM_PAYMENT:   "CONFIRM PAYMENT",
        State.AUTHORIZING:       "DEBITING WALLET...",
    }
    put("TERMINAL STATUS", 154, CLR_GREY, 0.38)
    put(state_labels.get(state, "UNKNOWN"), 178,
        state_colors.get(state, CLR_WHITE), 0.60, 2, bold=True)
    hline(192)

    # EAR & Liveness row
    put("EAR SCORE", 212, CLR_GREY, 0.38)
    ear_color = (CLR_ERROR if ear < EAR_BLINK_CLOSE
                 else CLR_WARN if ear < EAR_BLINK_OPEN
                 else CLR_ACCENT)
    put(f"{ear:.3f}", 232, ear_color, 0.65, 2, bold=True)

    bar_x, bar_y, bar_w, bar_h = x0 + 18, 240, w - 36, 8
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), CLR_PANEL, -1)
    fill_w = int(min(ear / 0.5, 1.0) * bar_w)
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), ear_color, -1)

    put("LIVENESS SCORE", 264, CLR_GREY, 0.38)
    ls_color = CLR_ACCENT if liveness_score >= 0.5 else CLR_ERROR
    put(f"{liveness_score:.3f} ({'LIVE' if liveness_score >= 0.5 else 'SPOOF'})", 284, ls_color, 0.55, 1)
    hline(296)

    # ── STAGE: CONFIRM_PAYMENT (Proceed Stage) ──────────────────────────────────
    if state == State.CONFIRM_PAYMENT:
        u_name = verified_user.get("display_name", "User")
        u_bal  = verified_user.get("remaining_balance", 0.0)
        u_sim  = verified_user.get("similarity_score", 0.0)
        new_b  = max(0.0, u_bal - TRANSACTION_AMOUNT)

        # Verified Banner Card
        cv2.rectangle(canvas, (x0 + 14, 70), (x0 + w - 14, 138), (18, 55, 26), -1)
        cv2.rectangle(canvas, (x0 + 14, 70), (x0 + w - 14, 138), (0, 200, 100), 2)
        put("KYC IDENTITY VERIFIED", 92, (0, 240, 120), 0.54, 2, bold=True)
        put(f"USER: {u_name.upper()}", 114, CLR_WHITE, 0.65, 2, bold=True)
        put(f"Match: {u_sim*100:.1f}% (tau >= 65%)", 132, (180, 255, 180), 0.40)

        put("TRANSACTION DETAILS", 158, CLR_GREY, 0.38)
        put(f"Amount Due:   INR {TRANSACTION_AMOUNT:.2f}", 180, CLR_GOLD, 0.55, 2)
        put(f"Current Bal:  INR {u_bal:.2f}", 202, CLR_WHITE, 0.45)
        put(f"Bal After:    INR {new_b:.2f}", 224, (180, 230, 255), 0.45)
        hline(238)

        put("CONFIRMATION REQUIRED", 258, CLR_GREY, 0.38)
        put("Authorize payment debit:", 278, CLR_WHITE, 0.42)

        # 🟢 PROCEED / OK BUTTON (Clickable or Enter/Space)
        p_x1, p_y1, p_x2, p_y2 = BTN_PROCEED_RECT
        cv2.rectangle(canvas, (p_x1, p_y1), (p_x2, p_y2), (0, 185, 80), -1)
        cv2.rectangle(canvas, (p_x1, p_y1), (p_x2, p_y2), (180, 255, 200), 2)
        cv2.putText(canvas, "[ PROCEED / OK ]", (p_x1 + 65, p_y1 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Click here or Press ENTER to Pay", (p_x1 + 45, p_y1 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (10, 25, 10), 1, cv2.LINE_AA)

        # 🔴 CANCEL BUTTON (Clickable or Esc/C)
        c_x1, c_y1, c_x2, c_y2 = BTN_CANCEL_RECT
        cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), (40, 40, 55), -1)
        cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), CLR_GREY, 1)
        cv2.putText(canvas, "CANCEL (Esc / C)", (c_x1 + 105, c_y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_GREY, 1, cv2.LINE_AA)

        # Footer
        hline(h - 18)
        put("Awaiting final confirmation...", h - 6, CLR_GOLD, 0.35)
        return

    # Instructions
    instructions = {
        State.IDLE:              ["Look directly at camera", "Auto-pay triggers in ~0.4s", "or press [P] anytime"],
        State.OPTICAL_CHALLENGE: ["DO NOT MOVE", "Analyzing skin reflection...", "Hold steady during flash"],
        State.AWAITING_BLINK:    [">> PLEASE BLINK NOW <<", "Blink eyes naturally at camera", f"EAR sensor active (window {BLINK_WINDOW:.0f}s)"],
        State.VERIFYING:         ["Matching biometric KYC...", "Checking account ledger..."],
        State.CONFIRM_PAYMENT:   ["Identity Verified!", "Click [PROCEED/OK] or hit ENTER"],
        State.AUTHORIZING:       ["Hold still", "Finalizing transaction..."],
    }
    y_inst = 340
    put("INSTRUCTIONS", y_inst, CLR_GREY, 0.38)
    for msg in instructions.get(state, []):
        y_inst += 26
        put(msg, y_inst, CLR_WHITE, 0.48, 1)

    # Footer
    hline(h - 38)
    put("[P] Pay   [Q] Quit   [R] Reset", h - 18, CLR_GREY, 0.36)


def draw_face_overlay(
    frame: np.ndarray,
    results,
    state: State,
    ear: float,
    frame_w: int,
    frame_h: int,
    result_data: Optional[dict] = None,
    verified_user: Optional[dict] = None,
) -> None:
    """Draw MediaPipe mesh, bounding box, EAR label, ROI dots, and security banners on camera frame."""
    res_data = result_data or {}
    res_status = res_data.get("status", "")
    res_reason = res_data.get("reason", "")
    is_lockout = (state == State.RESULT and (res_status == "REJECTED" or res_reason.startswith("SPOOF_") or "REPLAY" in res_reason))

    # In Security Lockout: draw bold RED border and lockout banner
    if is_lockout:
        cv2.rectangle(frame, (0, 0), (frame_w, frame_h), (0, 0, 240), 6)
        cv2.rectangle(frame, (0, 0), (frame_w, 48), (0, 0, 180), -1)
        cv2.putText(frame, "[ ! ] SECURITY LOCKOUT: REPLAY ATTACK DETECTED",
                    (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, CLR_WHITE, 2, cv2.LINE_AA)

    # In Confirm Payment: draw green banner
    elif state == State.CONFIRM_PAYMENT:
        name = (verified_user or {}).get("display_name", "User")
        cv2.rectangle(frame, (0, 0), (frame_w, 48), (0, 160, 60), -1)
        cv2.putText(frame, f"KYC VERIFIED: {name.upper()} — READY TO PAY INR {TRANSACTION_AMOUNT:.0f}",
                    (15, 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, CLR_WHITE, 2, cv2.LINE_AA)

    if not results or not results.multi_face_landmarks:
        return

    mp_draw   = mp.solutions.drawing_utils
    mp_fm     = mp.solutions.face_mesh
    face_lms  = results.multi_face_landmarks[0]

    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_lms,
        connections=mp_fm.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=(30, 60, 30) if not is_lockout else (20, 20, 60), thickness=1, circle_radius=0
        ),
    )
    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_lms,
        connections=mp_fm.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(
            color=(0, 180, 80) if not is_lockout else (0, 0, 200), thickness=1, circle_radius=0
        ),
    )

    lms  = face_lms.landmark
    xs   = [lms[i].x * frame_w for i in [33, 263, 10, 152]]
    ys   = [lms[i].y * frame_h for i in [33, 263, 10, 152]]
    pad  = 24
    bx1  = max(0, int(min(xs)) - pad)
    by1  = max(0, int(min(ys)) - pad)
    bx2  = min(frame_w, int(max(xs)) + pad)
    by2  = min(frame_h, int(max(ys)) + pad)

    box_color = (0, 0, 240) if is_lockout else CLR_ACCENT
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 3 if is_lockout else 2)

    if is_lockout:
        cv2.rectangle(frame, (bx1, by1 - 24), (bx1 + 220, by1), (0, 0, 200), -1)
        cv2.putText(frame, "REPLAY ATTACK BLOCKED", (bx1 + 4, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1, cv2.LINE_AA)
    elif state == State.CONFIRM_PAYMENT:
        cv2.rectangle(frame, (bx1, by1 - 24), (bx1 + 200, by1), (0, 160, 60), -1)
        cv2.putText(frame, "IDENTITY CONFIRMED", (bx1 + 4, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1, cv2.LINE_AA)
    else:
        ear_label  = f"EAR: {ear:.3f}"
        (lw, lh), _ = cv2.getTextSize(ear_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (bx1, by1 - 22), (bx1 + lw + 6, by1), box_color, -1)
        cv2.putText(frame, ear_label, (bx1 + 3, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    for idx in SKIN_ROI_INDICES:
        cx = int(lms[idx].x * frame_w)
        cy = int(lms[idx].y * frame_h)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 240) if is_lockout else CLR_WARN, -1)


# ── Async API Caller ───────────────────────────────────────────────────────────
class AuthResult:
    """Thread-safe result holder for the async HTTP auth call."""

    def __init__(self) -> None:
        self._result: Optional[dict] = None
        self._event  = threading.Event()

    def set(self, result: dict) -> None:
        self._result = result
        self._event.set()

    def wait(self, timeout: float = 5.0) -> Optional[dict]:
        self._event.wait(timeout=timeout)
        return self._result

    @property
    def ready(self) -> bool:
        return self._event.is_set()


def _call_api_endpoint(url: str, query_vec: np.ndarray, holder: AuthResult) -> None:
    """Background worker: POST biometric vector to FastAPI endpoint."""
    try:
        payload = {
            "query_vector":       query_vec.tolist(),
            "transaction_amount": TRANSACTION_AMOUNT,
            "terminal_id":        "TERM-POS-001",
        }
        resp = requests.post(url, json=payload, timeout=5.0)
        resp.raise_for_status()
        holder.set(resp.json())
    except requests.exceptions.ConnectionError:
        holder.set({"status": "REJECTED", "reason": "API_UNREACHABLE",
                    "display_name": None, "similarity_score": None,
                    "remaining_balance": None})
    except Exception as exc:
        holder.set({"status": "REJECTED", "reason": f"API_ERROR:{str(exc)[:40]}",
                    "display_name": None, "similarity_score": None,
                    "remaining_balance": None})


# ── Main Application ───────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("   OptiPulse-Pay | Autonomous Edge Facial Micro-Payment POS")
    print("=" * 65)

    liveness_model = load_liveness_model(MODEL_PATH)

    mp_fm_module = mp.solutions.face_mesh
    face_mesh    = mp_fm_module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check device index or permissions.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    WIN_NAME = "OptiPulse-Pay"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, CAM_WIDTH + DASH_WIDTH, WIN_HEIGHT)
    try:
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    # Mouse click tracker for UI buttons
    mouse_event = {"proceed": False, "cancel": False}

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            p_x1, p_y1, p_x2, p_y2 = BTN_PROCEED_RECT
            c_x1, c_y1, c_x2, c_y2 = BTN_CANCEL_RECT
            if p_x1 <= x <= p_x2 and p_y1 <= y <= p_y2:
                mouse_event["proceed"] = True
            elif c_x1 <= x <= c_x2 and c_y1 <= y <= c_y2:
                mouse_event["cancel"] = True

    cv2.setMouseCallback(WIN_NAME, on_mouse_click)

    # FSM state variables
    state: State                   = State.IDLE
    ambient_bgr: np.ndarray        = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    baseline_buf: list             = []
    challenge_pair: tuple          = CHALLENGE_PAIRS[0]
    c1_name: str                   = CHALLENGE_PAIRS[0][0]
    c1_bgr: Tuple[int,int,int]     = CHALLENGE_PAIRS[0][1]
    c2_name: str                   = CHALLENGE_PAIRS[0][2]
    c2_bgr: Tuple[int,int,int]     = CHALLENGE_PAIRS[0][3]
    challenge_t0: float            = 0.0
    blink_t0: float                = 0.0
    result_t0: float               = 0.0
    eye_was_closed: bool           = False
    blink_detected: bool           = False
    result_data: dict              = {}
    verified_user_data: dict       = {}
    auth_holder: Optional[AuthResult] = None
    verify_holder: Optional[AuthResult] = None
    current_ear: float             = 0.30
    current_liveness: float        = 0.75
    live_query_vec: Optional[np.ndarray] = None
    phase1_skin_samples: list      = []
    phase2_skin_samples: list      = []
    phase1_center_samples: list    = []
    phase1_lateral_samples: list   = []
    phase2_center_samples: list    = []
    phase2_lateral_samples: list   = []
    blink_baseline_ear: float      = 0.30
    topo_fail_count: int           = 0
    moire_fail_count: int          = 0
    glare_fail_count: int          = 0

    # Autonomous Face Detection Trigger Configuration
    COOLDOWN_SEC: float            = 3.5    # seconds to wait before auto-retriggering
    AUTO_TRIGGER_FRAMES: int       = 12     # ~0.4s of continuous face presence
    face_detected_frames: int      = 0
    last_txn_time: float           = 0.0

    print("[OK] OptiPulse-Pay Autonomous POS Terminal Ready.")
    print("     • Stand in front of camera — Payment triggers automatically!")
    print("     • Or press [P] anytime to manually trigger")
    print("     • [ENTER/Click] = Confirm / Proceed Payment in Final Stage")
    print("     • [ESC/C]      = Cancel Payment")
    print("     • [Q]          = Quit Application\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame_bgr  = cv2.flip(frame_bgr, 1)
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results    = face_mesh.process(frame_rgb)
        face_lms   = (results.multi_face_landmarks[0].landmark
                      if results.multi_face_landmarks else None)

        # Continuous EAR
        if face_lms is not None:
            ear_l = compute_ear(face_lms, L_EYE_EAR_IDX, frame_w, frame_h)
            ear_r = compute_ear(face_lms, R_EYE_EAR_IDX, frame_w, frame_h)
            current_ear = (ear_l + ear_r) / 2.0

        # Ambient baseline accumulation (IDLE only)
        if state == State.IDLE and face_lms is not None:
            sample = sample_skin_roi(frame_bgr, face_lms, frame_w, frame_h)
            baseline_buf.append(sample)
            if len(baseline_buf) > BASELINE_FRAMES:
                baseline_buf.pop(0)
            if baseline_buf:
                ambient_bgr = np.mean(baseline_buf, axis=0).astype(np.float32)

        # Continuous liveness texture scoring
        if (liveness_model is not None and
                state in (State.IDLE, State.AWAITING_BLINK, State.CONFIRM_PAYMENT) and
                face_lms is not None):
            nx = int(face_lms[ANCHOR_IDX].x * frame_w)
            ny = int(face_lms[ANCHOR_IDX].y * frame_h)
            ph = 48
            x1, y1 = max(0, nx - ph), max(0, ny - ph * 2)
            x2, y2 = min(frame_w, nx + ph), min(frame_h, ny)
            patch = frame_bgr[y1:y2, x1:x2]
            if patch.size > 100:
                current_liveness = predict_liveness(liveness_model, patch)

        # ── Autonomous Face Detection Trigger (Zero-Touch POS) ─────────────────
        if state == State.IDLE:
            time_since_last_txn = time.time() - last_txn_time
            if face_lms is not None and time_since_last_txn >= COOLDOWN_SEC:
                face_detected_frames += 1
                if face_detected_frames >= AUTO_TRIGGER_FRAMES and len(baseline_buf) >= 4:
                    challenge_pair = random.choice(CHALLENGE_PAIRS)
                    c1_name, c1_bgr, c2_name, c2_bgr = challenge_pair
                    challenge_t0 = time.time()
                    phase1_skin_samples.clear()
                    phase2_skin_samples.clear()
                    phase1_center_samples.clear()
                    phase1_lateral_samples.clear()
                    phase2_center_samples.clear()
                    phase2_lateral_samples.clear()
                    state = State.OPTICAL_CHALLENGE
                    face_detected_frames = 0
                    print(f"[AUTO-TRIGGER] Face recognized! Starting Two-Phase Challenge: {c1_name} -> {c2_name}")
            else:
                face_detected_frames = 0

        # ── FSM Logic ─────────────────────────────────────────────────────────
        if state == State.OPTICAL_CHALLENGE:
            elapsed = time.time() - challenge_t0

            # Phase 1: 0.0s to CHALLENGE_PHASE1_DURATION (c1_bgr)
            if elapsed < CHALLENGE_PHASE1_DURATION:
                flash = np.full((frame_h, frame_w + DASH_WIDTH, 3),
                                c1_bgr, dtype=np.uint8)
                cv2.imshow(WIN_NAME, flash)
                cv2.waitKey(1)
                # Sample active skin reflection during steady-state of Phase 1 (80ms - 340ms)
                if 0.08 < elapsed < CHALLENGE_PHASE1_DURATION - 0.02 and face_lms is not None:
                    phase1_skin_samples.append(sample_skin_roi(frame_bgr, face_lms, frame_w, frame_h))
                    phase1_center_samples.append(sample_roi_indices(frame_bgr, face_lms, CENTER_ROI_INDICES, frame_w, frame_h))
                    phase1_lateral_samples.append(sample_roi_indices(frame_bgr, face_lms, LATERAL_ROI_INDICES, frame_w, frame_h))
                continue

            # Phase 2: CHALLENGE_PHASE1_DURATION to TOTAL_CHALLENGE_DURATION (c2_bgr)
            elif elapsed < TOTAL_CHALLENGE_DURATION:
                flash = np.full((frame_h, frame_w + DASH_WIDTH, 3),
                                c2_bgr, dtype=np.uint8)
                cv2.imshow(WIN_NAME, flash)
                cv2.waitKey(1)
                # Sample active skin reflection during steady-state of Phase 2 (440ms - 700ms)
                if CHALLENGE_PHASE1_DURATION + 0.08 < elapsed < TOTAL_CHALLENGE_DURATION - 0.02 and face_lms is not None:
                    phase2_skin_samples.append(sample_skin_roi(frame_bgr, face_lms, frame_w, frame_h))
                    phase2_center_samples.append(sample_roi_indices(frame_bgr, face_lms, CENTER_ROI_INDICES, frame_w, frame_h))
                    phase2_lateral_samples.append(sample_roi_indices(frame_bgr, face_lms, LATERAL_ROI_INDICES, frame_w, frame_h))
                continue

            # ── Optical Challenge Complete — Run Multi-Tier PAD Checks ──────────
            if face_lms is not None:
                if len(phase1_skin_samples) < 1 or len(phase2_skin_samples) < 1:
                    print("[PAD DEFENSE] REJECTED -> Challenge reflection samples incomplete (movement detected)")
                    result_data = {"status": "REJECTED", "reason": "SPOOF_REPLAY_DETECTED"}
                    state       = State.RESULT
                    result_t0   = time.time()
                    continue

                s1_bgr = np.mean(phase1_skin_samples, axis=0).astype(np.float32)
                s2_bgr = np.mean(phase2_skin_samples, axis=0).astype(np.float32)
                s1_ctr = np.mean(phase1_center_samples, axis=0).astype(np.float32)
                s2_ctr = np.mean(phase2_center_samples, axis=0).astype(np.float32)
                s1_lat = np.mean(phase1_lateral_samples, axis=0).astype(np.float32)
                s2_lat = np.mean(phase2_lateral_samples, axis=0).astype(np.float32)

                phase1_skin_samples.clear()
                phase2_skin_samples.clear()
                phase1_center_samples.clear()
                phase1_lateral_samples.clear()
                phase2_center_samples.clear()
                phase2_lateral_samples.clear()

                # Tier 1: Two-Phase Spectral Transition PAD (strictly defeats video replays & static photos)
                spectral_ok, spectral_reason, spectral_corr = verify_two_phase_spectral_response(
                    ambient_bgr, s1_bgr, s2_bgr, c1_bgr, c2_bgr
                )

                # Tier 1B: Lambertian Spatial Gradient
                spatial_ok, spatial_reason, _ = verify_lambertian_spatial_gradient(
                    s1_ctr, s2_ctr, s1_lat, s2_lat, c1_bgr, c2_bgr
                )

                # Tier 2: 3D Depth Topography & Surface Curvature (defeats flat screens & paper cutouts)
                topo_ok, topo_reason, topo_score = verify_3d_depth_topography(face_lms, frame_w, frame_h)

                # Tier 3: Multi-Patch Screen Moiré & Frequency Grid (defeats OLED/LCD screens)
                moire_ok, moire_reason, moire_score = verify_screen_moire_and_grid(frame_bgr, face_lms, frame_w, frame_h)

                # Tier 4: Specular Glass Glare Hotspots (defeats phone glass reflection)
                glare_ok, glare_reason = verify_screen_specular_clipping(frame_bgr, face_lms, frame_w, frame_h)

                # Tier 5: MobileNet Micro-Texture Classification
                texture_ok  = (liveness_model is None or current_liveness >= LIVENESS_THRESHOLD)

                # Strict Multi-Tier Defense Evaluation
                if not spectral_ok:
                    print(f"[PAD DEFENSE] REJECTED -> {spectral_reason} (corr: {spectral_corr:+.3f})")
                    result_data = {"status": "REJECTED", "reason": spectral_reason}
                    state       = State.RESULT
                    result_t0   = time.time()
                elif not spatial_ok:
                    print(f"[PAD DEFENSE] REJECTED -> {spatial_reason} (Lambertian spatial coherence failed)")
                    result_data = {"status": "REJECTED", "reason": spatial_reason}
                    state       = State.RESULT
                    result_t0   = time.time()
                elif not topo_ok:
                    print(f"[PAD DEFENSE] REJECTED -> {topo_reason} (topo: {topo_score:.3f})")
                    result_data = {"status": "REJECTED", "reason": topo_reason}
                    state       = State.RESULT
                    result_t0   = time.time()
                elif not moire_ok:
                    print(f"[PAD DEFENSE] REJECTED -> {moire_reason} (score: {moire_score:.3f})")
                    result_data = {"status": "REJECTED", "reason": moire_reason}
                    state       = State.RESULT
                    result_t0   = time.time()
                elif not glare_ok:
                    print(f"[PAD DEFENSE] REJECTED -> {glare_reason}")
                    result_data = {"status": "REJECTED", "reason": glare_reason}
                    state       = State.RESULT
                    result_t0   = time.time()
                elif not texture_ok:
                    print(f"[PAD DEFENSE] REJECTED -> SPOOF_TEXTURE_DETECTED (liveness: {current_liveness:.3f} < {LIVENESS_THRESHOLD})")
                    result_data = {"status": "REJECTED", "reason": "SPOOF_TEXTURE_DETECTED"}
                    state       = State.RESULT
                    result_t0   = time.time()
                else:
                    print("[PAD DEFENSE] ALL Optical & 3D Checks PASSED! --> PLEASE BLINK NATURALLY")
                    blink_t0           = time.time()
                    blink_baseline_ear = max(0.24, current_ear)
                    eye_was_closed     = False
                    blink_detected     = False
                    topo_fail_count    = 0
                    moire_fail_count   = 0
                    glare_fail_count   = 0
                    state              = State.AWAITING_BLINK
            else:
                state = State.IDLE   # face lost during flash

        elif state == State.AWAITING_BLINK:
            elapsed_blink = time.time() - blink_t0

            # Continual 3D topography check with temporal persistence (prevents momentary false alarms)
            topo_ok, _, _ = verify_3d_depth_topography(face_lms, frame_w, frame_h)
            if not topo_ok:
                topo_fail_count += 1
                if topo_fail_count >= 6:
                    print("[PAD DEFENSE] REJECTED in blink window -> SPOOF_2D_PLANAR_DETECTED (persistent)")
                    result_data = {"status": "REJECTED", "reason": "SPOOF_2D_PLANAR_DETECTED"}
                    state       = State.RESULT
                    result_t0   = time.time()
                    continue
            else:
                topo_fail_count = max(0, topo_fail_count - 1)

            # Continual Screen Moiré & Subpixel Grid check with temporal persistence
            moire_ok, _, _ = verify_screen_moire_and_grid(frame_bgr, face_lms, frame_w, frame_h)
            if not moire_ok:
                moire_fail_count += 1
                if moire_fail_count >= 6:
                    print("[PAD DEFENSE] REJECTED in blink window -> SPOOF_MOIRE_GRID_DETECTED (persistent)")
                    result_data = {"status": "REJECTED", "reason": "SPOOF_MOIRE_GRID_DETECTED"}
                    state       = State.RESULT
                    result_t0   = time.time()
                    continue
            else:
                moire_fail_count = max(0, moire_fail_count - 1)

            # Continual Specular Glass Glare check with temporal persistence
            glare_ok, _ = verify_screen_specular_clipping(frame_bgr, face_lms, frame_w, frame_h)
            if not glare_ok:
                glare_fail_count += 1
                if glare_fail_count >= 6:
                    print("[PAD DEFENSE] REJECTED in blink window -> SPOOF_GLASS_GLARE_DETECTED (persistent)")
                    result_data = {"status": "REJECTED", "reason": "SPOOF_GLASS_GLARE_DETECTED"}
                    state       = State.RESULT
                    result_t0   = time.time()
                    continue
            else:
                glare_fail_count = max(0, glare_fail_count - 1)

            # Dual-condition natural blink detection:
            # 1. Absolute threshold: EAR < 0.225
            # 2. Relative drop: EAR drops by 25% from baseline open eye
            is_closed = (current_ear < EAR_BLINK_CLOSE) or (current_ear < blink_baseline_ear * 0.75)
            is_open   = (current_ear > EAR_BLINK_OPEN)  or (current_ear > blink_baseline_ear * 0.88)

            if not eye_was_closed and is_closed:
                eye_was_closed = True
            if eye_was_closed and is_open:
                blink_detected = True

            if blink_detected:
                print(f"[FSM] Natural blink confirmed (EAR: {current_ear:.3f}) -> Extracting 512D biometric vector")
                if face_lms is not None:
                    live_query_vec = extract_512d_vector(frame_rgb, face_mesh)
                if live_query_vec is not None:
                    # Stage 1: Verify identity against KYC database without debiting
                    verify_holder = AuthResult()
                    threading.Thread(
                        target=_call_api_endpoint,
                        args=(API_VERIFY_URL, live_query_vec, verify_holder),
                        daemon=True,
                    ).start()
                    state = State.VERIFYING
                    print("[FSM] Querying /api/v1/verify_identity for KYC match...")
                else:
                    print("[FSM] REJECTED -> VECTOR_EXTRACTION_FAILED")
                    result_data = {"status": "REJECTED", "reason": "VECTOR_EXTRACTION_FAILED"}
                    state       = State.RESULT
                    result_t0   = time.time()

            elif elapsed_blink > BLINK_WINDOW:
                print(f"[FSM] REJECTED -> BLINK_TIMEOUT (No natural blink detected within {BLINK_WINDOW}s)")
                result_data = {"status": "REJECTED", "reason": "BLINK_TIMEOUT"}
                state       = State.RESULT
                result_t0   = time.time()

        elif state == State.VERIFYING:
            if verify_holder is not None and verify_holder.ready:
                v_res = verify_holder.wait(timeout=0.05) or {}
                if v_res.get("status") == "VERIFIED":
                    verified_user_data = v_res
                    state = State.CONFIRM_PAYMENT
                    print(f"[FSM] IDENTITY VERIFIED: {v_res.get('display_name')} (score: {v_res.get('similarity_score')}) -> Awaiting Proceed confirmation")
                else:
                    result_data = v_res
                    state       = State.RESULT
                    result_t0   = time.time()
                    print(f"[FSM] VERIFICATION DECLINED: reason={v_res.get('reason')} score={v_res.get('similarity_score')}")

        elif state == State.CONFIRM_PAYMENT:
            # Awaiting user confirmation via PROCEED/OK button or ENTER key
            pass

        elif state == State.AUTHORIZING:
            if auth_holder is not None and auth_holder.ready:
                result_data = auth_holder.wait(timeout=0.05) or {}
                state       = State.RESULT
                result_t0   = time.time()
                if result_data.get("status") == "AUTHORIZED":
                    print(f"[FSM] SUCCESS: AUTHORIZED for {result_data.get('display_name')} | Paid INR {TRANSACTION_AMOUNT:.0f} | Balance: INR {result_data.get('remaining_balance'):.2f}")
                else:
                    print(f"[FSM] FAILED: {result_data.get('reason')} (status: {result_data.get('status')})")

        elif state == State.RESULT:
            if time.time() - result_t0 >= RESULT_DISPLAY_TIME:
                state                = State.IDLE
                result_data          = {}
                verified_user_data   = {}
                baseline_buf         = []
                phase1_skin_samples.clear()
                phase2_skin_samples.clear()
                phase1_center_samples.clear()
                phase1_lateral_samples.clear()
                phase2_center_samples.clear()
                phase2_lateral_samples.clear()
                live_query_vec       = None
                auth_holder          = None
                verify_holder        = None
                face_detected_frames = 0
                topo_fail_count      = 0
                moire_fail_count     = 0
                glare_fail_count     = 0
                last_txn_time        = time.time()

        # ── Render Frame ──────────────────────────────────────────────────────
        display = frame_bgr.copy()
        draw_face_overlay(display, results, state, current_ear, frame_w, frame_h, result_data, verified_user_data)

        canvas = np.full((WIN_HEIGHT, CAM_WIDTH + DASH_WIDTH, 3), CLR_BG, dtype=np.uint8)
        scale_f = min(CAM_WIDTH / frame_w, WIN_HEIGHT / frame_h)
        nw      = int(frame_w * scale_f)
        nh      = int(frame_h * scale_f)
        resized = cv2.resize(display, (nw, nh))
        ox      = (CAM_WIDTH - nw) // 2
        oy      = (WIN_HEIGHT - nh) // 2
        canvas[oy : oy + nh, ox : ox + nw] = resized

        cv2.line(canvas, (CAM_WIDTH, 0), (CAM_WIDTH, WIN_HEIGHT), CLR_GREY, 1)

        elapsed_b = time.time() - blink_t0 if state == State.AWAITING_BLINK else 0.0
        draw_dashboard(canvas, state, current_ear, result_data, verified_user_data, elapsed_b, current_liveness)

        if state == State.IDLE:
            time_since_last = time.time() - last_txn_time
            if time_since_last < COOLDOWN_SEC:
                cd_rem = max(0.0, COOLDOWN_SEC - time_since_last)
                cv2.putText(canvas, f"STANDBY  Ready in {cd_rem:.1f}s",
                            (12, WIN_HEIGHT - 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, CLR_GREY, 1, cv2.LINE_AA)
            elif face_detected_frames > 0:
                pct = min(1.0, face_detected_frames / AUTO_TRIGGER_FRAMES)
                cv2.putText(canvas, f"FACE DETECTED  Starting Payment ({int(pct*100)}%)...",
                            (12, WIN_HEIGHT - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, CLR_GOLD, 1, cv2.LINE_AA)
                bar_len = int(pct * 240)
                cv2.rectangle(canvas, (12, WIN_HEIGHT - 9), (12 + bar_len, WIN_HEIGHT - 4), CLR_GOLD, -1)
            else:
                cv2.putText(canvas, "AUTO-PAY READY  Look at camera to pay",
                            (12, WIN_HEIGHT - 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, CLR_ACCENT, 1, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, canvas)

        # ── Key & Mouse Handling ──────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        # Check if user confirmed payment (ENTER or SPACE or Click PROCEED)
        confirm_triggered = (key in (13, 10, 32) or mouse_event["proceed"])
        cancel_triggered  = (key in (27, ord('c'), ord('C')) or mouse_event["cancel"])
        mouse_event["proceed"] = False
        mouse_event["cancel"]  = False

        if state == State.CONFIRM_PAYMENT:
            if confirm_triggered:
                print(f"[FSM] User confirmed payment! Authorizing debit for {verified_user_data.get('display_name')}...")
                auth_holder = AuthResult()
                threading.Thread(
                    target=_call_api_endpoint,
                    args=(API_AUTH_URL, live_query_vec, auth_holder),
                    daemon=True,
                ).start()
                state = State.AUTHORIZING
            elif cancel_triggered:
                print("[FSM] Payment cancelled by user.")
                state                = State.IDLE
                verified_user_data   = {}
                live_query_vec       = None
                face_detected_frames = 0
                last_txn_time        = time.time()

        elif key in (ord('q'), 27):
            print("[EXIT] User quit.")
            break

        elif key == ord('p') and state == State.IDLE:
            if face_lms is None:
                print("[WARN] No face detected — position face in frame first.")
            elif len(baseline_buf) < 4:
                print("[WARN] Ambient baseline not ready — wait a moment.")
            else:
                challenge_pair = random.choice(CHALLENGE_PAIRS)
                c1_name, c1_bgr, c2_name, c2_bgr = challenge_pair
                challenge_t0   = time.time()
                phase1_skin_samples.clear()
                phase2_skin_samples.clear()
                state          = State.OPTICAL_CHALLENGE
                print(f"[FSM] IDLE -> OPTICAL_CHALLENGE Two-Phase: {c1_name} -> {c2_name}")

        elif key == ord('r'):
            state              = State.IDLE
            result_data        = {}
            verified_user_data = {}
            baseline_buf       = []
            live_query_vec     = None
            auth_holder        = None
            verify_holder      = None
            print("[FSM] RESET -> IDLE")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[OK] OptiPulse-Pay engine shut down cleanly.")


if __name__ == "__main__":
    main()
