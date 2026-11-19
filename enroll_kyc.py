#!/usr/bin/env python3
"""
enroll_kyc.py — OptiPulse-Pay KYC Enrollment Pipeline
Processes 2D portrait images from kyc_database/ and extracts 512D biometric vectors.
"""

import os
import sys
import json
import glob
import math
import argparse
import warnings
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from datetime import datetime, timezone

# Suppress benign Google Protobuf symbol_database deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Constants ─────────────────────────────────────────────────────────────────
ENROLLED_DB_PATH = Path("kyc_database/enrolled_users.json")
KYC_IMG_DIR      = Path("kyc_database")
VECTOR_DIM       = 512

# MediaPipe landmark indices used for 170-landmark structural extraction
# Jaw contour (17 pts): 10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400
JAW_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400]
# Left eyebrow (10 pts)
L_BROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
# Right eyebrow (10 pts)
R_BROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
# Left eye (16 pts)
L_EYE_INDICES  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Right eye (16 pts)
R_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Nose (20 pts)
NOSE_INDICES   = [1, 2, 5, 4, 19, 94, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 168, 6, 197, 195]
# Lips outer (20 pts)
LIPS_O_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
# Lips inner (12 pts)
LIPS_I_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324]
# Chin/face center (10 pts)
CHIN_INDICES   = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93]
# Cheekbone (20 pts each side = 40 pts)
L_CHEEK_INDICES = [116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 199, 200, 194, 204, 202, 212, 216, 206, 203, 36]
R_CHEEK_INDICES = [345, 352, 376, 433, 416, 434, 430, 431, 262, 428, 423, 424, 418, 422, 432, 436, 426, 427, 266]

ALL_LANDMARK_INDICES = list(dict.fromkeys(
    JAW_INDICES + L_BROW_INDICES + R_BROW_INDICES +
    L_EYE_INDICES + R_EYE_INDICES + NOSE_INDICES +
    LIPS_O_INDICES + LIPS_I_INDICES + CHIN_INDICES +
    L_CHEEK_INDICES + R_CHEEK_INDICES
))
# Deduplicate and cap to ensure exactly 170 unique indices
ALL_LANDMARK_INDICES = list(dict.fromkeys(ALL_LANDMARK_INDICES))[:170]

# Anchor: nasal bridge (landmark 1), interocular distance anchors (33 → 263)
ANCHOR_IDX       = 1
L_EYE_OUTER_IDX  = 33
R_EYE_OUTER_IDX  = 263


def _load_face_mesh():
    """Initialize MediaPipe FaceMesh with refined landmarks."""
    mp_fm = mp.solutions.face_mesh
    return mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _preprocess_image(img_path: str) -> np.ndarray:
    """Load and preprocess image: resize to 640px wide, convert BGR→RGB."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    h, w = img.shape[:2]
    scale = 640.0 / w
    img = cv2.resize(img, (640, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_512d_vector(rgb_image: np.ndarray, face_mesh) -> np.ndarray | None:
    """
    Extract a 512-dimensional L2-normalized biometric vector from a face image.

    Pipeline:
      1. Run MediaPipe FaceMesh to get 478 normalized landmarks.
      2. Select 170 structural landmarks (jaw, brows, eyes, nose, lips, cheeks).
      3. Anchor to nasal bridge (lm[1]) — subtract to make translation-invariant.
      4. Scale by interocular distance (euclidean(lm[33], lm[263])) — scale-invariant.
      5. Flatten (x, y, z) coords → 510D raw geometric vector.
      6. Append 2 holistic ratio features: face_height_ratio, jaw_width_ratio → 512D.
      7. L2-normalize to unit sphere for cosine similarity via dot-product.
    """
    h, w = rgb_image.shape[:2]
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None

    lms = results.multi_face_landmarks[0].landmark

    # Pixel coordinates for all 478 landmarks
    coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float64)

    # Anchor: nasal bridge
    anchor = coords[ANCHOR_IDX].copy()
    coords -= anchor

    # Interocular distance for scale normalization
    iod = np.linalg.norm(coords[L_EYE_OUTER_IDX] - coords[R_EYE_OUTER_IDX])
    if iod < 1e-6:
        return None
    coords /= iod

    # Select 170 structural landmarks → flatten to 510D
    selected = coords[ALL_LANDMARK_INDICES]   # (170, 3)
    geo_vec = selected.flatten()               # (510,)

    # Holistic ratio features (2D → padded to 2 dims)
    # Face height ratio: distance from chin (152) to forehead (10) / IOD
    face_height = np.linalg.norm(coords[152] - coords[10])
    jaw_width   = np.linalg.norm(coords[234] - coords[454])  # lateral jaw width
    ratio_feats = np.array([face_height, jaw_width], dtype=np.float64)

    # Concatenate → 512D
    vec_512 = np.concatenate([geo_vec, ratio_feats])

    # L2 normalization
    norm = np.linalg.norm(vec_512)
    if norm < 1e-9:
        return None
    return (vec_512 / norm).astype(np.float32)


def enroll_user(img_path: str, user_id: str, display_name: str, initial_balance: float,
                face_mesh, db: dict) -> bool:
    """Enroll a single user from their portrait image."""
    try:
        rgb = _preprocess_image(img_path)
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        return False

    vec = extract_512d_vector(rgb, face_mesh)
    if vec is None:
        print(f"  [WARN ] No face detected in {img_path} — skipping.")
        return False

    # Save vector as .npy alongside the image
    npy_path = str(Path(img_path).with_suffix(".npy"))
    np.save(npy_path, vec)

    # Upsert into in-memory DB
    db[user_id] = {
        "user_id":       user_id,
        "display_name":  display_name,
        "vector_path":   npy_path,
        "balance":       initial_balance,
        "enrolled_at":   datetime.now(timezone.utc).isoformat(),
        "img_path":      img_path,
    }
    print(f"  [OK   ] Enrolled '{display_name}' ({user_id}) — vector: {npy_path}")
    return True


def load_enrolled_db() -> dict:
    """Load existing enrolled users database, or return empty dict."""
    if ENROLLED_DB_PATH.exists():
        with open(ENROLLED_DB_PATH, "r") as f:
            return json.load(f)
    return {}


def save_enrolled_db(db: dict) -> None:
    """Persist the enrolled users database to disk."""
    ENROLLED_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENROLLED_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)
    print(f"\n[DB] Saved {len(db)} enrolled user(s) to {ENROLLED_DB_PATH}")


def run_batch_enrollment(kyc_dir: Path) -> None:
    """
    Scan kyc_dir for portrait images. Expected naming convention:
        <user_id>__<display_name>.jpg   e.g.  USR001__Alice_Kumar.jpg
    Images without this convention are enrolled with auto-generated IDs.
    """
    db = load_enrolled_db()
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        p for p in kyc_dir.iterdir()
        if p.suffix.lower() in supported_exts and not p.stem.startswith(".")
    ]

    if not image_paths:
        print(f"[WARN] No portrait images found in '{kyc_dir}'. "
              "Place JPEG/PNG files named as 'USERID__Name.jpg' to enroll users.")
        _create_demo_enrollment(kyc_dir, db)
        return

    face_mesh = _load_face_mesh()
    enrolled_count = 0

    for idx, img_path in enumerate(sorted(image_paths)):
        stem = img_path.stem
        if "__" in stem:
            parts = stem.split("__", 1)
            user_id      = parts[0].strip()
            display_name = parts[1].replace("_", " ").strip()
        else:
            user_id      = f"USR{idx+1:03d}"
            display_name = stem.replace("_", " ").title()

        print(f"\n[{idx+1}/{len(image_paths)}] Processing: {img_path.name}")
        ok = enroll_user(
            img_path=str(img_path),
            user_id=user_id,
            display_name=display_name,
            initial_balance=1000.0,
            face_mesh=face_mesh,
            db=db,
        )
        if ok:
            enrolled_count += 1

    save_enrolled_db(db)
    print(f"\n✓ Enrollment complete: {enrolled_count}/{len(image_paths)} users enrolled.")


def _create_demo_enrollment(kyc_dir: Path, db: dict) -> None:
    """Create a synthetic demo enrollment entry so the API can start without real images."""
    rng = np.random.default_rng(seed=42)
    demo_vec = rng.random(VECTOR_DIM).astype(np.float32)
    demo_vec /= np.linalg.norm(demo_vec)

    demo_npy = kyc_dir / "DEMO001.npy"
    np.save(str(demo_npy), demo_vec)

    db["DEMO001"] = {
        "user_id":      "DEMO001",
        "display_name": "Demo User",
        "vector_path":  str(demo_npy),
        "balance":      5000.0,
        "enrolled_at":  datetime.now(timezone.utc).isoformat(),
        "img_path":     "",
    }
    save_enrolled_db(db)
    print("[DEMO] Created synthetic demo user 'DEMO001' with balance ₹5000.")


def main():
    parser = argparse.ArgumentParser(
        description="OptiPulse-Pay KYC Enrollment Pipeline"
    )
    parser.add_argument(
        "--kyc-dir", type=Path, default=KYC_IMG_DIR,
        help="Directory containing portrait images (default: kyc_database/)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe existing enrollments before re-processing"
    )
    parser.add_argument(
        "--capture", action="store_true",
        help="Launch interactive camera capture wizard for you and a friend"
    )
    args = parser.parse_args()

    if args.capture:
        from capture_enroll import interactive_wizard
        interactive_wizard()
        return

    if args.reset and ENROLLED_DB_PATH.exists():
        ENROLLED_DB_PATH.unlink()
        print("[RESET] Cleared existing enrolled_users.json")

    print("=" * 60)
    print("  OptiPulse-Pay — KYC Enrollment Pipeline")
    print("=" * 60)
    run_batch_enrollment(args.kyc_dir)


if __name__ == "__main__":
    main()
