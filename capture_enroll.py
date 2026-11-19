#!/usr/bin/env python3
"""
capture_enroll.py — OptiPulse-Pay Interactive Face KYC Enrollment Tool
Captures high-resolution face portraits directly from the optical camera for the user and friend,
extracts 512D invariant biometric vectors, and registers them into the KYC ledger.
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path

# Suppress benign Google Protobuf symbol_database deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import numpy as np
import mediapipe as mp
import requests

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Paths & Constants ──────────────────────────────────────────────────────────
KYC_DIR          = Path("kyc_database")
ENROLLED_DB_PATH = KYC_DIR / "enrolled_users.json"
API_RELOAD_URL   = "http://127.0.0.1:8000/api/v1/reload_ledger"
VECTOR_DIM       = 512

# Structural 170 landmark indices matching enroll_kyc.py
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

ANCHOR_IDX      = 1
L_EYE_OUTER_IDX = 33
R_EYE_OUTER_IDX = 263

# UI Palette (BGR)
CLR_BG     = (22, 22, 30)
CLR_PANEL  = (32, 32, 44)
CLR_ACCENT = (0, 220, 130)
CLR_WARN   = (0, 175, 255)
CLR_ERROR  = (60, 60, 230)
CLR_WHITE  = (245, 245, 245)
CLR_GREY   = (130, 130, 140)
CLR_GOLD   = (30, 205, 235)


def extract_512d_vector(rgb_image: np.ndarray, face_mesh) -> np.ndarray | None:
    """Extract scale/pose invariant 512D unit vector."""
    h, w = rgb_image.shape[:2]
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None

    lms = results.multi_face_landmarks[0].landmark
    coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float64)

    # Translation invariance
    anchor = coords[ANCHOR_IDX].copy()
    coords -= anchor

    # Scale invariance via interocular distance
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
    norm = np.linalg.norm(vec_512)
    if norm < 1e-9:
        return None
    return (vec_512 / norm).astype(np.float32)


def load_db() -> dict:
    if ENROLLED_DB_PATH.exists():
        try:
            with open(ENROLLED_DB_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_db(db: dict) -> None:
    KYC_DIR.mkdir(parents=True, exist_ok=True)
    with open(ENROLLED_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def notify_api_server():
    """Hot-reload in-memory ledger if api_server.py is running."""
    try:
        resp = requests.post(API_RELOAD_URL, timeout=1.0)
        if resp.status_code == 200:
            print("  [API] Hot-reloaded ledger in running api_server.")
    except Exception:
        pass


def capture_and_enroll_interactive(user_id: str, display_name: str, initial_balance: float = 1000.0) -> bool:
    """Opens live camera and guides the user to capture their portrait."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera (index 0).")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    win_name = "OptiPulse-Pay KYC Camera Enrollment"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1000, 520)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    # Mouse click tracker for UI buttons
    button_clicked = {"capture": False, "cancel": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked inside CAPTURE button [670, 390, 960, 440]
            if 670 <= x <= 960 and 390 <= y <= 440:
                button_clicked["capture"] = True
            elif 670 <= x <= 960 and 455 <= y <= 495:
                button_clicked["cancel"] = True

    cv2.setMouseCallback(win_name, on_mouse)

    enrolled = False
    status_msg = "Center face inside the oval and look directly at camera"
    status_color = CLR_WHITE

    print(f"\n[ENROLLING] {display_name} ({user_id})")
    print("  Controls: [SPACE] or Click 'CAPTURE PHOTO' to enroll | [Q] to cancel\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        face_detected = bool(results.multi_face_landmarks)
        face_lms = results.multi_face_landmarks[0] if face_detected else None

        # Draw alignment ellipse
        center_x, center_y = w // 2, h // 2
        axes = (110, 150)
        ellipse_color = CLR_ACCENT if face_detected else CLR_WARN
        cv2.ellipse(frame_bgr, (center_x, center_y), axes, 0, 0, 360, ellipse_color, 2, cv2.LINE_AA)

        if face_lms:
            # Draw subtle mesh contours
            mp_draw.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_lms,
                connections=mp_fm.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 200, 80), thickness=1, circle_radius=0)
            )
            status_msg = "✓ FACE ALIGNED — READY TO CAPTURE!"
            status_color = CLR_ACCENT
        else:
            status_msg = "NO FACE DETECTED — MOVE CLOSER"
            status_color = CLR_WARN

        # Composite canvas: Left=Camera (640), Right=Panel (360)
        canvas = np.full((520, 1000, 3), CLR_BG, dtype=np.uint8)
        canvas[20:20+h, 15:15+w] = frame_bgr

        # Right control panel
        px = 670
        cv2.rectangle(canvas, (px - 10, 20), (980, 500), CLR_PANEL, -1)
        cv2.putText(canvas, "KYC BIOMETRIC ENROLLMENT", (px, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, CLR_GOLD, 1, cv2.LINE_AA)
        cv2.line(canvas, (px, 68), (960, 68), CLR_GREY, 1)

        cv2.putText(canvas, "USER PROFILE:", (px, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREY, 1)
        cv2.putText(canvas, f"ID:    {user_id}", (px, 128), cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_WHITE, 1)
        cv2.putText(canvas, f"NAME:  {display_name}", (px, 158), cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_WHITE, 1)
        cv2.putText(canvas, f"WALLET: INR {initial_balance:.2f}", (px, 188), cv2.FONT_HERSHEY_DUPLEX, 0.60, CLR_GOLD, 1)

        cv2.line(canvas, (px, 210), (960, 210), CLR_GREY, 1)
        cv2.putText(canvas, "INSTRUCTIONS:", (px, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREY, 1)
        cv2.putText(canvas, "1. Good front lighting", (px, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1)
        cv2.putText(canvas, "2. Face straight at camera", (px, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1)
        cv2.putText(canvas, "3. Neutral expression", (px, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1)

        # Status indicator
        cv2.putText(canvas, status_msg, (px, 355), cv2.FONT_HERSHEY_DUPLEX, 0.48, status_color, 1, cv2.LINE_AA)

        # CAPTURE Button
        cap_btn_color = CLR_ACCENT if face_detected else (60, 60, 60)
        cv2.rectangle(canvas, (px, 390), (960, 440), cap_btn_color, -1)
        cv2.putText(canvas, "[ CAPTURE PHOTO (SPACE) ]", (px + 20, 422),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 0, 0) if face_detected else CLR_GREY, 2, cv2.LINE_AA)

        # CANCEL Button
        cv2.rectangle(canvas, (px, 455), (960, 495), (45, 45, 55), -1)
        cv2.putText(canvas, "CANCEL / EXIT (Q)", (px + 70, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_GREY, 1, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(1) & 0xFF

        trigger_capture = (key == 32 or key == ord('c') or button_clicked["capture"])
        button_clicked["capture"] = False

        if trigger_capture:
            if not face_detected:
                print("  [WARN] Cannot capture: no face detected. Position face in oval.")
                continue

            # Extract 512D vector from current high quality frame
            vec = extract_512d_vector(frame_rgb, face_mesh)
            if vec is None:
                print("  [WARN] Vector extraction failed — please try again.")
                continue

            # Format filenames: <USER_ID>__<NAME>.jpg and .npy
            clean_name = display_name.replace(" ", "_")
            img_filename = f"{user_id}__{clean_name}.jpg"
            npy_filename = f"{user_id}__{clean_name}.npy"
            img_path = KYC_DIR / img_filename
            npy_path = KYC_DIR / npy_filename

            KYC_DIR.mkdir(parents=True, exist_ok=True)
            # Save uncompressed portrait
            cv2.imwrite(str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
            # Save 512D float32 numpy vector
            np.save(str(npy_path), vec)

            # Update database
            db = load_db()
            db[user_id] = {
                "user_id": user_id,
                "display_name": display_name,
                "vector_path": str(npy_path),
                "balance": float(initial_balance),
                "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "img_path": str(img_path),
            }
            save_db(db)
            notify_api_server()

            # Render flash / success card on canvas
            cv2.rectangle(canvas, (px, 385), (960, 495), (0, 180, 70), -1)
            cv2.putText(canvas, "✓ ENROLLED SUCCESSFULLY!", (px + 25, 425),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, "512D Biometric Vector Saved", (px + 40, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Starting Balance: INR {initial_balance:.0f}", (px + 45, 480),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(win_name, canvas)
            cv2.waitKey(2000)

            print(f"  ✓ SUCCESS! Enrolled '{display_name}' ({user_id})")
            print(f"    Image : {img_path}")
            print(f"    Vector: {npy_path} (dim=512, unit-norm=1.000)")
            print(f"    Balance: INR {initial_balance:.2f}")
            enrolled = True
            break

        if key in (ord('q'), 27) or button_clicked["cancel"]:
            print("  [CANCELLED] Enrollment cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    return enrolled


def interactive_wizard():
    print("=" * 65)
    print("   OptiPulse-Pay — Interactive Multi-User KYC Enrollment")
    print("=" * 65)
    print("This tool takes photos of you and your friend via camera,")
    print("computes 512D unit-normalized biometric vectors, and saves them.")
    print("=" * 65)

    db = load_db()
    existing_count = len(db)
    print(f"Currently Enrolled Users: {existing_count}")
    for uid, data in db.items():
        print(f"  • {uid}: {data.get('display_name')} (INR {data.get('balance'):.2f})")
    print("-" * 65)

    while True:
        default_id = f"USR{len(load_db()) + 1:03d}"
        print(f"\n[ENROLL PERSON]")
        try:
            uid_input = input(f"Enter User ID (press Enter for '{default_id}'): ").strip()
            uid = uid_input if uid_input else default_id

            name_input = input("Enter Full Name (e.g., Niranjan or Friend): ").strip()
            name = name_input if name_input else f"User {uid}"

            bal_input = input("Enter Initial Wallet Balance in INR (default ₹1000): ").strip()
            try:
                balance = float(bal_input) if bal_input else 1000.0
            except ValueError:
                balance = 1000.0
        except (KeyboardInterrupt, EOFError):
            print("\nExiting enrollment wizard.")
            break

        print(f"\nLaunching camera for '{name}' ({uid})...")
        success = capture_and_enroll_interactive(uid, name, balance)

        if not success:
            print("Enrollment for this user was not completed.")

        try:
            more = input("\nDo you want to enroll your friend / another person now? (Y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if more in ("n", "no", "q", "quit"):
            break

    print("\n" + "=" * 65)
    final_db = load_db()
    print(f"Total Enrolled Users: {len(final_db)}")
    for uid, data in final_db.items():
        print(f"  ✓ {uid}: {data.get('display_name')} | ₹{data.get('balance'):.2f}")
    print("You can now start api_server.py and optipulse_engine.py!")
    print("=" * 65)


if __name__ == "__main__":
    interactive_wizard()

