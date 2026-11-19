# OptiPulse-Pay 💳

> **Autonomous Edge Facial Micro-Payment Gateway**
> Active Presentation Attack Detection · Zero-Hardware-Overhead KYC · Sub-700ms Verification

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  optipulse_engine.py  (POS Terminal FSM)                │
│  ┌──────────┐  ┌───────────────────┐  ┌─────────────┐  │
│  │ RGB Cam  │→ │  MediaPipe Mesh   │→ │  512D Vec   │  │
│  └──────────┘  └───────────────────┘  └──────┬──────┘  │
│                                               │ POST    │
│  ┌──────────────────┐   ┌────────────────────▼──────┐  │
│  │ Optical Challenge│   │  api_server.py (FastAPI)  │  │
│  │ EAR Blink PAD    │   │  1:N Cosine Match τ≥0.72  │  │
│  └──────────────────┘   └───────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
mini-project/
├── requirements.txt            # Pinned Python dependencies
├── enroll_kyc.py               # KYC portrait → 512D vector enrollment
├── api_server.py               # FastAPI payment gateway (port 8000)
├── synthetic_data_gen.py       # Liveness training dataset generator
├── train_liveness_model.py     # MobileNetV3-Small binary classifier trainer
├── optipulse_engine.py         # Main POS terminal application (FSM)
├── kyc_database/               # KYC portrait images + enrolled_users.json
├── models/                     # Trained model weights (.pth)
├── data/
│   ├── train/live/             # Synthetic live skin patches
│   ├── train/spoof/            # Synthetic spoof attack patches
│   ├── val/live/
│   └── val/spoof/
└── logs/
    └── transactions.log        # Audit trail
```

## Prerequisites

- **Python 3.10+**
- **Webcam** (built-in or USB RGB camera)
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **RAM**: 4 GB minimum (8 GB recommended)
- **GPU**: Optional (CUDA accelerates model inference; CPU works fine)

---

## Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note (Windows + CUDA):** For GPU acceleration replace the torch line with the CUDA wheel:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Step-by-Step Usage

### Step 1 — Enroll KYC Users (You & Your Friend)

You have two simple ways to enroll:

#### Method A: Live Interactive Camera Capture (Recommended)
Use the interactive camera wizard to capture high-resolution portraits for yourself and your friend:
```bash
python capture_enroll.py
```
*(or `python enroll_kyc.py --capture`)*

- Opens live camera with an alignment oval guide and real-time mesh detection.
- Press **[SPACE]** or click **[CAPTURE PHOTO]** to capture your face and generate your 512D KYC vector.
- Automatically prompts you to capture your friend's face right after!
- If `api_server.py` is running, it automatically hot-reloads the ledger on the fly.

#### Method B: Batch Enrollment from Image Files
Place standard 2D portrait JPEG/PNG images in `kyc_database/`.
Name files using the convention: `<USER_ID>__<Full_Name>.jpg`

```
kyc_database/
  USR001__Niranjan.jpg
  USR002__Friend_Name.png
```

Run batch enrollment:
```bash
python enroll_kyc.py
```

Options:
```
--kyc-dir PATH    Directory containing portrait images (default: kyc_database/)
--capture         Launch interactive camera capture wizard
--reset           Wipe existing enrollments and re-process all images
```

This extracts 512D L2-normalized biometric vectors using:
- **170 structural MediaPipe landmarks** (jaw, brows, eyes, nose, lips, cheeks)
- **Nasal bridge anchoring** (landmark 1) for translation invariance
- **Interocular distance normalization** (landmarks 33→263) for scale invariance
- **2 holistic ratio features** (face height + jaw width ratios)

Saved to `kyc_database/enrolled_users.json` and individual `.npy` vector files.

---

### Step 2 — Generate Synthetic Training Data

```bash
python synthetic_data_gen.py
```

Options:
```
--n-live  N     Number of live patches (default: 2000)
--n-spoof N     Number of spoof patches (default: 2000)
--seed    N     Random seed (default: 42)
```

Generates 64×64 px skin patches:
- **Live**: Lambertian reflectance model, Fitzpatrick I–VI skin tones, organic micro-texture, pore simulation
- **Spoof**: LCD Moiré scanlines, Gaussian/salt-and-pepper print grain, screen glare, 3D mask artifacts

---

### Step 3 — Train Liveness Model

```bash
python train_liveness_model.py
```

Options:
```
--epochs     N     Training epochs (default: 25)
--batch-size N     Batch size (default: 64)
--lr         F     Learning rate (default: 0.001)
--no-pretrain      Train from scratch (no ImageNet weights)
```

Trains a **MobileNetV3-Small** binary classifier:
- Input: 64×64 RGB skin patch
- Output: scalar ∈ [0, 1] — `> 0.5` = LIVE
- Optimizer: AdamW + Cosine Annealing LR
- Best checkpoint saved to `models/optipulse_liveness_mobilenet.pth`

---

### Step 4 — Launch the API Server

Open **Terminal 1**:
```bash
python api_server.py
```

The gateway starts at `http://127.0.0.1:8000`.

Verify with:
```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "enrolled_users": 2,
  "server_time": "2026-11-03T09:38:00Z",
  "version": "1.0.0"
}
```

Admin endpoints:
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/docs` | GET | Interactive Swagger UI |
| `/api/v1/authenticate_payment` | POST | Biometric payment auth |
| `/api/v1/ledger` | GET | View all enrolled users + balances |
| `/api/v1/reload_ledger` | POST | Hot-reload DB without restart |

---

### Step 5 — Launch the POS Terminal

Open **Terminal 2**:
```bash
python optipulse_engine.py
```

The terminal opens a dual-canvas window:
- **Left panel**: Live camera feed with MediaPipe face mesh overlay, bounding box, and skin ROI markers
- **Right panel**: Dark-mode POS dashboard with EAR score, liveness score, status, and result

#### POS Controls & Autonomous Mode
| Key / Input | Action |
|-------------|--------|
| **Autonomous (Hands-Free)** | **Stand in front of the camera** — Payment initializes automatically after ~0.4s! |
| `P` | Manual trigger override (initiates payment immediately) |
| `ENTER` / `SPACE` / **Mouse Click** | **PROCEED / OK**: Confirms and executes final payment in the confirmation stage |
| `ESC` / `C` / **Mouse Click** | **CANCEL**: Cancels pending payment and returns to Standby |
| `Q` | Quit application |
| `R` | Reset terminal to IDLE state |

---

## End-to-End System Flow (Matching Architectural Blueprint)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Webcam Video Feed                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│          Real-Time Face Detection & Alignment (MediaPipe Face Mesh)         │
└───────────────────┬─────────────────────────────────────┬───────────────────┘
                    │                                     │
                    ▼                                     ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│        Active Blink Analysis         │  │      Optical Screen Challenge     │
│   • Eye-Aspect-Ratio (EAR) tracking  │  │  • Complementary 2-phase flash    │
│   • Physiological micro-blinks       │  │  • Lambertian diffuse reflectance │
│   • Continuous 3D/grid monitoring    │  │  • Specular glass glare rejection │
└───────────────────┬──────────────────┘  └───────────────────┬───────────────┘
                    │                                         │
                    └───────────────────┬─────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  Liveness Verification Gate │
                         └──────┬───────────────┬──────┘
                                │               │
                        [PASSED]│               │[FAILED]
                                │               ▼
                                │    ╔════════════════════════════════════════╗
                                │    ║    [🔒] SECURITY LOCKOUT              ║
                                │    ║    (REPLAY ATTACK DETECTED)            ║
                                │    ║    • Red Lockout screen & camera frame ║
                                │    ║    • Biometric pipeline aborted        ║
                                │    ║    • Video replays & screens locked    ║
                                │    ╚════════════════════════════════════════╝
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      512D Biometric Vector Extraction                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Cosine Similarity Match (Against KYC Database)              │
│                          Threshold τ ≥ 0.65                                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Confirm Payment (Proceed Stage)                         │
│   • Displays: User Name, Amount (₹150), Current & New Wallet Balance        │
│   • Action: Click '[ PROCEED / OK ]' or press ENTER / SPACE to confirm      │
│   • Cancel: Click '[ CANCEL ]' or press ESC / C                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Authorizing (Wallet Debit)                          │
│   • Asynchronous REST API call to /api/v1/authenticate_payment              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Result: PAID ₹150 | User: <Name>                           │
│   • Green receipt card with updated wallet balance and unique TXN ID        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Tier Presentation Attack Detection (PAD)

### Tier 1 — Two-Phase Synchronized Spectral Challenge-Response (Defeats Video Replays)
The terminal executes an unpredictable **Two-Phase orthogonal color sequence** (e.g. Cyan `[255, 255, 0]` $\to$ Magenta `[255, 0, 255]`) for 560ms:
- **Physical principle**: Real diffuse human skin absorbs and reflects incident light in real time, shifting color along the emitted challenge transition vector ($\rho_{\text{transition}} \ge 0.32$).
- **Defeating Video Replays**: A pre-recorded video playing on a phone has autonomous, pre-recorded pixel dynamics that cannot match the unpredictable millisecond screen flash sequence. Furthermore, strict channel sign checks ensure that emitted increases and decreases (e.g., Red delta $\ge -0.35$ and Green delta $\le 0.35$) must physically match the light. Any video replay immediately triggers **Security Lockout (`REPLAY ATTACK DETECTED`)**.

### Tier 1B — Lambertian Spatial Gradient (Central Axis vs Lateral Cheeks)
- Under frontal screen illumination, facial surface normal facing the camera (nose tip and bridge) exhibits higher diffuse reflection than the lateral cheeks curving away from the camera.
- Flat smartphone screens and paper printouts have uniform planar normals and fail spatial coherence.

### Tier 2 — 3D Anatomical Depth Topography (Defeats Flat 2D Screens & Photos)
Uses 468 MediaPipe landmarks to measure 3D volumetric face curvature:
- **Nose Protrusion Ratio**: $(\text{cheeks}_z - \text{nose}_z) / \text{IOD} \ge 0.40$.
- **3D Non-Planar SVD Residual**: Mean perpendicular distance to best-fit plane $\ge 0.050 \cdot \text{IOD}$.
- Flat phone displays, tablets, and printed cutouts lie on a 2D plane and are immediately rejected (`SPOOF_2D_PLANAR_DETECTED`).

### Tier 3 — Multi-Patch 2D FFT Moiré & Subpixel Grid Detection (Defeats OLED/LCD Displays)
Analyzes spatial frequencies and chromatic gradients across multiple facial regions (forehead landmark 151, left cheek 117, right cheek 346):
- **High-Frequency Peak-to-Median Ratio (PMR)**: Detects periodic subpixel lattice spikes ($\le 1.65$).
- **Chrominance Gradient Variance**: Detects subpixel color alternation noise ($\le 380.0$). Real human skin has smooth organic chrominance ($\le 50$), while OLED/LCD screens exhibit values $\ge 7,000$, instantly triggering `SPOOF_MOIRE_GRID_DETECTED`.

### Tier 4 — Specular Glass Glare Hotspot Detection
Detects specular glass reflections ($> 238$ clipping) across the face bounding box ($\le 3.0\%$). Real human skin exhibits matte Lambertian diffusion; glass surfaces produce localized white glare spots (`SPOOF_GLASS_GLARE_DETECTED`).

### Tier 5 — Biological Blink Kinematics with Continual Monitoring
Requires voluntary biological eyelid movement (EAR drops below 0.225 or $25\%$ below open baseline, recovering above 0.250). Continual 3D topography, Moiré grid, and glass glare monitoring on every single frame of the blink window prevents screen substitution attacks during eye closure.

---

## Biometric Vector Mathematics

The 512D enrollment and query vectors are computed identically in both `enroll_kyc.py` and `optipulse_engine.py`:

```
coords[i] = [x_i, y_i, z_i]  (pixel space, 478 landmarks)

anchor    = coords[1]          # nasal bridge
coords   -= anchor             # translation-invariant

IOD       = ||coords[33] - coords[263]||   # interocular distance
coords   /= IOD                # scale-invariant

geo_vec   = flatten(coords[ALL_170_INDICES])  → 510D
ratio_vec = [||coords[152]-coords[10]|| , ||coords[234]-coords[454]||]  → 2D

v_512     = concat(geo_vec, ratio_vec)  → 512D
v_unit    = v_512 / ||v_512||₂          → L2-normalized
```

Cosine similarity via dot product (both vectors unit-normalized):
```
sim(q, e) = q · e  ∈ [-1, 1]
MATCH if sim ≥ τ = 0.72
```

---

## API Payload Reference

### POST `/api/v1/authenticate_payment`

**Request:**
```json
{
  "query_vector": [0.023, -0.041, ...],   // 512 floats, L2-normalized
  "transaction_amount": 150.0,
  "terminal_id": "TERM-POS-001"
}
```

**Response (AUTHORIZED):**
```json
{
  "status": "AUTHORIZED",
  "reason": "IDENTITY_VERIFIED",
  "user_id": "USR001",
  "display_name": "Alice Kumar",
  "similarity_score": 0.8821,
  "remaining_balance": 850.0,
  "transaction_id": "TXN-018F6A2C3D5E1000",
  "terminal_id": "TERM-POS-001",
  "server_timestamp": "2026-11-03T09:41:00Z",
  "hmac_signature": "a3f9..."
}
```

**Response (DECLINED):**
```json
{
  "status": "DECLINED",
  "reason": "BIOMETRIC_MISMATCH",
  "similarity_score": 0.5103,
  "transaction_id": "TXN-...",
  ...
}
```

---

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `No face detected in enrollment` | Use well-lit, front-facing portrait; avoid extreme angles |
| `API_UNREACHABLE` on terminal | Start `api_server.py` first in a separate terminal |
| `Liveness model not found` | Run `synthetic_data_gen.py` then `train_liveness_model.py` |
| Camera not opening | Change `cv2.VideoCapture(0)` index to `1` or `2` |
| Optical challenge always fails | Ensure room lighting isn't blocking monitor glow on face |
| Low EAR detected incorrectly | Re-adjust lighting; avoid glasses with thick frames |
| `INSUFFICIENT_FUNDS` | Top up via `/api/v1/reload_ledger` after editing JSON balance |

---

## Security Notes

- All API responses are HMAC-SHA256 signed for tamper detection
- Biometric vectors are stored as `.npy` files locally — never transmitted raw beyond localhost
- Transaction audit log is appended to `logs/transactions.log`
- The system operates entirely offline after enrollment — no cloud dependency

---

## License

MIT License — For research and educational purposes. Not intended for production financial transactions without formal security audit and regulatory compliance review.
