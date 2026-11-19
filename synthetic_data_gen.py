#!/usr/bin/env python3
"""
synthetic_data_gen.py — OptiPulse-Pay Liveness Dataset Synthesizer
Generates training images for binary liveness classification:
  data/train/live/   — organic skin gradient patches with Lambertian reflectance
  data/train/spoof/  — presentation attack artifacts (LCD Moiré, print grain, screen replay)
  data/val/live/     — held-out live samples (20% split)
  data/val/spoof/    — held-out spoof samples (20% split)
"""

import os
import sys
import random
import math
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Configuration ──────────────────────────────────────────────────────────────
PATCH_SIZE    = 64          # Square patch dimensions (px)
N_LIVE        = 2000        # Number of live samples to generate
N_SPOOF       = 2000        # Number of spoof samples to generate
VAL_SPLIT     = 0.20        # Validation fraction
SEED          = 42

TRAIN_LIVE_DIR  = Path("data/train/live")
TRAIN_SPOOF_DIR = Path("data/train/spoof")
VAL_LIVE_DIR    = Path("data/val/live")
VAL_SPOOF_DIR   = Path("data/val/spoof")


# ── Lambertian Skin Reflectance Model ─────────────────────────────────────────
def _lambertian_reflectance(size: int, base_rgb: tuple[int,int,int],
                             light_dir: tuple[float,float,float],
                             albedo: float = 0.45) -> np.ndarray:
    """
    Generate a Lambertian-shaded skin patch.
    I(x,y) = albedo * max(0, N(x,y) · L) * base_color

    A smooth sinusoidal normal map approximates organic skin curvature.
    light_dir is a unit 3D vector [Lx, Ly, Lz].
    """
    img = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy = size / 2.0, size / 2.0
    Lx, Ly, Lz = light_dir

    for y in range(size):
        for x in range(size):
            # Organic normal perturbation via sinusoidal curvature
            nx = math.sin((x - cx) / size * math.pi)
            ny = math.sin((y - cy) / size * math.pi)
            nz = math.sqrt(max(0.0, 1.0 - nx*nx - ny*ny))
            # Lambertian diffuse intensity
            dot = max(0.0, nx*Lx + ny*Ly + nz*Lz)
            intensity = albedo * dot + 0.15  # ambient fill
            intensity = min(1.0, intensity)
            img[y, x, 0] = base_rgb[0] * intensity
            img[y, x, 1] = base_rgb[1] * intensity
            img[y, x, 2] = base_rgb[2] * intensity

    return np.clip(img, 0, 255).astype(np.uint8)


def _add_skin_texture(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add subtle organic skin micro-texture: fine Perlin-like noise + pore simulation."""
    size = patch.shape[0]
    # Multi-octave noise approximation (sum of scaled random noise)
    noise = np.zeros((size, size), dtype=np.float32)
    for octave in range(4):
        scale = 2 ** octave
        block = rng.standard_normal((size // scale + 1, size // scale + 1)).astype(np.float32)
        upsampled = np.array(Image.fromarray(block).resize((size, size), Image.BILINEAR))
        noise += upsampled / (2 ** (octave + 1))

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    texture_strength = rng.uniform(3.0, 8.0)
    for c in range(3):
        channel = patch[:, :, c].astype(np.float32)
        channel += (noise - 0.5) * texture_strength
        patch[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # Simulate sparse skin pores (dark micro-dots)
    n_pores = rng.integers(5, 20)
    for _ in range(n_pores):
        px, py = rng.integers(0, size, size=2)
        r = rng.integers(0, 2)
        intensity_drop = rng.uniform(15, 40)
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if 0 <= py+dy < size and 0 <= px+dx < size:
                    patch[py+dy, px+dx] = np.clip(
                        patch[py+dy, px+dx].astype(np.float32) - intensity_drop, 0, 255
                    ).astype(np.uint8)
    return patch


def generate_live_patch(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a single live skin patch with:
    - Randomized skin tone base (Fitzpatrick I–VI range)
    - Random lighting direction
    - Lambertian shading + organic micro-texture
    - Subtle Gaussian blur for natural softness
    """
    # Fitzpatrick scale skin tone bases (R, G, B)
    skin_tones = [
        (255, 224, 196), (240, 200, 168), (210, 168, 130),
        (180, 130, 100), (140, 90,  60),  (100, 60,  35),
    ]
    base_rgb = rng.choice(skin_tones)
    # Randomize hue slightly
    jitter = rng.integers(-12, 12, size=3)
    base_rgb = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(base_rgb, jitter))

    # Random light direction (upper hemisphere)
    theta = rng.uniform(0, math.pi / 3)
    phi   = rng.uniform(0, 2 * math.pi)
    light_dir = (
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    )

    albedo = rng.uniform(0.35, 0.55)
    patch  = _lambertian_reflectance(PATCH_SIZE, base_rgb, light_dir, albedo)
    patch  = _add_skin_texture(patch, rng)

    # Subtle Gaussian blur (simulates camera depth-of-field over skin)
    blur_radius = rng.uniform(0.4, 1.2)
    from PIL import ImageFilter
    pil = Image.fromarray(patch).filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    return np.array(pil)


# ── Spoof Artifact Generators ──────────────────────────────────────────────────
def _add_lcd_moire(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate LCD Moiré scanline pattern from screen-replay attacks.
    Horizontal scanlines with subpixel RGB fringing and periodic intensity dips.
    """
    result = patch.astype(np.float32).copy()
    size   = patch.shape[0]
    freq   = rng.uniform(2.5, 6.0)        # scanlines per 64px
    amplitude = rng.uniform(8.0, 20.0)   # intensity variation

    for y in range(size):
        # Primary scanline dip
        dip = amplitude * (0.5 + 0.5 * math.sin(2 * math.pi * freq * y / size))
        result[y, :, 0] -= dip * rng.uniform(0.8, 1.2)
        result[y, :, 1] -= dip * rng.uniform(0.6, 1.0)
        result[y, :, 2] -= dip * rng.uniform(0.9, 1.3)

        # Subpixel RGB column fringing
        for x in range(size):
            fringe = rng.uniform(-3.0, 3.0)
            result[y, x, int(x % 3)] += fringe

    # Add horizontal banding noise
    band_noise = rng.normal(0, 5, (size, 1, 3)).astype(np.float32)
    result += np.tile(band_noise, (1, size, 1))

    return np.clip(result, 0, 255).astype(np.uint8)


def _add_print_grain(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate printed photo attack artifacts:
    - Gaussian ink noise (CMYK halftone approximation)
    - Salt-and-pepper noise (ink dropout/oversaturation)
    - Paper fiber micro-texture
    """
    result = patch.astype(np.float32).copy()
    size   = patch.shape[0]

    # Gaussian ink noise (halftone grain)
    sigma = rng.uniform(4.0, 12.0)
    gauss_noise = rng.normal(0, sigma, patch.shape).astype(np.float32)
    result += gauss_noise

    # Salt-and-pepper noise (ink dropout / oversaturation)
    sp_prob = rng.uniform(0.01, 0.04)
    sp_mask = rng.random((size, size))
    salt_mask   = sp_mask > (1 - sp_prob / 2)
    pepper_mask = sp_mask < (sp_prob / 2)
    result[salt_mask]   = 255.0
    result[pepper_mask] = 0.0

    # Paper fiber texture: correlated low-frequency noise
    fiber_scale = rng.integers(4, 10)
    fiber_noise = rng.normal(0, 3, (size // fiber_scale + 1, size // fiber_scale + 1, 3))
    fiber_upsampled = np.array(
        Image.fromarray(fiber_noise.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    ).astype(np.float32)
    result += fiber_upsampled * rng.uniform(0.3, 0.8)

    return np.clip(result, 0, 255).astype(np.uint8)


def _add_screen_replay_artifacts(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate digital screen replay attack (tablet/phone display of a face photo).
    Combines LCD Moiré with color quantization and screen reflections.
    """
    result = _add_lcd_moire(patch, rng)

    # Color quantization (LCD pixel grid)
    quant_levels = rng.integers(32, 64)
    result = (result.astype(np.float32) / 255.0 * quant_levels).astype(np.int32)
    result = (result.astype(np.float32) / quant_levels * 255.0).astype(np.uint8)

    # Screen glare hot-spot
    cx = rng.integers(10, PATCH_SIZE - 10)
    cy = rng.integers(10, PATCH_SIZE - 10)
    glare_radius = rng.uniform(5, 15)
    glare_intensity = rng.uniform(20, 60)
    for y in range(PATCH_SIZE):
        for x in range(PATCH_SIZE):
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < glare_radius:
                glow = glare_intensity * (1.0 - dist / glare_radius)
                result[y, x] = np.clip(result[y, x].astype(np.float32) + glow, 0, 255).astype(np.uint8)

    return result


def _add_mask_artifact(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate 3D mask/silicone face attack:
    - Uniform low-frequency color with reduced micro-texture variance
    - Specular highlight (non-Lambertian flat specularity)
    - Edge seam artifact at patch border
    """
    result = patch.astype(np.float32).copy()
    size   = patch.shape[0]

    # Flatten micro-texture (masks have less surface detail)
    mean_color = result.mean(axis=(0, 1), keepdims=True)
    result = result * 0.3 + mean_color * 0.7  # blend toward flat color

    # Specular highlight (non-Lambertian — flat shine)
    spec_x = rng.integers(5, size - 5)
    spec_y = rng.integers(5, size - 5)
    spec_r = rng.uniform(3, 10)
    spec_intensity = rng.uniform(60, 120)
    for y in range(size):
        for x in range(size):
            dist = math.sqrt((x - spec_x)**2 + (y - spec_y)**2)
            if dist < spec_r:
                spec_val = spec_intensity * math.exp(-dist**2 / (2 * (spec_r/2)**2))
                result[y, x] = np.clip(result[y, x] + spec_val, 0, 255)

    # Edge seam (mask edge artifact)
    seam_width = rng.integers(2, 5)
    result[:seam_width, :]  = np.clip(result[:seam_width, :] * 0.5, 0, 255)
    result[-seam_width:, :] = np.clip(result[-seam_width:, :] * 0.5, 0, 255)

    return np.clip(result, 0, 255).astype(np.uint8)


def generate_spoof_patch(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a single spoof patch with randomly selected attack type:
    - 0: LCD Moiré (screen replay)
    - 1: Print grain (paper photo)
    - 2: Screen replay full composite
    - 3: Mask artifact
    """
    # Start from a semi-realistic skin base
    skin_tones = [
        (255, 224, 196), (240, 200, 168), (210, 168, 130),
        (180, 130, 100), (140, 90,  60),
    ]
    base_rgb = rng.choice(skin_tones)
    patch = np.full((PATCH_SIZE, PATCH_SIZE, 3), base_rgb, dtype=np.uint8)

    # Add flat gradient (no organic Lambertian curvature)
    for y in range(PATCH_SIZE):
        gradient = rng.uniform(-5, 5)
        patch[y, :] = np.clip(patch[y, :].astype(np.float32) + gradient, 0, 255).astype(np.uint8)

    attack_type = rng.integers(0, 4)
    if attack_type == 0:
        return _add_lcd_moire(patch, rng)
    elif attack_type == 1:
        return _add_print_grain(patch, rng)
    elif attack_type == 2:
        return _add_screen_replay_artifacts(patch, rng)
    else:
        return _add_mask_artifact(patch, rng)


# ── Dataset Generation ─────────────────────────────────────────────────────────
def generate_dataset(n_live: int, n_spoof: int, seed: int) -> None:
    """Generate the full liveness training and validation dataset."""
    rng = np.random.default_rng(seed=seed)
    random.seed(seed)

    for d in [TRAIN_LIVE_DIR, TRAIN_SPOOF_DIR, VAL_LIVE_DIR, VAL_SPOOF_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    n_val_live  = int(n_live  * VAL_SPLIT)
    n_val_spoof = int(n_spoof * VAL_SPLIT)
    n_tr_live   = n_live  - n_val_live
    n_tr_spoof  = n_spoof - n_val_spoof

    print("=" * 60)
    print("  OptiPulse-Pay — Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  Live  : {n_tr_live} train | {n_val_live} val")
    print(f"  Spoof : {n_tr_spoof} train | {n_val_spoof} val")
    print(f"  Patch : {PATCH_SIZE}x{PATCH_SIZE} px  |  Seed: {seed}")
    print()

    # Generate LIVE patches
    print("[1/4] Generating LIVE train patches...")
    for i in range(n_tr_live):
        patch = generate_live_patch(rng)
        Image.fromarray(patch).save(TRAIN_LIVE_DIR / f"live_{i:05d}.png")
        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{n_tr_live}")

    print("[2/4] Generating LIVE val patches...")
    for i in range(n_val_live):
        patch = generate_live_patch(rng)
        Image.fromarray(patch).save(VAL_LIVE_DIR / f"live_val_{i:05d}.png")

    # Generate SPOOF patches
    print("[3/4] Generating SPOOF train patches...")
    for i in range(n_tr_spoof):
        patch = generate_spoof_patch(rng)
        Image.fromarray(patch).save(TRAIN_SPOOF_DIR / f"spoof_{i:05d}.png")
        if (i + 1) % 200 == 0:
            print(f"      {i+1}/{n_tr_spoof}")

    print("[4/4] Generating SPOOF val patches...")
    for i in range(n_val_spoof):
        patch = generate_spoof_patch(rng)
        Image.fromarray(patch).save(VAL_SPOOF_DIR / f"spoof_val_{i:05d}.png")

    total = n_tr_live + n_val_live + n_tr_spoof + n_val_spoof
    print(f"\n✓ Dataset complete: {total} patches generated.")
    print(f"  Train live : {TRAIN_LIVE_DIR}")
    print(f"  Train spoof: {TRAIN_SPOOF_DIR}")
    print(f"  Val live   : {VAL_LIVE_DIR}")
    print(f"  Val spoof  : {VAL_SPOOF_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="OptiPulse-Pay Synthetic Liveness Dataset Generator"
    )
    parser.add_argument("--n-live",  type=int, default=N_LIVE,  help="Number of live samples")
    parser.add_argument("--n-spoof", type=int, default=N_SPOOF, help="Number of spoof samples")
    parser.add_argument("--seed",    type=int, default=SEED,    help="Random seed")
    args = parser.parse_args()
    generate_dataset(args.n_live, args.n_spoof, args.seed)


if __name__ == "__main__":
    main()
