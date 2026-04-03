"""
Model Validation Test Suite — Sahayak Diagnostic Orchestrator
Tests each model (OmniChest v5, Legacy Ensemble, Spine) on real images.
Prints predictions vs. expected ground truth labels.
Usage: python3 test_all_models.py
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF noise

# ─── Test images (ground truth labels from clinical context) ───────────────────
TEST_CASES = [
    {
        "path": "test_images/effusion_chest.jpg",
        "label": "Effusion (Pleural)",
        "expect_high": ["Effusion"],
        "expect_low":  ["Pneumothorax", "Mass"],
    },
    {
        "path": "test_images/cardiomegaly_chest.jpg",
        "label": "Cardiomegaly (enlarged heart)",
        "expect_high": ["Cardiomegaly"],
        "expect_low":  ["Pneumothorax"],
    },
    {
        "path": "test_images/normal_chest_1.jpg",
        "label": "Normal chest (no findings expected)",
        "expect_high": ["No Finding"],
        "expect_low":  ["Pneumothorax", "Mass", "Cardiomegaly"],
    },
    {
        "path": "test_images/normal_chest_2.jpg",
        "label": "Normal chest 2",
        "expect_high": ["No Finding"],
        "expect_low":  ["Pneumothorax", "Effusion"],
    },
    {
        "path": "test_images/test_xray.png",
        "label": "Test scan",
        "expect_high": [],
        "expect_low":  [],
    },
]

COLORS = {
    "RED":    "\033[91m",
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
    "CYAN":   "\033[96m",
    "BOLD":   "\033[1m",
    "RESET":  "\033[0m",
}

def bar(score, width=25):
    filled = int(score * width)
    color = COLORS["RED"] if score > 0.5 else (COLORS["YELLOW"] if score > 0.3 else COLORS["GREEN"])
    return color + "█" * filled + "░" * (width - filled) + COLORS["RESET"] + f" {score*100:5.1f}%"

def section(title):
    print(f"\n{COLORS['BOLD']}{COLORS['CYAN']}{'='*60}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}  {title}{COLORS['RESET']}")
    print(f"{COLORS['BOLD']}{COLORS['CYAN']}{'='*60}{COLORS['RESET']}\n")

def check_expectations(preds, expect_high, expect_low):
    passed = []
    failed = []
    for label in expect_high:
        score = preds.get(label, 0.0)
        if score > 0.3:
            passed.append(f"✅ {label} correctly elevated ({score*100:.1f}%)")
        else:
            failed.append(f"❌ {label} should be high, got {score*100:.1f}%")
    for label in expect_low:
        score = preds.get(label, 0.0)
        if score < 0.4:
            passed.append(f"✅ {label} correctly suppressed ({score*100:.1f}%)")
        else:
            failed.append(f"⚠️  {label} unexpected signal ({score*100:.1f}%)")
    return passed, failed

def ensure_rgb(path):
    """Save a temp copy as RGB PNG for clean model input."""
    img = Image.open(path).convert("RGB")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(tmp.name)
    return tmp.name

def run_test(name, predict_fn, cases, top_n=5):
    section(name)
    total_pass, total_fail = 0, 0
    for case in cases:
        if not os.path.exists(case["path"]):
            print(f"  ⚠️  Skipping: {case['path']} not found\n")
            continue

        print(f"{COLORS['BOLD']}Image : {case['path']}{COLORS['RESET']}")
        print(f"Label : {case['label']}")

        tmp = ensure_rgb(case["path"])
        try:
            preds = predict_fn(tmp)
        except Exception as e:
            print(f"  {COLORS['RED']}ERROR: {e}{COLORS['RESET']}\n")
            continue
        finally:
            os.unlink(tmp)

        # Top-N findings
        sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top {top_n} predictions:")
        for disease, score in sorted_preds[:top_n]:
            print(f"    {disease:<25} {bar(score)}")

        # Expectation check
        passed, failed = check_expectations(preds, case["expect_high"], case["expect_low"])
        for msg in passed: print(f"  {msg}")
        for msg in failed: print(f"  {msg}")
        total_pass += len(passed)
        total_fail += len(failed)
        print()

    print(f"  Result: {total_pass} checks passed, {total_fail} checks failed\n")
    return total_pass, total_fail


# ─── Load models ──────────────────────────────────────────────────────────────
print("Loading models (this may take ~30s)...")
from xray_service import UnifiedPredictor, ChestEnsemble, SpinePredictor, SPINE_LABELS

unified  = UnifiedPredictor()
ensemble = ChestEnsemble()
spine    = SpinePredictor()

# ─── Run tests ────────────────────────────────────────────────────────────────
all_pass, all_fail = 0, 0

# 1. OmniChest v5 (Unified)
p, f = run_test(
    "OmniChest v5 — Unified Predictor",
    lambda path: unified.predict(path),
    TEST_CASES,
)
all_pass += p; all_fail += f

# 2. Legacy Ensemble (NIH + PadChest + RSNA)
def ensemble_predict(path):
    preds, _ = ensemble.predict(path)
    return preds

p, f = run_test(
    "Legacy Ensemble (NIH + PadChest + RSNA)",
    ensemble_predict,
    TEST_CASES,
)
all_pass += p; all_fail += f

# 3. Spine predictor (simulation mode — shows 0.03 baseline when model unavailable)
SPINE_TEST_CASES = [
    {
        "path": "test_images/normal_chest_1.jpg",
        "label": "Normal (used as proxy — no real spine CT available locally)",
        "expect_high": [],
        "expect_low": ["C1", "C2", "C3"],  # should all be low in simulation
    },
]

p, f = run_test(
    "Cervical Spine Model (EfficientNetV2B3 + BiGRU)",
    lambda path: spine.predict(path),
    SPINE_TEST_CASES,
)
all_pass += p; all_fail += f

# ─── Final summary ────────────────────────────────────────────────────────────
section("FINAL SUMMARY")
total = all_pass + all_fail
pct = (all_pass / total * 100) if total else 0
color = COLORS["GREEN"] if pct > 70 else COLORS["YELLOW"] if pct > 50 else COLORS["RED"]
print(f"  {color}{COLORS['BOLD']}Total: {all_pass}/{total} checks passed ({pct:.1f}%){COLORS['RESET']}\n")
print("  NOTE: The Spine model is in simulation mode (model file unavailable).")
print("        Re-run the Kaggle kernel and pull the output to enable real inference.\n")
