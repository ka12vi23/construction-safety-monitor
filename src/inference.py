"""
Construction Safety Monitor — Inference Script
Usage: python src/inference.py --image path/to/image.jpg
"""

import argparse
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# ── Config ──────────────────────────────────────────
MODEL_PATH = 'weights/best.pt'
CONF_THRESHOLD = 0.35
VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']
SAFE_CLASSES      = ['Hardhat', 'Safety Vest', 'Mask', 'Person']


# ── Compliance logic ─────────────────────────────────
def run_inference(image_path: str, conf: float = CONF_THRESHOLD,
                  save_output: bool = False):
    """
    Run safety inference on a single image.
    Returns annotated image + compliance report dict.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(MODEL_PATH)
    results = model(image_path, conf=conf)[0]

    violations  = []
    safe_items  = []

    for box in results.boxes:
        cls_name   = model.names[int(box.cls)]
        confidence = round(float(box.conf), 2)

        if cls_name in VIOLATION_CLASSES:
            violations.append({'type': cls_name, 'confidence': confidence})
        elif cls_name in SAFE_CLASSES:
            safe_items.append({'type': cls_name, 'confidence': confidence})

    is_safe = len(violations) == 0

    report = {
        'verdict'         : 'SAFE'   if is_safe else 'UNSAFE',
        'is_safe'         : is_safe,
        'violation_count' : len(violations),
        'violations'      : violations,
        'compliant_items' : safe_items,
    }

    _print_report(report)
    annotated = _show_result(results, report, image_path, save_output)

    return report, annotated


# ── Helpers ──────────────────────────────────────────
def _print_report(report: dict):
    icon = '✅' if report['is_safe'] else '🚨'
    print("=" * 45)
    print("  SAFETY COMPLIANCE REPORT")
    print("=" * 45)
    print(f"  Verdict   : {icon} {report['verdict']}")
    print(f"  Violations: {report['violation_count']}")
    for v in report['violations']:
        print(f"    ⚠️  {v['type']}  (conf: {v['confidence']})")
    print(f"  Compliant : {len(report['compliant_items'])}")
    for s in report['compliant_items']:
        print(f"    ✅  {s['type']}  (conf: {s['confidence']})")
    print("=" * 45)


def _show_result(results, report: dict, image_path: str,
                 save_output: bool):
    import numpy as np

    annotated     = results.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    title_color = 'green' if report['is_safe'] else 'red'
    title_text  = f"Safety Monitor — {report['verdict']}"

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_rgb)
    plt.title(title_text, fontsize=14, color=title_color, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_output:
        out_path = image_path.replace('.jpg', '_result.jpg')
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved: {out_path}")

    plt.show()
    return annotated_rgb


# ── CLI entry point ───────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Construction Safety Monitor — single image inference'
    )
    parser.add_argument('--image',  required=True,
                        help='Path to input image')
    parser.add_argument('--conf',   type=float, default=CONF_THRESHOLD,
                        help='Confidence threshold (default: 0.35)')
    parser.add_argument('--save',   action='store_true',
                        help='Save annotated output image')
    args = parser.parse_args()

    run_inference(args.image, conf=args.conf, save_output=args.save)
