

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# ── Config ──────────────────────────────────────────
MODEL_PATH       = 'weights/best.pt'
CONF_THRESHOLD   = 0.35
VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']
SAFE_CLASSES      = ['Hardhat', 'Safety Vest', 'Mask']

_model = None  # lazy load


def _get_model():
    """Load model once and reuse."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Download best.pt and place in weights/ folder."
            )
        _model = YOLO(MODEL_PATH)
    return _model


# ── Main function ────────────────────────────────────
def check_compliance(image_path: str,
                     conf: float = CONF_THRESHOLD,
                     show_image: bool = True) -> dict:
    """
    Check safety compliance for a single image.

    Args:
        image_path : path to image file
        conf       : confidence threshold (0.0–1.0)
        show_image : display annotated result

    Returns:
        report dict with keys:
            verdict, is_safe, violation_count,
            violations, compliant_items
    """
    model   = _get_model()
    results = model(image_path, conf=conf)[0]

    violations = []
    safe_items = []

    for box in results.boxes:
        cls_name   = model.names[int(box.cls)]
        confidence = round(float(box.conf), 2)

        if cls_name in VIOLATION_CLASSES:
            violations.append({
                'type'      : cls_name,
                'confidence': confidence,
                'bbox'      : box.xyxy[0].tolist()
            })
        elif cls_name in SAFE_CLASSES:
            safe_items.append({
                'type'      : cls_name,
                'confidence': confidence
            })

    is_safe = len(violations) == 0

    report = {
        'verdict'          : 'SAFE' if is_safe else 'UNSAFE',
        'is_safe'          : is_safe,
        'violation_count'  : len(violations),
        'violations'       : violations,
        'compliant_items'  : safe_items,
        'avg_violation_conf': (
            round(sum(v['confidence'] for v in violations)
                  / len(violations), 2)
            if violations else None
        )
    }

    if show_image:
        _visualise(results, report)

    return report


def batch_check(image_paths: list,
                conf: float = CONF_THRESHOLD) -> list:
    """
    Run compliance check on a list of images.

    Returns:
        list of report dicts, one per image
    """
    reports = []
    for path in image_paths:
        print(f"\nChecking: {os.path.basename(path)}")
        report = check_compliance(path, conf=conf, show_image=False)
        report['image_path'] = path
        reports.append(report)

    # Summary
    total    = len(reports)
    n_unsafe = sum(1 for r in reports if not r['is_safe'])
    print("\n" + "=" * 45)
    print(f"  BATCH SUMMARY")
    print("=" * 45)
    print(f"  Total images : {total}")
    print(f"  Safe         : {total - n_unsafe}")
    print(f"  Unsafe       : {n_unsafe}")
    print(f"  Violation rate: {round(n_unsafe/total*100, 1)}%")
    print("=" * 45)

    return reports


# ── Visualisation ────────────────────────────────────
def _visualise(results, report: dict):
    annotated     = results.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    color = 'green' if report['is_safe'] else 'red'
    title = f"Safety Monitor — {report['verdict']}"
    if not report['is_safe']:
        title += f"  |  {report['violation_count']} violation(s)"

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_rgb)
    plt.title(title, fontsize=13, color=color, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
