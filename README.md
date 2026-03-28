# Construction Safety Monitor

An AI-powered computer vision system that automatically monitors construction sites 
and determines in real-time whether a situation is **safe or unsafe**.

Built for the Newnop Associate AI/ML Engineer assignment.

---

##  What It Does

- Detects workers in a scene regardless of distance or pose
- Recognises PPE: hard hats, safety vests, masks
- Flags safety violations with confidence scores
- Produces human-readable compliance reports

---

##  Model Performance

| Class | mAP50 |
|---|---|
| Hardhat | 0.887 |
| Safety Vest | 0.828 |
| NO-Safety Vest | 0.700 |
| Person | 0.735 |
| NO-Hardhat | 0.548 |
| **Overall** | **0.575** |

Trained on YOLOv8s — 50 epochs — Tesla T4 GPU — 0.46 hours

---

##  Safety Rules Defined

| Rule | Violation |
|---|---|
| Hard hat must be worn | NO-Hardhat detected |
| Safety vest must be worn | NO-Safety Vest detected |
| Mask required in zones | NO-Mask detected |

---

##  Dataset

- **Base:** Construction Site Safety dataset (Roboflow Universe)
- **Extended:** Custom additional images added manually
- **Total:** 1,748 images after augmentation
- **Split:** 521 train / 114 val / 82 test
- **Classes:** 25 (including Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest)
- **Augmentations:** Horizontal flip, ±15% brightness, 640×640 resize

---

##  Setup & Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Inference on an Image
```bash
python src/inference.py --image path/to/image.jpg
```

### Run Compliance Check
```python
from src.compliance_checker import check_compliance
report = check_compliance('path/to/image.jpg')
```

### Training (Google Colab)
Open the notebook:
```
notebooks/Construction_Safety_Monitor.ipynb
```

---

##  Project Structure
```
construction-safety-monitor/
├── README.md
├── requirements.txt
├── safety_rules.md
├── dataset_documentation.md
├── src/
│   ├── train.py
│   ├── inference.py
│   └── compliance_checker.py
└── notebooks/
    └── Construction_Safety_Monitor.ipynb
```

---
##  Innovation Beyond Baseline

This system goes beyond a simple safe/unsafe classifier in several ways:

### Confidence Scoring
Every violation includes a confidence score (0.0–1.0) rather than
just a binary flag. This surfaces model uncertainty so operators
can prioritise high-confidence alerts over borderline detections.

### Human-Readable Violation Reports
Each inference run produces a structured compliance report showing
exactly which PPE item is missing per worker — not just a scene-level
flag. Example output:
```
SAFETY COMPLIANCE REPORT
Verdict   : 🚨 UNSAFE
Violations: 2
  ⚠️  NO-Hardhat        (conf: 0.82)
  ⚠️  NO-Safety Vest    (conf: 0.91)
Compliant : 1
  ✅  Hardhat            (conf: 0.93)
```

### Batch Inference with Summary Stats
The `batch_check()` function processes entire folders of images
and returns a violation rate — useful for site-wide audits,
not just single-frame checks.

### Edge Case Handling
Documented failure modes for partial occlusion, distance, and
low-light scenes — with confidence threshold tuning to reduce
false positives in ambiguous cases.

##  Known Limitations

- Model struggles with distant or partially occluded workers
- NO-Hardhat detection weaker than Hardhat (mAP 0.548)
- Dataset skewed toward outdoor construction scenes
- Some false positives on non-construction PPE

---

##  Tech Stack

- **Model:** YOLOv8s (Ultralytics)
- **Training:** Google Colab T4 GPU
- **Dataset:** Roboflow Universe + custom extensions
- **Inference:** Python + OpenCV + Supervision
