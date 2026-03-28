# Dataset Documentation

## Source

**Base Dataset:** Construction Site Safety Computer Vision Dataset  
**Source:** Roboflow Universe — roboflow-universe-projects  
**Link:** https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety  
**License:** Public (Roboflow Universe)

---

## Custom Extensions

Additional images were collected manually via Google Images searches:
- `"construction site workers safety"`
- `"construction worker no helmet"`
- `"building site PPE violation"`
- `"indoor construction workers"`

Images were selected to improve diversity across:
- Indoor vs outdoor environments
- Different lighting conditions (daylight, shadow, artificial)
- Safe and unsafe scenes (balanced)

All custom images were annotated using Roboflow's annotation tool
with the same class labels as the base dataset.

---

## Final Dataset Stats

| Split | Images |
|---|---|
| Training | 521 |
| Validation | 114 |
| Testing | 82 |
| **Total (after augmentation)** | **1,748** |

---

## Classes (25 total)

| Class | Type |
|---|---|
| Person | Worker detection |
| Hardhat | PPE compliant |
| NO-Hardhat | PPE violation |
| Safety Vest | PPE compliant |
| NO-Safety Vest | PPE violation |
| Mask | PPE compliant |
| NO-Mask | PPE violation |
| Safety Cone | Environment |
| Ladder | Environment |
| Excavator | Equipment |
| + 15 others | Vehicles/equipment |

---

## Annotation Strategy

- Bounding box annotations for all classes
- Tool used: Roboflow annotation interface
- All workers annotated regardless of distance
- Partially visible workers included where class is determinable
- Unannotated images (193) excluded from training split

---

## Preprocessing & Augmentation

| Step | Setting |
|---|---|
| Resize | 640 × 640 (Stretch) |
| Auto-Orient | On |
| Horizontal Flip | On |
| Brightness | ±15% |

---

## Diversity Coverage

| Dimension | Coverage |
|---|---|
| Indoor scenes |  Included |
| Outdoor scenes |  Included |
| Daylight | Included |
| Overcast/shadow |  Included |
| Artificial lighting |  Included |
| Safe scenes |  Included |
| Unsafe scenes |  Included |
| Multiple workers |  Included |
| Partial occlusion |  Included |
