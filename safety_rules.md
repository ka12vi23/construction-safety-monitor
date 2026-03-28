# Safety Rules Definition

## Rule 1 — Hard Hat Required
**Violation:** Worker detected without a hard hat (NO-Hardhat)
**Confidence threshold:** 0.35
**Applies to:** All workers in any construction zone

Examples of violations:
- Worker with bare head on site
- Worker with hard hat removed while working
- Hard hat worn backwards or incorrectly

---

## Rule 2 — Safety Vest Required
**Violation:** Worker detected without high-visibility vest (NO-Safety Vest)
**Confidence threshold:** 0.35
**Applies to:** All active work zones

Examples of violations:
- Worker in plain clothing with no vest
- Vest present but not worn (held in hand)
- Vest worn open/unfastened

---

## Rule 3 — Mask Required in Dusty Zones
**Violation:** Worker detected without mask (NO-Mask)
**Confidence threshold:** 0.35
**Applies to:** Indoor construction, demolition, dusty environments

Examples of violations:
- Worker without face covering in dust zone
- Mask pulled down below chin

---

## Compliance Logic

A scene is marked **SAFE** when:
- Zero violation-class detections above confidence threshold

A scene is marked **UNSAFE** when:
- One or more of: NO-Hardhat, NO-Safety Vest, NO-Mask detected

Each violation includes:
- Violation type
- Confidence score (0.0 - 1.0)
- Bounding box location

---

## Known Edge Cases

| Situation | System Behaviour |
|---|---|
| Worker partially occluded | May miss violation — flagged as limitation |
| Worker far from camera | Lower confidence score, may be filtered |
| Hard hat in hand, not worn | May not detect as violation |
| Multiple workers in frame | Each assessed independently |
