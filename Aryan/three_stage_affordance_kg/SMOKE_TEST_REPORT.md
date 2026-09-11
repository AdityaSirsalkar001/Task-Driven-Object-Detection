# Smoke Test Report

## Scope

The pipeline was run on one image from each of the nine dataset task folders. One additional run used a natural-language prompt to test Qwen task normalization. These are smoke tests, not accuracy benchmarks.

| Task | Pipeline status | Returned detections | Time (seconds) |
| --- | --- | ---: | ---: |
| dig_hole | success | 6 | 35.4 |
| extinguish_fire | success | 4 | 21.6 |
| open_parcel | success | 9 | 49.5 |
| place_flowers | success | 5 | 39.4 |
| pour_sugar | success | 18 | 42.8 |
| sit_comfortably | success | 7 | 26.4 |
| smear_butter | success | 3 | 42.9 |
| step_on | success | 6 | 42.8 |
| wine | success after status fix | 25 | 39.5 |

## Manual Inspection

- `smear_butter` was the clearest example: bread, butter, and a knife were visible.
- `open_parcel` found visible scissors, but some parcel and box-cutter boxes appeared false.
- `place_flowers` found flowers and a vase, but broad aliases produced overlapping concept labels.
- Several other tasks produced false positive boxes. A pipeline `success` only means Florence returned a task-conditioned candidate, not that a human verified it.
- Florence provides no calibrated object confidence in the current interface. The system stores `od_confidence: null` and does not invent a final confidence score. This report was produced before the latest optional-reference filter: standard `<OD>` now adds supporting evidence but is not a hard veto. A canonical query or two overlapping same-family aliases can support an unscored result.

## Conclusion

All three software stages execute and save inspectable JSON and boxed images. Detection quality is not yet proven. Manual ground-truth annotations are required before reporting precision, recall, F1, Recall@1, Recall@3, or false positives per image.
