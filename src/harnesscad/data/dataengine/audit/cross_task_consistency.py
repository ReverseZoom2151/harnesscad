"""Cross-task class, box, and mask consistency checks."""

from __future__ import annotations
from harnesscad.data.dataengine.schemas.anomaly_schema import AnomalyAsset


def validate_asset(asset: AnomalyAsset) -> tuple[str, ...]:
    errors = []
    labels = set(asset.labels)
    for i, box in enumerate(asset.boxes):
        if not (0 <= box.x1 < box.x2 <= asset.width and
                0 <= box.y1 < box.y2 <= asset.height):
            errors.append(f"box_bounds:{i}")
        if labels and box.label not in labels:
            errors.append(f"box_label:{i}")
    for i, mask in enumerate(asset.masks):
        if len(mask.points) < 3:
            errors.append(f"mask_empty:{i}")
            continue
        if any(not (0 <= x <= asset.width and 0 <= y <= asset.height)
               for x, y in mask.points):
            errors.append(f"mask_bounds:{i}")
        if labels and mask.label not in labels:
            errors.append(f"mask_label:{i}")
        matching = [b for b in asset.boxes if b.label == mask.label]
        if matching:
            xs, ys = zip(*mask.points)
            if not any(b.x1 <= min(xs) and b.x2 >= max(xs) and
                       b.y1 <= min(ys) and b.y2 >= max(ys) for b in matching):
                errors.append(f"box_mask:{i}")
    return tuple(errors)
