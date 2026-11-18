"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework."""
from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    stems_set = set(image_stems)
    reviewed_set = set(reviewed)
    n_images = len(stems_set)

    n_errors = sum(1 for i in issues if i.severity == ERROR)
    warn_images = {
        i.image
        for i in issues
        if i.severity == WARN and i.image in stems_set and i.image not in reviewed_set
    }
    n_warn_images = len(warn_images)
    warn_ratio = round(n_warn_images / n_images, 4) if n_images > 0 else 0.0
    hold_thresh = cfg["gate"]["warn_ratio_hold"]

    reasons = []
    if n_errors >= 1:
        decision = "Rework"
        reasons.append(f"còn {n_errors} lỗi bắt buộc (ERROR)")
    elif warn_ratio >= hold_thresh:
        decision = "Hold"
        reasons.append(
            f"tỉ lệ ảnh cảnh báo chưa duyệt {n_warn_images}/{n_images} ({warn_ratio:.2%}) >= ngưỡng {hold_thresh:.2%}"
        )
    else:
        decision = "Release"
        reasons.append(
            f"0 lỗi bắt buộc, tỉ lệ ảnh cảnh báo chưa duyệt {n_warn_images}/{n_images} ({warn_ratio:.2%}) < ngưỡng {hold_thresh:.2%}"
        )

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons,
    }
