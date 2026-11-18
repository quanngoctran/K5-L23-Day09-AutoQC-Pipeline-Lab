"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp."""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    c = cfg["label_format"]
    tol = c["overflow_tol"]
    min_box_px = c["min_box_px"]
    max_box_frac = c["max_box_frac"]
    dup_iou = c["dup_box_iou"]
    n_classes = len(ds.classes)

    issues = []
    for lf in ds.label_files():
        stem = lf.stem
        lines = ds.read_label_lines(stem)
        if not lines:
            continue

        img_size = ds.image_size(stem)
        seen_boxes = []  # list[(cls_id, cx, cy, w, h)] của các dòng hợp lệ trước đó

        for line_no, raw_text in lines:
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(raw_text)
            except ValueError as e:
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, f"sai định dạng: {e}", line=line_no))
                continue

            if not (0 <= cls_id < n_classes):
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem, f"id lớp {cls_id} ngoài 0..{n_classes - 1}", line=line_no))
                continue

            if w <= 0 or h <= 0:
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem, f"kích thước box <= 0 (w={w}, h={h})", line=line_no))
                continue

            # Kiểm tra tràn ảnh (mép vượt [0, 1] quá overflow_tol)
            x1, y1 = cx - w / 2.0, cy - h / 2.0
            x2, y2 = cx + w / 2.0, cy + h / 2.0
            overflow = max(0.0 - x1, 0.0 - y1, x2 - 1.0, y2 - 1.0)
            if overflow > tol:
                issues.append(
                    Issue(
                        "label_format",
                        ERROR,
                        "BOX_OUT_OF_IMAGE",
                        stem,
                        f"box tràn khỏi ảnh mức {overflow:.4f}",
                        line=line_no,
                        extra={"overflow": round(float(overflow), 6)},
                    )
                )

            # Kiểm tra box trùng lặp với dòng hợp lệ trước đó (cùng lớp, IoU > dup_box_iou)
            box = (cx, cy, w, h)
            is_dup = any(prev_cls == cls_id and iou_xywh(prev_box, box) > dup_iou for prev_cls, prev_box in seen_boxes)
            if is_dup:
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem, "box trùng lặp với dòng trước đó", line=line_no))
            seen_boxes.append((cls_id, box))

            # Kiểm tra cảnh báo box quá nhỏ / quá lớn
            if img_size is not None:
                img_w, img_h = img_size
                short_side_px = min(w * img_w, h * img_h)
                if short_side_px < min_box_px:
                    issues.append(
                        Issue(
                            "label_format",
                            WARN,
                            "BOX_TINY",
                            stem,
                            f"cạnh ngắn box {short_side_px:.1f}px < {min_box_px}px",
                            line=line_no,
                            extra={"short_px": round(float(short_side_px), 2)},
                        )
                    )

            if w * h > max_box_frac:
                issues.append(
                    Issue(
                        "label_format",
                        WARN,
                        "BOX_HUGE",
                        stem,
                        f"diện tích box {w * h:.3f} > {max_box_frac}",
                        line=line_no,
                        extra={"frac": round(float(w * h), 4)},
                    )
                )

    return issues
