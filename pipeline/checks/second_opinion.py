"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l)."""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    c = cfg["second_opinion"]
    conf_thresh = c["conf"]
    min_area_px = c["min_area_px"]
    fp_conf = c["fp_conf"]
    iou_missing = c["iou_missing"]
    iou_match = c["iou_match"]
    iou_loose = c["iou_loose"]
    n_classes = len(ds.classes)

    issues = []
    for img_p in ds.image_files():
        stem = img_p.stem
        if stem not in preds:
            continue
        raw_lines = ds.read_label_lines(stem)
        if raw_lines is None:
            continue

        img_size = ds.image_size(stem)
        img_w, img_h = img_size if img_size else (0, 0)

        # Lấy danh sách nhãn hợp lệ kèm số dòng gốc
        valid_labels = []  # (line_no, cls_id, (cx, cy, w, h))
        for line_no, text in raw_lines:
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= cls_id < n_classes and w > 0 and h > 0:
                valid_labels.append((line_no, cls_id, (cx, cy, w, h)))

        img_preds = preds[stem]

        # 1. Duyệt từng dự đoán đạt ngưỡng conf và min_area_px
        for p in img_preds:
            p_cls, p_conf, px, py, pw, ph = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
            if p_conf < conf_thresh:
                continue
            area_px = (pw * img_w) * (ph * img_h)
            if area_px < min_area_px:
                continue

            p_box = (px, py, pw, ph)
            cands = []  # (iou, lbl_cls)
            for _, lbl_cls, lbl_box in valid_labels:
                iou = iou_xywh(p_box, lbl_box)
                if iou >= iou_missing:
                    cands.append((iou, lbl_cls))

            if not cands:
                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_MISSING",
                        stem,
                        f"model thấy lớp {p_cls} (conf={p_conf:.2f}) nhưng không có nhãn",
                        extra={"conf": round(p_conf, 4), "class": p_cls},
                    )
                )
                continue

            same_cls_ious = [iou for iou, lbl_cls in cands if lbl_cls == p_cls]
            if same_cls_ious:
                best_same = max(same_cls_ious)
                if best_same < iou_loose:
                    issues.append(
                        Issue(
                            "second_opinion",
                            WARN,
                            "POSSIBLE_LOOSE_BOX",
                            stem,
                            f"box nhãn khớp lỏng với dự đoán lớp {p_cls} (IoU={best_same:.2f} < {iou_loose})",
                            extra={"conf": round(p_conf, 4), "iou": round(float(best_same), 4)},
                        )
                    )
            else:
                best_diff = max(iou for iou, _ in cands)
                if best_diff >= iou_match:
                    issues.append(
                        Issue(
                            "second_opinion",
                            WARN,
                            "POSSIBLE_WRONG_CLASS",
                            stem,
                            f"model đoán lớp {p_cls} (conf={p_conf:.2f}, IoU={best_diff:.2f}) khác lớp nhãn",
                            extra={"conf": round(p_conf, 4), "iou": round(float(best_diff), 4), "pred_class": p_cls},
                        )
                    )

        # 2. Kiểm tra POSSIBLE_FALSE_POS cho từng dòng nhãn hợp lệ
        fp_preds = [
            (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            for p in img_preds
            if float(p[1]) >= fp_conf
        ]
        for line_no, lbl_cls, lbl_box in valid_labels:
            max_iou = max((iou_xywh(lbl_box, pb) for pb in fp_preds), default=0.0)
            if max_iou < iou_missing:
                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_FALSE_POS",
                        stem,
                        f"nhãn lớp {lbl_cls} không khớp dự đoán nào (max IoU={max_iou:.2f})",
                        line=line_no,
                        extra={"iou": round(float(max_iou), 4)},
                    )
                )

    return issues
