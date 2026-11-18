"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN."""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line

DROP_LINE_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_IMG_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def fix(ds, issues, cfg):
    fixes = []
    removed_stems = set()

    # 1. Xử lý lỗi cấp dataset (data.yaml)
    has_bad_class_name = any(i.severity == ERROR and i.code == "BAD_CLASS_NAME" for i in issues)
    if has_bad_class_name:
        before_names = ds.yaml_names()
        new_data = {"path": ".", "names": dict(enumerate(cfg["classes"]))}
        if ds.yaml_path.exists():
            loaded = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                loaded["names"] = dict(enumerate(cfg["classes"]))
                new_data = loaded
        ds.yaml_path.write_text(yaml.safe_dump(new_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        fixes.append(
            {
                "image": "data.yaml",
                "line": "",
                "action": "rename_classes",
                "before": str(before_names),
                "after": str(dict(enumerate(cfg["classes"]))),
            }
        )

    # 2. Xử lý ảnh cần loại bỏ (DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE)
    for i in issues:
        if i.severity != ERROR:
            continue
        if i.code in REMOVE_IMG_CODES and i.image not in removed_stems:
            removed_stems.add(i.image)
            img_p = ds.image_path(i.image)
            if img_p and img_p.exists():
                img_p.unlink()
            lbl_p = ds.label_path(i.image)
            if lbl_p.exists():
                lbl_p.unlink()
            fixes.append(
                {
                    "image": i.image,
                    "line": "",
                    "action": "remove_image",
                    "before": i.code,
                    "after": "đã xoá ảnh và nhãn",
                }
            )

    # 3. Xử lý MISSING_LABEL_FILE (chuyển người gán, không tạo file rỗng)
    seen_missing = set()
    for i in issues:
        if i.severity == ERROR and i.code == "MISSING_LABEL_FILE":
            if i.image in removed_stems or i.image in seen_missing:
                continue
            seen_missing.add(i.image)
            fixes.append(
                {
                    "image": i.image,
                    "line": "",
                    "action": "needs_label",
                    "before": "(không có file)",
                    "after": "chuyển người gán",
                }
            )

    # 4. Gom lỗi theo từng file nhãn (drop_line ưu tiên cao hơn clip_box trên cùng một dòng)
    per_file_actions = defaultdict(dict)  # stem -> {line_no: action}
    for i in issues:
        if i.severity != ERROR or i.line is None:
            continue
        if i.image in removed_stems:
            continue
        if i.code in DROP_LINE_CODES:
            per_file_actions[i.image][i.line] = "drop_line"
        elif i.code == "BOX_OUT_OF_IMAGE":
            if per_file_actions[i.image].get(i.line) != "drop_line":
                per_file_actions[i.image][i.line] = "clip_box"

    for stem, line_actions in sorted(per_file_actions.items()):
        lbl_p = ds.label_path(stem)
        if not lbl_p.exists():
            continue
        raw_lines = lbl_p.read_text(encoding="utf-8", errors="replace").splitlines()
        new_lines = []

        for idx, raw_text in enumerate(raw_lines, 1):
            act = line_actions.get(idx)
            if act == "drop_line":
                fixes.append(
                    {
                        "image": stem,
                        "line": idx,
                        "action": "drop_line",
                        "before": raw_text.strip(),
                        "after": "",
                    }
                )
            elif act == "clip_box":
                try:
                    cls_id, cx, cy, w, h = parse_yolo_line(raw_text)
                    x1 = max(0.0, min(1.0, cx - w / 2.0))
                    y1 = max(0.0, min(1.0, cy - h / 2.0))
                    x2 = max(0.0, min(1.0, cx + w / 2.0))
                    y2 = max(0.0, min(1.0, cy + h / 2.0))
                    nw = x2 - x1
                    nh = y2 - y1
                except ValueError:
                    nw, nh = 0.0, 0.0

                if nw <= 0.0 or nh <= 0.0:
                    fixes.append(
                        {
                            "image": stem,
                            "line": idx,
                            "action": "drop_line",
                            "before": raw_text.strip(),
                            "after": "",
                        }
                    )
                else:
                    ncx = (x1 + x2) / 2.0
                    ncy = (y1 + y2) / 2.0
                    after_str = f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}"
                    new_lines.append(after_str)
                    fixes.append(
                        {
                            "image": stem,
                            "line": idx,
                            "action": "clip_box",
                            "before": raw_text.strip(),
                            "after": after_str,
                        }
                    )
            else:
                new_lines.append(raw_text)

        content = "\n".join(new_lines) + ("\n" if new_lines else "")
        lbl_p.write_text(content, encoding="utf-8")

    return fixes
