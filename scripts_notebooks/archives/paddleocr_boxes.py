#!/usr/bin/env python3
"""PaddleOCR + geometric box ordering for multi-column pages (CPU-friendly)."""

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR and reorder lines by detected columns."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/mathieu/Documents/memoire/code_memoire/transcription/output/paddleocr_boxes",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--lang",
        default="fr",
        help="OCR language (default: fr).",
    )
    parser.add_argument(
        "--keep-model-source-check",
        action="store_true",
        help="Keep model source check enabled.",
    )
    return parser


def _bbox_to_xyxy(bbox: Any) -> Tuple[float, float, float, float]:
    # Paddle boxes can be [x1,y1,x2,y2] or polygon [[x,y],...].
    arr = np.asarray(bbox)
    if arr.ndim == 1 and arr.size == 4:
        x1, y1, x2, y2 = arr.tolist()
        return float(x1), float(y1), float(x2), float(y2)

    if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
        xs = arr[:, 0].astype(float).tolist()
        ys = arr[:, 1].astype(float).tolist()
        return min(xs), min(ys), max(xs), max(ys)

    pts = list(bbox)
    xs = [float(p[0]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    ys = [float(p[1]) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs or not ys:
        raise ValueError(f"Unsupported bbox format: {type(bbox)}")
    return min(xs), min(ys), max(xs), max(ys)


def _extract_lines(result_obj: Any) -> List[Dict[str, Any]]:
    texts = None
    boxes = None
    scores = None

    # 1) attribute-style access
    texts = getattr(result_obj, "rec_texts", None)
    boxes = getattr(result_obj, "rec_boxes", None)
    scores = getattr(result_obj, "rec_scores", None)

    # 2) mapping-style access
    if (texts is None or boxes is None) and hasattr(result_obj, "get"):
        try:
            texts = result_obj.get("rec_texts", texts)
            boxes = result_obj.get("rec_boxes", boxes)
            scores = result_obj.get("rec_scores", scores)
        except Exception:
            pass

    # 3) json container, often in result_obj.json["res"]
    if texts is None or boxes is None:
        try:
            j = getattr(result_obj, "json", None)
            if isinstance(j, dict):
                maybe = j.get("res", j)
                if isinstance(maybe, dict):
                    texts = maybe.get("rec_texts", texts)
                    boxes = maybe.get("rec_boxes", boxes)
                    scores = maybe.get("rec_scores", scores)
        except Exception:
            pass

    lines: List[Dict[str, Any]] = []
    if not isinstance(texts, list) or boxes is None:
        return lines

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            continue
        t = text.strip()
        if not t:
            continue
        box = boxes[i]
        x1, y1, x2, y2 = _bbox_to_xyxy(box)
        w = max(0.0, x2 - x1)
        h = max(1.0, y2 - y1)
        score = float(scores[i]) if scores is not None else None
        lines.append(
            {
                "index": i,
                "text": t,
                "score": score,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )
    return lines


def _column_boundaries(lines: List[Dict[str, Any]], img_w: float) -> List[Tuple[float, float]]:
    if len(lines) < 4:
        return [(0.0, img_w)]

    x_centers = sorted(l["cx"] for l in lines)
    gaps = [x_centers[i + 1] - x_centers[i] for i in range(len(x_centers) - 1)]
    if not gaps:
        return [(0.0, img_w)]

    med_gap = median(gaps)
    min_big_gap = max(img_w * 0.12, med_gap * 3.0)

    split_positions = []
    for i, g in enumerate(gaps):
        if g >= min_big_gap:
            split_positions.append((x_centers[i] + x_centers[i + 1]) / 2.0)

    if not split_positions:
        # Fallback: try 2-column clustering when there is no obvious large gap.
        try:
            x = np.array([[l["cx"]] for l in lines], dtype=np.float32)
            km = KMeans(n_clusters=2, random_state=0, n_init=10)
            labels = km.fit_predict(x)
            centers = sorted(float(c[0]) for c in km.cluster_centers_)
            sep = centers[1] - centers[0]
            n0 = int(np.sum(labels == np.argmin(km.cluster_centers_.flatten())))
            n1 = int(np.sum(labels == np.argmax(km.cluster_centers_.flatten())))
            min_cluster = max(4, int(0.12 * len(lines)))
            if sep >= 0.18 * img_w and n0 >= min_cluster and n1 >= min_cluster:
                split_positions = [(centers[0] + centers[1]) / 2.0]
            else:
                return [(0.0, img_w)]
        except Exception:
            return [(0.0, img_w)]

    cuts = [0.0] + split_positions + [img_w]
    bounds: List[Tuple[float, float]] = []
    for i in range(len(cuts) - 1):
        bounds.append((cuts[i], cuts[i + 1]))
    return bounds


def _assign_column(cx: float, bounds: List[Tuple[float, float]]) -> int:
    for idx, (l, r) in enumerate(bounds):
        if l <= cx < r:
            return idx
    return len(bounds) - 1


def reorder_lines_by_columns(lines: List[Dict[str, Any]], img_w: float) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]]]:
    if not lines:
        return [], [(0.0, img_w)]

    full_width_threshold = img_w * 0.72
    full_width = [l for l in lines if l["w"] >= full_width_threshold]
    body = [l for l in lines if l["w"] < full_width_threshold]

    body_min_y = min((l["cy"] for l in body), default=0.0)
    body_max_y = max((l["cy"] for l in body), default=0.0)
    med_h = median([l["h"] for l in lines]) if lines else 20.0

    top_full = sorted(
        [l for l in full_width if l["cy"] <= body_min_y + 0.6 * med_h],
        key=lambda x: (x["cy"], x["x1"]),
    )
    bottom_full = sorted(
        [l for l in full_width if l["cy"] >= body_max_y - 0.6 * med_h],
        key=lambda x: (x["cy"], x["x1"]),
    )

    middle_full = [
        l
        for l in full_width
        if l not in top_full and l not in bottom_full
    ]

    bounds = _column_boundaries(body, img_w)
    for l in body:
        l["column"] = _assign_column(l["cx"], bounds)

    ordered: List[Dict[str, Any]] = []
    ordered.extend(top_full)

    for col_id in range(len(bounds)):
        col_lines = [l for l in body if l.get("column") == col_id]
        col_lines.sort(key=lambda x: (x["cy"], x["x1"]))
        ordered.extend(col_lines)

    middle_full.sort(key=lambda x: (x["cy"], x["x1"]))
    ordered.extend(middle_full)
    ordered.extend(bottom_full)

    return ordered, bounds


def main() -> int:
    args = build_parser().parse_args()
    image_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"Image introuvable: {image_path}")
        return 1

    if not args.keep_model_source_check:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

    try:
        from PIL import Image
        from paddleocr import PaddleOCR
    except Exception as exc:
        print("Import impossible. Installe dans le venv:")
        print("  python -m pip install -U paddlepaddle paddleocr pillow")
        print(f"Détail: {exc}")
        return 2

    img_w, _ = Image.open(str(image_path)).size

    print(f"Image: {image_path}")
    print("Initialisation PaddleOCR...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang,
    )

    print("OCR en cours...")
    t0 = perf_counter()
    results = list(ocr.predict(str(image_path)))
    dt = perf_counter() - t0
    if not results:
        print("Aucun résultat renvoyé.")
        return 3

    first = results[0]
    lines = _extract_lines(first)
    ordered, bounds = reorder_lines_by_columns(lines, float(img_w))

    out_prefix = out_dir / image_path.stem
    raw_json = out_prefix.with_name(f"{out_prefix.name}_raw_lines.json")
    ordered_json = out_prefix.with_name(f"{out_prefix.name}_ordered_lines.json")
    txt_path = out_prefix.with_name(f"{out_prefix.name}_full_text.txt")
    meta_path = out_prefix.with_name(f"{out_prefix.name}_meta.json")

    raw_json.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    ordered_json.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    txt_path.write_text("\n".join(l["text"] for l in ordered), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "width": img_w,
                "n_lines_raw": len(lines),
                "n_lines_ordered": len(ordered),
                "column_bounds": bounds,
                "elapsed_seconds": round(dt, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Terminé en {dt:.1f}s")
    print(f"Colonnes détectées: {len(bounds)}")
    print(f"Sorties: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
