#!/usr/bin/env python3
"""Tesseract OCR + geometric column reconstruction (CPU-only)."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Tesseract OCR and reorder text lines by detected columns."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--out-dir",
        default="/Users/mathieu/Documents/memoire/code_memoire/transcription/output/tesseract_boxes",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--lang",
        default="fra",
        help="Tesseract language pack (default: fra).",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=3,
        help="Tesseract page segmentation mode (default: 3).",
    )
    parser.add_argument(
        "--tesseract-bin",
        default="/opt/homebrew/bin/tesseract",
        help="Absolute path to tesseract binary.",
    )
    return parser


def run_tesseract_tsv(
    image_path: Path, lang: str, psm: int, tesseract_bin: str
) -> str:
    cmd = [
        tesseract_bin,
        str(image_path),
        "stdout",
        "-l",
        lang,
        "--psm",
        str(psm),
        "tsv",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Tesseract failed")
    return proc.stdout


def parse_tsv_to_lines(tsv_text: str) -> List[Dict]:
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    words_by_line: Dict[Tuple[int, int, int, int], List[Dict]] = {}

    for row in reader:
        try:
            level = int(row["level"])
        except Exception:
            continue
        if level != 5:
            continue

        text = (row.get("text") or "").strip()
        if not text:
            continue

        conf_raw = row.get("conf", "-1")
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0
        if conf < 0:
            continue

        try:
            left = float(row["left"])
            top = float(row["top"])
            width = float(row["width"])
            height = float(row["height"])
        except Exception:
            continue

        key = (
            int(row["block_num"]),
            int(row["par_num"]),
            int(row["line_num"]),
            int(row["page_num"]),
        )
        words_by_line.setdefault(key, []).append(
            {
                "text": text,
                "conf": conf,
                "x1": left,
                "y1": top,
                "x2": left + width,
                "y2": top + height,
            }
        )

    lines = []
    for idx, key in enumerate(sorted(words_by_line.keys()), start=1):
        words = words_by_line[key]
        words.sort(key=lambda w: (w["x1"], w["y1"]))
        text = " ".join(w["text"] for w in words).strip()
        if not text:
            continue
        x1 = min(w["x1"] for w in words)
        y1 = min(w["y1"] for w in words)
        x2 = max(w["x2"] for w in words)
        y2 = max(w["y2"] for w in words)
        lines.append(
            {
                "index": idx,
                "line_key": key,
                "text": text,
                "score": float(np.mean([w["conf"] for w in words])),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": max(0.0, x2 - x1),
                "h": max(1.0, y2 - y1),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )

    return lines


def detect_column_bounds(lines: List[Dict], img_w: float) -> List[Tuple[float, float]]:
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
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def assign_column(cx: float, bounds: List[Tuple[float, float]]) -> int:
    for idx, (left, right) in enumerate(bounds):
        if left <= cx < right:
            return idx
    return len(bounds) - 1


def reorder_lines(lines: List[Dict], img_w: float) -> Tuple[List[Dict], List[Tuple[float, float]]]:
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
    middle_full = [l for l in full_width if l not in top_full and l not in bottom_full]

    bounds = detect_column_bounds(body, img_w)
    for l in body:
        l["column"] = assign_column(l["cx"], bounds)

    ordered: List[Dict] = []
    ordered.extend(top_full)
    for col_id in range(len(bounds)):
        col_lines = [l for l in body if l.get("column") == col_id]
        col_lines.sort(key=lambda x: (x["cy"], x["x1"]))
        ordered.extend(col_lines)
    middle_full.sort(key=lambda x: (x["cy"], x["x1"]))
    ordered.extend(middle_full)
    ordered.extend(bottom_full)
    return ordered, bounds


def make_output_stem(image_path: Path) -> str:
    """Build a unique output stem from the image relative path."""
    try:
        rel = image_path.resolve().relative_to(Path.cwd().resolve())
    except Exception:
        rel = image_path.name

    if isinstance(rel, Path):
        rel_no_suffix = rel.with_suffix("")
        parts = list(rel_no_suffix.parts)
        return "__".join(parts)
    return str(rel)


def main() -> int:
    args = build_parser().parse_args()
    image_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"Image introuvable: {image_path}")
        return 1

    if not Path(args.tesseract_bin).exists():
        print(f"Binaire tesseract introuvable: {args.tesseract_bin}")
        return 2

    img_w, _ = Image.open(str(image_path)).size

    print(f"Image: {image_path}")
    print("OCR Tesseract en cours...")
    t0 = perf_counter()
    try:
        tsv = run_tesseract_tsv(
            image_path, lang=args.lang, psm=args.psm, tesseract_bin=args.tesseract_bin
        )
    except Exception as exc:
        print(f"Echec Tesseract: {exc}")
        return 3
    lines = parse_tsv_to_lines(tsv)
    ordered, bounds = reorder_lines(lines, float(img_w))
    dt = perf_counter() - t0

    out_prefix = out_dir / make_output_stem(image_path)
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
                "tesseract_lang": args.lang,
                "psm": args.psm,
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
