#!/usr/bin/env python3
"""Orchestrateur : extrait les crops + transcriptions GT depuis l'export
Label Studio, en s'appuyant sur les images sources de
data/benchmark_ocr/images_dataset/.

Pour chaque tâche annotée, le script :
  1. Identifie l'image source (le file_upload Label Studio est préfixé d'un
     hash de 8 caractères ; on le strip pour retomber sur page_NNNN.png).
  2. Désambigüise la revue d'origine via les dimensions (original_width,
     original_height) renvoyées par Label Studio (les 3 revues partagent les
     mêmes noms de fichier page_NNNN.png).
  3. Pour chaque région annotée (bbox + label + transcription), convertit
     les coordonnées (en pourcentages dans LS) en pixels, ajoute une marge,
     crop l'image native et écrit :
        crops/<revue>/<page_stem>__<region_id>.png   (l'image cropée)
        crops/<revue>/<page_stem>__<region_id>.txt   (la transcription GT)
        crops/<revue>/<page_stem>__<region_id>.json  (métadonnées)

Usage :
    python extract_crops.py \\
        --label-studio-json data/benchmark_ocr/project-16-at-XXX.json \\
        --images-dataset    data/benchmark_ocr/images_dataset \\
        --out-dir           data/benchmark_ocr/crops \\
        [--margin-px 15]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


# Hash LS = 8 hex + dash, ex. "de5093cd-page_0003.png"
LS_HASH_PREFIX_RE = re.compile(r"^[a-f0-9]{8}-(.+)$")
DEFAULT_MARGIN_PX = 15


# ---------- 1. Indexation des images sources ----------

def index_source_images(images_dataset: Path) -> Dict[str, List[Dict]]:
    """Construit un index {nom_fichier: [{'path', 'w', 'h', 'revue'}, ...]}.

    Permet de désambigüiser quand plusieurs revues ont le même nom de page.
    """
    index: Dict[str, List[Dict]] = defaultdict(list)
    if not images_dataset.is_dir():
        raise FileNotFoundError(f"images_dataset introuvable : {images_dataset}")

    for revue_dir in sorted(images_dataset.iterdir()):
        if not revue_dir.is_dir():
            continue
        for page_file in sorted(revue_dir.glob("page_*.png")):
            with Image.open(page_file) as img:
                w, h = img.size
            index[page_file.name].append({
                "path": page_file,
                "w": w,
                "h": h,
                "revue": revue_dir.name,
            })
    return index


# ---------- 2. Lecture du JSON Label Studio ----------

def strip_ls_hash(file_upload: str) -> str:
    """Retire le préfixe hash 8 hex de Label Studio."""
    m = LS_HASH_PREFIX_RE.match(file_upload)
    return m.group(1) if m else file_upload


def aggregate_regions(result: List[Dict]) -> Dict[str, Dict]:
    """Regroupe par region_id les 3 entrées (bbox / labels / textarea) que
    Label Studio sort dans `result`. Retourne {id: {x,y,w,h,label,text,
    original_width,original_height}}.
    """
    regions: Dict[str, Dict] = {}
    for item in result:
        rid = item.get("id")
        if not rid:
            continue
        value = item.get("value", {}) or {}
        from_name = item.get("from_name")

        regions.setdefault(rid, {
            "id": rid,
            "x": None, "y": None, "w": None, "h": None,
            "label": None,
            "text": None,
            "original_width": item.get("original_width"),
            "original_height": item.get("original_height"),
        })
        r = regions[rid]

        if from_name == "bbox" and "x" in value:
            r["x"] = value["x"]
            r["y"] = value["y"]
            r["w"] = value["width"]
            r["h"] = value["height"]
        elif from_name == "label" and "labels" in value:
            labels = value.get("labels") or []
            r["label"] = labels[0] if labels else None
        elif from_name == "transcription" and "text" in value:
            texts = value.get("text") or []
            r["text"] = texts[0] if texts else ""

        # Au cas où les dims n'étaient pas sur l'item courant
        if r["original_width"] is None and item.get("original_width"):
            r["original_width"] = item.get("original_width")
        if r["original_height"] is None and item.get("original_height"):
            r["original_height"] = item.get("original_height")

    return regions


# ---------- 3. Découpage ----------

def percent_to_pixel_bbox(
    region: Dict, ow: int, oh: int, margin_px: int, img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """Convertit (x%, y%, w%, h%) → (left, top, right, bottom) en pixels,
    avec marge ajoutée et clipping sur la taille image."""
    x1 = (region["x"] / 100.0) * ow
    y1 = (region["y"] / 100.0) * oh
    x2 = x1 + (region["w"] / 100.0) * ow
    y2 = y1 + (region["h"] / 100.0) * oh

    left = max(0, int(round(x1 - margin_px)))
    top = max(0, int(round(y1 - margin_px)))
    right = min(img_w, int(round(x2 + margin_px)))
    bottom = min(img_h, int(round(y2 + margin_px)))
    return left, top, right, bottom


def find_source(
    file_upload: str,
    original_width: Optional[int],
    original_height: Optional[int],
    index: Dict[str, List[Dict]],
) -> Optional[Dict]:
    """Trouve l'image source correspondant au file_upload Label Studio."""
    name = strip_ls_hash(file_upload)
    candidates = index.get(name, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Plusieurs revues ont la même page_NNNN.png → désambigüiser via dims
    if original_width and original_height:
        for c in candidates:
            if c["w"] == original_width and c["h"] == original_height:
                return c
    return candidates[0]  # fallback : prendre le premier (en log d'avertissement)


# ---------- 4. Pipeline principal ----------

def process(
    ls_json: Path,
    images_dataset: Path,
    out_dir: Path,
    margin_px: int = DEFAULT_MARGIN_PX,
) -> int:
    print(f"Indexation de {images_dataset}...")
    index = index_source_images(images_dataset)
    n_indexed = sum(len(v) for v in index.values())
    n_revues = sum(1 for d in images_dataset.iterdir() if d.is_dir())
    print(f"  {n_indexed} images indexées dans {n_revues} revues")

    print(f"Lecture de l'export Label Studio : {ls_json}")
    tasks = json.loads(ls_json.read_text(encoding="utf-8"))
    print(f"  {len(tasks)} tâches dans l'export")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_crops = 0
    n_skipped_empty = 0
    n_skipped_no_source = 0
    n_skipped_no_bbox = 0

    for task in tasks:
        annotations = task.get("annotations") or []
        if not annotations:
            continue
        # On prend la dernière annotation non-cancelled
        ann = None
        for a in reversed(annotations):
            if not a.get("was_cancelled"):
                ann = a
                break
        if ann is None:
            continue

        result = ann.get("result") or []
        if not result:
            continue

        regions = aggregate_regions(result)

        # Identifier la source
        file_upload = task.get("file_upload", "")
        # original_width/height sont posés par item dans `result` ; on prend
        # le premier qui en a.
        ow = oh = None
        for r in regions.values():
            if r.get("original_width") and r.get("original_height"):
                ow = r["original_width"]
                oh = r["original_height"]
                break

        source = find_source(file_upload, ow, oh, index)
        if source is None:
            print(f"  [skip] source introuvable pour {file_upload}", file=sys.stderr)
            n_skipped_no_source += 1
            continue

        # Charger l'image native une fois par page
        with Image.open(source["path"]) as img:
            img.load()
            img_w, img_h = img.size
            page_stem = source["path"].stem  # ex: page_0003
            revue = source["revue"]

            crops_revue_dir = out_dir / revue
            crops_revue_dir.mkdir(parents=True, exist_ok=True)

            for region in regions.values():
                # Skip si bbox incomplète
                if any(region.get(k) is None for k in ("x", "y", "w", "h")):
                    n_skipped_no_bbox += 1
                    continue
                # Skip si pas de transcription
                if not region.get("text"):
                    n_skipped_empty += 1
                    continue

                left, top, right, bottom = percent_to_pixel_bbox(
                    region, ow=img_w, oh=img_h,
                    margin_px=margin_px, img_w=img_w, img_h=img_h,
                )
                if right <= left or bottom <= top:
                    n_skipped_no_bbox += 1
                    continue

                crop = img.crop((left, top, right, bottom))
                stem = f"{page_stem}__{region['id']}"

                crop_png = crops_revue_dir / f"{stem}.png"
                crop_txt = crops_revue_dir / f"{stem}.txt"
                crop_meta = crops_revue_dir / f"{stem}.json"

                crop.save(crop_png)
                crop_txt.write_text(region["text"], encoding="utf-8")
                crop_meta.write_text(
                    json.dumps({
                        "region_id": region["id"],
                        "label": region["label"],
                        "source_image": str(source["path"]),
                        "revue": revue,
                        "page_stem": page_stem,
                        "bbox_pct": {
                            "x": region["x"], "y": region["y"],
                            "w": region["w"], "h": region["h"],
                        },
                        "bbox_px": {
                            "left": left, "top": top,
                            "right": right, "bottom": bottom,
                        },
                        "margin_px": margin_px,
                        "n_chars_gt": len(region["text"]),
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                n_crops += 1

    print()
    print(f"OK. {n_crops} crops écrits dans {out_dir}")
    if n_skipped_no_source:
        print(f"  {n_skipped_no_source} tâches ignorées (source introuvable)")
    if n_skipped_no_bbox:
        print(f"  {n_skipped_no_bbox} régions ignorées (bbox manquante/invalide)")
    if n_skipped_empty:
        print(f"  {n_skipped_empty} régions ignorées (transcription vide)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extrait crops + GT depuis un export Label Studio."
    )
    p.add_argument("--label-studio-json", required=True, type=Path)
    p.add_argument("--images-dataset", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--margin-px", type=int, default=DEFAULT_MARGIN_PX,
                   help=f"Marge ajoutée autour de la bbox (défaut : {DEFAULT_MARGIN_PX})")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return process(
        ls_json=args.label_studio_json.expanduser().resolve(),
        images_dataset=args.images_dataset.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        margin_px=args.margin_px,
    )


if __name__ == "__main__":
    sys.exit(main())
