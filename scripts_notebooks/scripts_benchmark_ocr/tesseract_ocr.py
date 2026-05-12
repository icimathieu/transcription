#!/usr/bin/env python3
"""Tesseract OCR pour un crop mono-bloc (benchmark OCR par région).

Adaptation de scripts_notebooks/tesseract_boxes.py : on suppose que l'image
en entrée est un crop d'UNE seule zone GT (1 colonne / 1 titre / 1 bloc),
donc pas de reconstruction de colonnes, pas de réordonnancement complexe.
On lit Tesseract en TSV (mots + boîtes), on regroupe par ligne, on trie
top-to-bottom puis left-to-right dans chaque ligne, on concatène.

Usage CLI :
    python tesseract_ocr.py --image crop.png [--out result.txt]

Usage module :
    from tesseract_ocr import transcribe
    text = transcribe(Path("crop.png"))
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_TESSERACT_BIN = "/opt/homebrew/bin/tesseract"
DEFAULT_LANG = "fra"
DEFAULT_PSM = 6  # uniform block of text (mono-bloc)


def _run_tesseract_tsv(image_path: Path, lang: str, psm: int, tesseract_bin: str) -> str:
    cmd = [
        tesseract_bin,
        str(image_path),
        "stdout",
        "-l", lang,
        "--psm", str(psm),
        "tsv",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Tesseract failed")
    return proc.stdout


def _parse_tsv(tsv_text: str) -> List[Dict]:
    """Regroupe les mots par ligne (block_num, par_num, line_num)."""
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    words_by_line: Dict[Tuple[int, int, int], List[Dict]] = {}

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

        try:
            conf = float(row.get("conf", "-1"))
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
        )
        words_by_line.setdefault(key, []).append({
            "text": text,
            "x1": left,
            "y1": top,
            "x2": left + width,
            "y2": top + height,
        })

    lines = []
    for key in sorted(words_by_line.keys()):
        words = words_by_line[key]
        words.sort(key=lambda w: w["x1"])
        text = " ".join(w["text"] for w in words).strip()
        if not text:
            continue
        y1 = min(w["y1"] for w in words)
        y2 = max(w["y2"] for w in words)
        lines.append({"text": text, "cy": (y1 + y2) / 2.0})
    return lines


def transcribe(
    image_path: Path,
    *,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
    tesseract_bin: str = DEFAULT_TESSERACT_BIN,
) -> str:
    """Renvoie la transcription concaténée (lignes séparées par \\n)."""
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    tsv = _run_tesseract_tsv(image_path, lang=lang, psm=psm, tesseract_bin=tesseract_bin)
    lines = _parse_tsv(tsv)
    lines.sort(key=lambda l: l["cy"])
    return "\n".join(l["text"] for l in lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tesseract OCR sur un crop mono-bloc.")
    p.add_argument("--image", required=True, help="Chemin vers le crop.")
    p.add_argument("--out", default=None, help="Fichier texte de sortie (sinon stdout).")
    p.add_argument("--lang", default=DEFAULT_LANG)
    p.add_argument("--psm", type=int, default=DEFAULT_PSM)
    p.add_argument("--tesseract-bin", default=DEFAULT_TESSERACT_BIN)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if not Path(args.tesseract_bin).exists():
        print(f"Binaire tesseract introuvable: {args.tesseract_bin}", file=sys.stderr)
        return 2

    text = transcribe(
        Path(args.image),
        lang=args.lang,
        psm=args.psm,
        tesseract_bin=args.tesseract_bin,
    )

    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
