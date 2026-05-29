#!/usr/bin/env python3
"""PaddleOCR — variante mobile (PP-OCRv5_mobile_det au lieu de _server_det).

Beaucoup plus léger en RAM/CPU que la variante par défaut, au prix d'une
qualité de détection un peu moindre. Utile pour faire tourner sur Mac CPU
sans risque de OOM, et pour estimer la perte de qualité acceptable pour un
déploiement sur gros volumes.

Bench distinct de paddleocr_ocr.py : la sortie va dans
ocr_outputs/paddleocr_mobile/<revue>/...

Usage :
    python paddleocr_mobile.py --image crop.png --out result.txt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LANG = "fr"
DEFAULT_MAX_SIDE = 1500  # mêmes raisons que paddleocr_ocr.py
_OCR_INSTANCE = None


# On réutilise les utilitaires de paddleocr_ocr (extraction lignes, downscale).
# On garde le code DRY : on importe la version "server" comme base puis on
# remplace juste le constructeur PaddleOCR avec les modèles mobile.

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import paddleocr_ocr  # noqa: E402


def _get_ocr(lang: str = DEFAULT_LANG, keep_model_source_check: bool = False):
    """Lazy-init + cache d'une instance PaddleOCR avec modèles MOBILE."""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE

    if not keep_model_source_check:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

    from paddleocr import PaddleOCR
    _OCR_INSTANCE = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",   # <-- variante légère
        # text_recognition_model_name reste latin_PP-OCRv5_mobile_rec par défaut
        lang=lang,
    )
    return _OCR_INSTANCE


def transcribe(
    image_path: Path,
    *,
    lang: str = DEFAULT_LANG,
    keep_model_source_check: bool = False,
    max_side: int = DEFAULT_MAX_SIDE,
) -> str:
    """Renvoie la transcription concaténée (lignes triées top-to-bottom)."""
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    img_for_ocr = paddleocr_ocr._maybe_downscale(image_path, max_side)

    ocr = _get_ocr(lang=lang, keep_model_source_check=keep_model_source_check)
    results = list(ocr.predict(str(img_for_ocr)))
    if not results:
        return ""

    lines = paddleocr_ocr._extract_lines(results[0])
    lines.sort(key=lambda l: (l["cy"], l["cx"]))
    return "\n".join(l["text"] for l in lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PaddleOCR mobile sur un crop mono-bloc.")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--lang", default=DEFAULT_LANG)
    p.add_argument("--keep-model-source-check", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    text = transcribe(
        Path(args.image),
        lang=args.lang,
        keep_model_source_check=args.keep_model_source_check,
    )
    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
