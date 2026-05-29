#!/usr/bin/env python3
"""PaddleOCR server SANS downscale (full-res benchmark).

Variante de paddleocr_ocr.py qui désactive le downscale (max_side=99999).
Conçue pour être lancée en subprocess-par-crop via run_benchmark.py
--subprocess-per-crop, sinon Paddle se fait OOM-kill sur les gros crops
(jusqu'à 15 Mpx dans le dataset benchmark).

Sortie OCR : ocr_outputs/paddleocr_fullres/<revue>/...
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import paddleocr_ocr  # noqa: E402


DEFAULT_LANG = "fr"
NO_DOWNSCALE = 99999  # désactive effectivement le downscale


def transcribe(
    image_path: Path,
    *,
    lang: str = DEFAULT_LANG,
    keep_model_source_check: bool = False,
) -> str:
    """Délègue à paddleocr_ocr.transcribe mais avec max_side énorme."""
    return paddleocr_ocr.transcribe(
        image_path,
        lang=lang,
        keep_model_source_check=keep_model_source_check,
        max_side=NO_DOWNSCALE,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="PaddleOCR server full-res.")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--lang", default=DEFAULT_LANG)
    args = p.parse_args()
    text = transcribe(Path(args.image), lang=args.lang)
    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
