#!/usr/bin/env python3
"""PaddleOCR pour un crop mono-bloc (benchmark OCR par région).

Adaptation de scripts_notebooks/paddleocr_boxes.py : on suppose que l'image
en entrée est un crop d'UNE seule zone GT, donc pas de reconstruction de
colonnes — juste un tri top-to-bottom des lignes détectées.

Particularité : on cache l'instance PaddleOCR (lourde à charger) entre
appels successifs depuis le même processus pour ne pas payer le coût
d'initialisation à chaque crop.

Usage CLI :
    python paddleocr_ocr.py --image crop.png [--out result.txt]

Usage module :
    from paddleocr_ocr import transcribe
    text = transcribe(Path("crop.png"))
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_LANG = "fr"
DEFAULT_MAX_SIDE = 2000  # downscale crops dont le côté max dépasse cette valeur.
                         # nécessaire car Paddle (modèle server_det) consomme
                         # beaucoup de RAM sur les très grands crops. Tests :
                         #   - max_side=full   → OOM/timeout sur la plupart des
                         #     crops > 5 Mpx (cf. benchmark_review.md).
                         #   - max_side=1500   → marche partout, ~37 min/160 crops.
                         #   - max_side=2000   → compromis qualité/RAM retenu.
                         # 2000 px ≈ 270 dpi sur les pages Gallica scannées en
                         # full res (typiquement 4000-5000 px de côté max).
_OCR_INSTANCE = None  # cache d'instance PaddleOCR


def _bbox_to_xyxy(bbox: Any) -> Tuple[float, float, float, float]:
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
    """Extraction tolérante (selon version Paddle) des lignes texte + bbox."""
    texts = getattr(result_obj, "rec_texts", None)
    boxes = getattr(result_obj, "rec_boxes", None)
    scores = getattr(result_obj, "rec_scores", None)

    if (texts is None or boxes is None) and hasattr(result_obj, "get"):
        try:
            texts = result_obj.get("rec_texts", texts)
            boxes = result_obj.get("rec_boxes", boxes)
            scores = result_obj.get("rec_scores", scores)
        except Exception:
            pass

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
        x1, y1, x2, y2 = _bbox_to_xyxy(boxes[i])
        lines.append({
            "text": t,
            "score": float(scores[i]) if scores is not None else None,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
        })
    return lines


def _get_ocr(lang: str = DEFAULT_LANG, keep_model_source_check: bool = False):
    """Lazy-init + cache d'une instance PaddleOCR."""
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
        lang=lang,
    )
    return _OCR_INSTANCE


def _maybe_downscale(image_path: Path, max_side: int) -> Path:
    """Si le crop dépasse max_side sur son plus grand côté, on écrit une
    version réduite dans un tmp et on renvoie ce chemin. Sinon on renvoie
    le chemin original. Évite l'OOM sur les très grands crops (>5 Mpx).
    """
    from PIL import Image
    import tempfile
    with Image.open(image_path) as img:
        w, h = img.size
        m = max(w, h)
        if m <= max_side:
            return image_path
        scale = max_side / m
        new_size = (int(w * scale), int(h * scale))
        # tmp persistant le temps de l'inférence ; on garde le suffix .png
        tmp = Path(tempfile.gettempdir()) / f"paddleocr_ds_{image_path.stem}.png"
        img.resize(new_size, Image.LANCZOS).save(tmp, "PNG")
        return tmp


def transcribe(
    image_path: Path,
    *,
    lang: str = DEFAULT_LANG,
    keep_model_source_check: bool = False,
    max_side: int = DEFAULT_MAX_SIDE,
) -> str:
    """Renvoie la transcription concaténée (lignes triées top-to-bottom).

    Si le crop dépasse `max_side` (px) sur son plus grand côté, il est
    automatiquement redimensionné en LANCZOS avant inférence — sinon Paddle
    consomme >5 GB RAM et se fait OOM-kill sur macOS.
    """
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    img_for_ocr = _maybe_downscale(image_path, max_side)

    ocr = _get_ocr(lang=lang, keep_model_source_check=keep_model_source_check)
    results = list(ocr.predict(str(img_for_ocr)))
    if not results:
        return ""

    lines = _extract_lines(results[0])
    # Tri top-to-bottom (cy) puis left-to-right (cx) en cas d'égalité.
    lines.sort(key=lambda l: (l["cy"], l["cx"]))
    return "\n".join(l["text"] for l in lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PaddleOCR sur un crop mono-bloc.")
    p.add_argument("--image", required=True, help="Chemin vers le crop.")
    p.add_argument("--out", default=None, help="Fichier texte de sortie (sinon stdout).")
    p.add_argument("--lang", default=DEFAULT_LANG)
    p.add_argument("--keep-model-source-check", action="store_true")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        text = transcribe(
            Path(args.image),
            lang=args.lang,
            keep_model_source_check=args.keep_model_source_check,
        )
    except ImportError as exc:
        print("Import impossible. Installer dans le venv :", file=sys.stderr)
        print("  python -m pip install -U paddlepaddle paddleocr pillow", file=sys.stderr)
        print(f"Détail : {exc}", file=sys.stderr)
        return 2

    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
