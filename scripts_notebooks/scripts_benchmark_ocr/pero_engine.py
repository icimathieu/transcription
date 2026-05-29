#!/usr/bin/env python3
"""pero-ocr pour un crop mono-bloc (benchmark OCR par région).

pero-ocr (https://github.com/DCGM/pero-ocr) est un pipeline OCR maintenu
par DCGM/VUT Brno, avec layout analysis + reconnaissance + LM. Il existe
des modèles tiers pour le français historique (cf. HAL hal-03682991).

Pré-requis (à faire UNE FOIS avant d'utiliser ce script) :

  # 1. Installer pero-ocr et ses deps
  pip install pero-ocr

  # 2. Télécharger un modèle. Le repo officiel donne accès à des modèles
  #    via leur Nextcloud :
  #      https://pero-ocr.fit.vutbr.cz/
  #    Pour le français imprimé XIXe, le pack « european printed » fonctionne
  #    raisonnablement zero-shot, mais des modèles dédiés FR (cf. HAL ci-dessus)
  #    existent — placer le dossier modèle quelque part puis :
  export PERO_OCR_CONFIG=/chemin/vers/modele/config.ini

Le script lit la variable d'env PERO_OCR_CONFIG (ou l'arg --config) pour
trouver le config.ini du modèle. Le fichier config.ini référence en chemins
relatifs les sous-modèles (PARSE_FOLDER, OCR, LANGUAGE_MODEL...), il doit
donc rester dans le dossier du modèle téléchargé.

Usage CLI :
    export PERO_OCR_CONFIG=~/pero_models/european_printed/config.ini
    python pero_ocr.py --image crop.png [--out result.txt]

Usage module :
    from pero_ocr import transcribe
    text = transcribe(Path("crop.png"), config_path=Path("..."))
"""

import argparse
import configparser
import os
import sys
from pathlib import Path
from typing import List, Optional


_PARSER_INSTANCE = None  # cache d'instance PageParser


def _get_parser(config_path: Path):
    """Lazy-init + cache d'un PageParser pero-ocr.

    Le config.ini référence les sous-modèles en chemins relatifs au dossier
    du config — d'où le passage de `config_path` en plus de la config parsée.
    """
    global _PARSER_INSTANCE
    if _PARSER_INSTANCE is not None:
        return _PARSER_INSTANCE

    from pero_ocr.document_ocr.page_parser import PageParser

    config = configparser.ConfigParser()
    config.read(str(config_path))
    _PARSER_INSTANCE = PageParser(config, config_path=str(config_path.parent))
    return _PARSER_INSTANCE


def transcribe(
    image_path: Path,
    *,
    config_path: Optional[Path] = None,
) -> str:
    """Renvoie la transcription concaténée (lignes triées top-to-bottom).

    `config_path` : chemin vers le config.ini d'un modèle pero-ocr.
    Si None, on lit la variable d'environnement PERO_OCR_CONFIG.
    """
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if config_path is None:
        env = os.environ.get("PERO_OCR_CONFIG")
        if not env:
            raise RuntimeError(
                "PERO_OCR_CONFIG non défini. Donner --config ou exporter "
                "PERO_OCR_CONFIG=/chemin/vers/config.ini d'un modèle pero-ocr."
            )
        config_path = Path(env)
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config.ini pero-ocr introuvable : {config_path}")

    import cv2
    from pero_ocr.core.layout import PageLayout

    parser = _get_parser(config_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Image illisible : {image_path}")

    h, w = image.shape[:2]
    page_layout = PageLayout(id=image_path.stem, page_size=(h, w))
    page_layout = parser.process_page(image, page_layout)

    # Respecter l'ordre de lecture fixé par pero (SmartRegionSorter) :
    # on itère région par région, et dans chaque région ligne par ligne via
    # l'ordre fourni par l'API (déjà top-to-bottom au sein d'une colonne).
    # Trier globalement par cy mélange les colonnes ligne-à-ligne — bug
    # documenté dans benchmark_review.md.
    region_texts: List[str] = []
    for region in page_layout.regions:
        # Au sein d'une région : tri top-to-bottom par y de baseline.
        lines_in_region = []
        for line in region.lines:
            text = (line.transcription or "").strip()
            if not text:
                continue
            try:
                ys = [pt[1] for pt in line.baseline]
                cy = float(sum(ys) / len(ys)) if ys else 0.0
            except Exception:
                cy = 0.0
            lines_in_region.append((cy, text))
        lines_in_region.sort(key=lambda t: t[0])
        if lines_in_region:
            region_texts.append("\n".join(t for _, t in lines_in_region))
    return "\n".join(region_texts)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pero-ocr sur un crop mono-bloc.")
    p.add_argument("--image", required=True, help="Chemin vers le crop.")
    p.add_argument("--out", default=None, help="Fichier texte de sortie (sinon stdout).")
    p.add_argument(
        "--config",
        default=None,
        help="Chemin vers config.ini d'un modèle pero-ocr (sinon $PERO_OCR_CONFIG).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        text = transcribe(
            Path(args.image),
            config_path=Path(args.config) if args.config else None,
        )
    except ImportError as exc:
        print("Import impossible. Installer dans le venv :", file=sys.stderr)
        print("  pip install pero-ocr", file=sys.stderr)
        print(f"Détail : {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
