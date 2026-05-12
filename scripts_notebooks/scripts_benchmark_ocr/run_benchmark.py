#!/usr/bin/env python3
"""Lance les 3 moteurs OCR (Tesseract, PaddleOCR, pero-ocr) sur tous les
crops produits par extract_crops.py.

Pour chaque crop `crops/<revue>/<page>__<region>.png`, écrit la sortie de
chaque moteur dans `ocr_outputs/<engine>/<revue>/<page>__<region>.txt` et
mesure le temps d'inférence.

Skip-if-exists : si le fichier de sortie existe déjà et n'est pas vide, on
saute (permet de relancer après interruption).

Usage :
    python run_benchmark.py \\
        --crops-dir   data/benchmark_ocr/crops \\
        --out-dir     data/benchmark_ocr/ocr_outputs \\
        [--engines tesseract,paddleocr,pero] \\
        [--pero-config /chemin/vers/config.ini]
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List


SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


ENGINE_MODULE = {
    "tesseract": "tesseract_ocr",
    "paddleocr": "paddleocr_ocr",
    "pero": "pero_engine",  # renommé pour éviter le conflit avec le package pero_ocr
}


def load_engine(name: str, **kwargs) -> Callable[[Path], str]:
    """Importe le module du moteur et renvoie un callable transcribe(path)."""
    if name not in ENGINE_MODULE:
        raise ValueError(f"Moteur inconnu : {name}")
    mod = importlib.import_module(ENGINE_MODULE[name])
    transcribe = getattr(mod, "transcribe")

    if name == "pero":
        pero_config = kwargs.get("pero_config")
        return lambda p: transcribe(p, config_path=pero_config)
    return lambda p: transcribe(p)


def list_crops(crops_dir: Path) -> List[Path]:
    """Liste tous les .png sous crops_dir/<revue>/, triés."""
    return sorted(crops_dir.rglob("*.png"))


def out_path(out_dir: Path, engine: str, crop_path: Path, crops_dir: Path) -> Path:
    """ocr_outputs/<engine>/<revue>/<stem>.txt"""
    rel = crop_path.relative_to(crops_dir)
    return out_dir / engine / rel.with_suffix(".txt")


def run_engine(
    name: str,
    transcribe: Callable[[Path], str],
    crops: List[Path],
    crops_dir: Path,
    out_dir: Path,
    skip_existing: bool = True,
) -> List[Dict]:
    """Lance un moteur sur tous les crops. Renvoie liste de dicts (timings)."""
    print(f"\n=== Moteur : {name} ({len(crops)} crops) ===")
    timings: List[Dict] = []
    n_done = 0
    n_skipped = 0
    n_errors = 0

    for crop in crops:
        target = out_path(out_dir, name, crop, crops_dir)
        target.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and target.exists() and target.stat().st_size > 0:
            n_skipped += 1
            continue

        try:
            t0 = perf_counter()
            text = transcribe(crop)
            dt = perf_counter() - t0
        except Exception as exc:
            print(f"  [err] {crop.name} : {exc}", file=sys.stderr)
            n_errors += 1
            continue

        target.write_text(text or "", encoding="utf-8")
        timings.append({
            "engine": name,
            "crop": str(crop.relative_to(crops_dir)),
            "elapsed_s": round(dt, 3),
            "n_chars_out": len(text or ""),
        })
        n_done += 1
        if n_done % 10 == 0:
            print(f"  {n_done} crops traités...")

    print(f"  Fait : {n_done} | sautés : {n_skipped} | erreurs : {n_errors}")
    return timings


def write_timings_csv(out_dir: Path, all_timings: List[Dict]) -> None:
    """Append au timings.csv (pour que les runs successifs sur des moteurs
    différents accumulent leurs mesures). Le header n'est écrit que si le
    fichier n'existe pas encore."""
    if not all_timings:
        return
    csv_path = out_dir / "timings.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["engine", "crop", "elapsed_s", "n_chars_out"])
        if write_header:
            writer.writeheader()
        writer.writerows(all_timings)
    print(f"\nTimings ajoutés à {csv_path} ({len(all_timings)} lignes)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lance les 3 moteurs OCR sur les crops.")
    p.add_argument("--crops-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--engines",
        default="tesseract,paddleocr,pero",
        help="Liste séparée par virgule (défaut : tous)",
    )
    p.add_argument(
        "--pero-config",
        default=None,
        type=Path,
        help="Chemin config.ini pero-ocr (sinon $PERO_OCR_CONFIG)",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Forcer la régénération même si la sortie existe déjà",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    crops_dir = args.crops_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if not crops_dir.is_dir():
        print(f"Crops introuvables : {crops_dir}", file=sys.stderr)
        return 1

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    for e in engines:
        if e not in ENGINE_MODULE:
            print(f"Moteur inconnu : {e}", file=sys.stderr)
            return 2

    crops = list_crops(crops_dir)
    if not crops:
        print(f"Aucun crop dans {crops_dir}", file=sys.stderr)
        return 3
    print(f"{len(crops)} crops trouvés sous {crops_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_timings: List[Dict] = []

    for engine in engines:
        try:
            transcribe = load_engine(engine, pero_config=args.pero_config)
        except Exception as exc:
            print(f"[skip moteur '{engine}'] chargement impossible : {exc}", file=sys.stderr)
            continue

        timings = run_engine(
            name=engine,
            transcribe=transcribe,
            crops=crops,
            crops_dir=crops_dir,
            out_dir=out_dir,
            skip_existing=not args.no_skip_existing,
        )
        all_timings.extend(timings)

    write_timings_csv(out_dir, all_timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
