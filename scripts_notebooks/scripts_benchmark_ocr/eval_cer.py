#!/usr/bin/env python3
"""Évaluation CER/WER par crop puis agrégations pondérées.

Lit :
  - data/benchmark_ocr/crops/<revue>/<page>__<zone>.txt   (référence GT)
  - data/benchmark_ocr/crops/<revue>/<page>__<zone>.json  (label, métadonnées)
  - data/benchmark_ocr/ocr_outputs/<engine>/<revue>/<page>__<zone>.txt (sortie OCR)

Écrit :
  - results/results.csv      : 1 ligne par (crop × moteur × variante)
  - results/summary.txt      : agrégations globales / par moteur / par type / par revue

Métrique : CER et WER via jiwer (distance de Levenshtein normalisée).

Variantes :
  - "brut"  : on garde les césures `-\\n` telles quelles
  - "joint" : on recolle les mots coupés (`-\\n` → "")

Normalisations communes (réf et hyp identiques) :
  - Unicode NFC
  - Sauts de ligne `\\n` → espace
  - Espaces multiples → simple
  - Casse conservée

Agrégation : pondérée par longueur de référence
    CER_global = Σ(S+D+I) / Σ N_ref     (PAS la moyenne des CER par crop)

Cohérent avec la méthodologie ICDAR / REID2017 documentée dans
data/benchmark_ocr/benchmark_review.md.

Usage :
    python eval_cer.py \\
        --crops-dir       data/benchmark_ocr/crops \\
        --ocr-outputs-dir data/benchmark_ocr/ocr_outputs \\
        --out-dir         data/benchmark_ocr/results \\
        [--engines tesseract,paddleocr,pero]
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer


WHITESPACE_RE = re.compile(r"\s+")
HYPHEN_LINEBREAK_RE = re.compile(r"-\n")


# ---------- Normalisation ----------

def normalize(text: str, *, variant: str = "brut") -> str:
    """Normalisation appliquée identiquement à ref et hyp.

    variant ∈ {"brut", "joint"} :
      - brut  : conserve les césures dans la GT (le `-` reste collé au mot
                tronqué). Les `\n` sont ensuite remplacés par un espace.
      - joint : recolle les mots coupés (`-\n` → ""), puis `\n` → espace.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", text)
    if variant == "joint":
        t = HYPHEN_LINEBREAK_RE.sub("", t)
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


# ---------- Métriques ----------

def edits_chars(ref: str, hyp: str) -> Tuple[int, int]:
    """Renvoie (nb_edits_chars, longueur_ref).

    nb_edits = substitutions + insertions + deletions sur les caractères.
    """
    out = jiwer.process_characters([ref], [hyp])
    n = out.substitutions + out.insertions + out.deletions
    return n, len(ref)


def edits_words(ref: str, hyp: str) -> Tuple[int, int]:
    """Renvoie (nb_edits_mots, longueur_ref_en_mots)."""
    out = jiwer.process_words([ref], [hyp])
    n = out.substitutions + out.insertions + out.deletions
    n_ref_words = len(ref.split()) if ref else 0
    return n, n_ref_words


# ---------- Découverte des crops ----------

def list_crops(crops_dir: Path) -> List[Path]:
    """Tous les .txt sous crops/<revue>/, hors ocr_outputs."""
    return sorted(p for p in crops_dir.rglob("*.txt") if p.is_file())


def load_meta(crop_txt: Path) -> Dict:
    """Charge le sidecar .json à côté du .txt (label, page_stem, revue, ...)."""
    meta_path = crop_txt.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def hyp_path(ocr_outputs_dir: Path, engine: str, crop_txt: Path, crops_dir: Path) -> Path:
    rel = crop_txt.relative_to(crops_dir)
    return ocr_outputs_dir / engine / rel


# ---------- Pipeline ----------

def evaluate(
    crops_dir: Path,
    ocr_outputs_dir: Path,
    out_dir: Path,
    engines: List[str],
) -> int:
    crops = list_crops(crops_dir)
    if not crops:
        print(f"Aucun crop dans {crops_dir}", file=sys.stderr)
        return 1
    print(f"{len(crops)} crops trouvés.")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    rows: List[Dict] = []

    # Agrégations: dict[(engine, variant)][bucket_key] -> (n_edits_c, n_chars,
    #                                                      n_edits_w, n_words)
    # bucket_key types: ("global",), ("type", label), ("revue", revue)
    agg: Dict[Tuple[str, str], Dict[Tuple, List[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0, 0])
    )

    n_missing_hyp = defaultdict(int)

    for crop_txt in crops:
        meta = load_meta(crop_txt)
        revue = meta.get("revue", crop_txt.parent.name)
        page_stem = meta.get("page_stem", "")
        zone_id = meta.get("region_id", crop_txt.stem.split("__", 1)[-1])
        type_zone = meta.get("label", "?")

        ref_raw = crop_txt.read_text(encoding="utf-8")

        for engine in engines:
            hp = hyp_path(ocr_outputs_dir, engine, crop_txt, crops_dir)
            if not hp.exists():
                n_missing_hyp[engine] += 1
                continue
            hyp_raw = hp.read_text(encoding="utf-8")

            for variant in ("brut", "joint"):
                ref = normalize(ref_raw, variant=variant)
                hyp = normalize(hyp_raw, variant=variant)
                if not ref:
                    continue  # GT vide après normalisation, on saute

                n_edits_c, n_chars = edits_chars(ref, hyp)
                n_edits_w, n_words = edits_words(ref, hyp)
                cer = n_edits_c / n_chars if n_chars > 0 else 0.0
                wer = n_edits_w / n_words if n_words > 0 else 0.0

                rows.append({
                    "revue": revue,
                    "page": page_stem,
                    "zone_id": zone_id,
                    "type_zone": type_zone,
                    "engine": engine,
                    "variant": variant,
                    "n_chars_ref": n_chars,
                    "n_chars_hyp": len(hyp),
                    "n_edits_chars": n_edits_c,
                    "cer": round(cer, 4),
                    "n_words_ref": n_words,
                    "n_edits_words": n_edits_w,
                    "wer": round(wer, 4),
                })

                # Agrégations
                for bucket in (("global",), ("type", type_zone), ("revue", revue)):
                    a = agg[(engine, variant)][bucket]
                    a[0] += n_edits_c
                    a[1] += n_chars
                    a[2] += n_edits_w
                    a[3] += n_words

    if not rows:
        print("Aucune ligne de résultat — vérifier que les sorties OCR existent.",
              file=sys.stderr)
        return 2

    # Écriture CSV
    fieldnames = [
        "revue", "page", "zone_id", "type_zone", "engine", "variant",
        "n_chars_ref", "n_chars_hyp", "n_edits_chars", "cer",
        "n_words_ref", "n_edits_words", "wer",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Écrit : {csv_path}  ({len(rows)} lignes)")

    # Écriture résumé
    summary_path = out_dir / "summary.txt"
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("RÉSUMÉ BENCHMARK OCR — CER/WER agrégés (pondérés par longueur)")
    lines.append("=" * 70)
    lines.append("")
    if any(n_missing_hyp.values()):
        lines.append("Sorties OCR manquantes (crop sans .txt côté moteur) :")
        for e, n in n_missing_hyp.items():
            lines.append(f"  - {e}: {n}")
        lines.append("")

    for variant in ("brut", "joint"):
        lines.append(f"### Variante : {variant}")
        lines.append("")

        # Global par moteur
        lines.append("Global (toutes zones, toutes revues) :")
        lines.append(f"  {'moteur':<14} {'CER':>8} {'WER':>8}  {'n_chars':>10}  {'n_words':>10}")
        for engine in engines:
            a = agg[(engine, variant)].get(("global",))
            if not a:
                continue
            cer = a[0] / a[1] if a[1] else 0.0
            wer = a[2] / a[3] if a[3] else 0.0
            lines.append(f"  {engine:<14} {cer:>8.4f} {wer:>8.4f}  {a[1]:>10}  {a[3]:>10}")
        lines.append("")

        # Par type_zone
        all_types = sorted({k[1] for (e, v), buckets in agg.items()
                            for k in buckets if k[0] == "type" and v == variant})
        if all_types:
            lines.append("Par type_zone :")
            header = "  " + f"{'moteur':<14}" + "".join(f" {t:>10}" for t in all_types)
            lines.append(header)
            for engine in engines:
                cells = [f"{engine:<14}"]
                for t in all_types:
                    a = agg[(engine, variant)].get(("type", t))
                    if a and a[1]:
                        cells.append(f"{a[0]/a[1]:>10.4f}")
                    else:
                        cells.append(f"{'—':>10}")
                lines.append("  " + " ".join(cells))
            lines.append("  (CER pondéré)")
            lines.append("")

        # Par revue
        all_revues = sorted({k[1] for (e, v), buckets in agg.items()
                             for k in buckets if k[0] == "revue" and v == variant})
        if all_revues:
            lines.append("Par revue :")
            for engine in engines:
                lines.append(f"  {engine}:")
                for r in all_revues:
                    a = agg[(engine, variant)].get(("revue", r))
                    if a and a[1]:
                        cer = a[0] / a[1]
                        wer = a[2] / a[3] if a[3] else 0.0
                        lines.append(f"    {r:<60} CER={cer:.4f}  WER={wer:.4f}  N={a[1]}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("")

    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Écrit : {summary_path}")
    print()
    print(summary_text)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Évalue CER/WER de chaque moteur OCR contre la GT par crop."
    )
    p.add_argument("--crops-dir", required=True, type=Path)
    p.add_argument("--ocr-outputs-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--engines",
        default="tesseract,paddleocr,pero",
        help="Liste séparée par virgule (défaut : tous)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    return evaluate(
        crops_dir=args.crops_dir.expanduser().resolve(),
        ocr_outputs_dir=args.ocr_outputs_dir.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        engines=engines,
    )


if __name__ == "__main__":
    sys.exit(main())
