#!/usr/bin/env python3
"""Statistiques de temps d'inférence par moteur (lit ocr_outputs/timings.csv).

Produit :
  - moyenne / médiane / p95 / total / débit
  - projections pour 1k / 10k / 100k / 1M crops
  - décomposition par taille de zone (n_chars_out comme proxy)

Note sur le « cold start » : le tout premier crop traité par chaque moteur
inclut le chargement du modèle (lourd pour Paddle/pero, négligeable pour
Tesseract). On reporte donc les stats avec et **sans** ce cold start.

Usage :
    python eval_timing.py \\
        --timings-csv data/benchmark_ocr/ocr_outputs/timings.csv \\
        [--out data/benchmark_ocr/results/timing_summary.txt]
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s/60:.1f} min"
    if s < 86400:
        return f"{s/3600:.1f} h"
    return f"{s/86400:.1f} j"


def load_timings(path: Path) -> Dict[str, List[Dict]]:
    """Renvoie {engine: [{crop, elapsed_s, n_chars_out}, ...]} dans l'ordre du fichier."""
    by_engine: Dict[str, List[Dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_engine[row["engine"]].append({
                "crop": row["crop"],
                "elapsed_s": float(row["elapsed_s"]),
                "n_chars_out": int(row["n_chars_out"]),
            })
    return by_engine


def stats_for(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "total": 0, "mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    return {
        "n": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Stats de temps d'inférence OCR.")
    p.add_argument("--timings-csv", required=True, type=Path)
    p.add_argument("--out", default=None, type=Path,
                   help="Fichier de sortie (sinon stdout uniquement).")
    args = p.parse_args()

    csv_path = args.timings_csv.expanduser().resolve()
    if not csv_path.exists():
        print(f"Introuvable : {csv_path}", file=sys.stderr)
        return 1

    by_engine = load_timings(csv_path)
    if not by_engine:
        print("CSV vide.", file=sys.stderr)
        return 2

    try:
        display_path = csv_path.relative_to(Path.cwd())
    except ValueError:
        display_path = csv_path

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("TEMPS D'INFÉRENCE PAR MOTEUR")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Source : {display_path}")
    lines.append("")

    # Stats principales (avec et sans cold start)
    header = (f"  {'moteur':<11} {'n':>3}  {'total':>9}  {'mean':>8}  "
              f"{'median':>8}  {'p95':>8}  {'min':>7}  {'max':>7}")
    lines.append("Avec cold start (1ère exécution = chargement du modèle inclus) :")
    lines.append(header)
    for engine, rows in by_engine.items():
        elapsed = [r["elapsed_s"] for r in rows]
        s = stats_for(elapsed)
        lines.append(
            f"  {engine:<11} {s['n']:>3}  {s['total']:>8.1f}s  "
            f"{s['mean']:>7.2f}s  {s['median']:>7.2f}s  "
            f"{s['p95']:>7.2f}s  {s['min']:>6.2f}s  {s['max']:>6.2f}s"
        )
    lines.append("")

    lines.append("Sans cold start (on retire le 1er crop de chaque moteur) :")
    lines.append(header)
    hot_stats = {}
    for engine, rows in by_engine.items():
        if len(rows) < 2:
            continue
        hot = [r["elapsed_s"] for r in rows[1:]]
        s = stats_for(hot)
        hot_stats[engine] = s
        lines.append(
            f"  {engine:<11} {s['n']:>3}  {s['total']:>8.1f}s  "
            f"{s['mean']:>7.2f}s  {s['median']:>7.2f}s  "
            f"{s['p95']:>7.2f}s  {s['min']:>6.2f}s  {s['max']:>6.2f}s"
        )
    lines.append("")

    # Cold start observé
    lines.append("Cold start (durée du 1er crop) :")
    for engine, rows in by_engine.items():
        if rows:
            lines.append(f"  {engine:<11} {rows[0]['elapsed_s']:>7.2f}s "
                         f"(crop {rows[0]['crop']})")
    lines.append("")

    # Projections (basées sur la médiane "hot")
    lines.append("Projections (extrapolées sur médiane hot, sans cold start) :")
    lines.append("  Le total inclut 1 cold start unique amorti.")
    lines.append("")
    lines.append(f"  {'moteur':<11}  {'1 box':>10}  {'10 box':>10}  "
                 f"{'1k box':>10}  {'100k box':>12}  {'1M box':>14}")
    for engine, s in hot_stats.items():
        med = s["median"]
        cold = by_engine[engine][0]["elapsed_s"]
        for n in [1, 10, 1000, 100_000, 1_000_000]:
            pass  # juste pour itérer
        proj = lambda n: cold + med * n  # noqa: E731
        lines.append(
            f"  {engine:<11}  {fmt_seconds(proj(1)):>10}  "
            f"{fmt_seconds(proj(10)):>10}  {fmt_seconds(proj(1000)):>10}  "
            f"{fmt_seconds(proj(100_000)):>12}  {fmt_seconds(proj(1_000_000)):>14}"
        )
    lines.append("")

    # Corrélation taille de zone ↔ temps (médiane par bucket de n_chars)
    lines.append("Temps médian par taille de zone (n_chars_out) :")
    lines.append(f"  {'moteur':<11}  {'<100 chars':>13}  {'100-500':>10}  "
                 f"{'500-2000':>10}  {'≥2000':>10}")
    for engine, rows in by_engine.items():
        buckets = {"<100": [], "100-500": [], "500-2000": [], "≥2000": []}
        for r in rows[1:]:  # hot only
            n = r["n_chars_out"]
            if n < 100:
                buckets["<100"].append(r["elapsed_s"])
            elif n < 500:
                buckets["100-500"].append(r["elapsed_s"])
            elif n < 2000:
                buckets["500-2000"].append(r["elapsed_s"])
            else:
                buckets["≥2000"].append(r["elapsed_s"])

        def cell(vs):
            if not vs:
                return "—"
            return f"{statistics.median(vs):.2f}s"
        lines.append(
            f"  {engine:<11}  "
            f"{cell(buckets['<100']):>13}  "
            f"{cell(buckets['100-500']):>10}  "
            f"{cell(buckets['500-2000']):>10}  "
            f"{cell(buckets['≥2000']):>10}"
        )
    lines.append("")

    text = "\n".join(lines)
    print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nÉcrit : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
