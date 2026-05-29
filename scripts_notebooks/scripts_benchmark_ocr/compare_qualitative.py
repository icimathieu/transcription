#!/usr/bin/env python3
"""Rapport HTML de comparaison qualitative OCR.

Pour chaque crop, génère une "carte" qui affiche :
  - l'image du crop,
  - le label (corps / titre / auteur / autre),
  - la transcription GT,
  - la sortie de chaque moteur OCR, avec CER calculé en direct,
  - un diff caractère par caractère colorisé (vert = match, rouge = erreur).

Sortie : un fichier HTML par revue, dans <out-dir>/comparison_<revue>.html
+ un index.html qui pointe vers chacun.

Utile pour l'analyse qualitative : on voit *où* chaque moteur se trompe
(diacritiques, ligatures, début/fin de ligne, etc.), pas juste un CER global.

Usage :
    python compare_qualitative.py \\
        --crops-dir data/benchmark_ocr/crops \\
        --ocr-outputs-dir data/benchmark_ocr/ocr_outputs \\
        --out-dir data/benchmark_ocr/results \\
        [--engines tesseract,paddleocr,pero]
"""

import argparse
import difflib
import html
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer


# ---------- Normalisation (identique à eval_cer.py) ----------

WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_diff(text: str) -> str:
    """Normalise pour l'affichage côte-à-côte mais on garde les sauts de ligne."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_for_cer(text: str) -> str:
    """Normalisation pour calcul CER (NFC + espaces collapsés + \\n → espace)."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


# ---------- Diff caractère par caractère ----------

def char_diff_html(ref: str, hyp: str) -> Tuple[str, str]:
    """Renvoie (ref_html, hyp_html) avec spans colorisés selon le diff."""
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    ref_parts: List[str] = []
    hyp_parts: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_seg = html.escape(ref[i1:i2]).replace("\n", "↵\n")
        hyp_seg = html.escape(hyp[j1:j2]).replace("\n", "↵\n")
        if tag == "equal":
            ref_parts.append(f'<span class="ok">{ref_seg}</span>')
            hyp_parts.append(f'<span class="ok">{hyp_seg}</span>')
        elif tag == "replace":
            ref_parts.append(f'<span class="sub">{ref_seg}</span>')
            hyp_parts.append(f'<span class="sub">{hyp_seg}</span>')
        elif tag == "delete":
            ref_parts.append(f'<span class="del">{ref_seg}</span>')
        elif tag == "insert":
            hyp_parts.append(f'<span class="ins">{hyp_seg}</span>')
    return "".join(ref_parts), "".join(hyp_parts)


def cer_for(ref: str, hyp: str) -> float:
    """CER simple basé sur jiwer.process_characters."""
    if not ref:
        return 0.0
    out = jiwer.process_characters([ref], [hyp])
    n_edits = out.substitutions + out.insertions + out.deletions
    return n_edits / len(ref)


# ---------- Découverte des crops ----------

def list_crops(crops_dir: Path) -> List[Path]:
    return sorted(p for p in crops_dir.rglob("*.txt") if p.is_file())


def load_meta(crop_txt: Path) -> Dict:
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


# ---------- HTML rendering ----------

LABEL_COLORS = {
    "corps":  "#3338fb",
    "titre":  "#d50008",
    "auteur": "#FFC069",
    "autre":  "#00aa2a",
}

CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  margin: 0; padding: 20px; background: #f6f6f7; color: #222;
}
h1 { font-size: 20px; margin: 0 0 12px; }
h2 { font-size: 15px; margin: 24px 0 8px; color: #555; font-weight: 600; }
.toolbar { position: sticky; top: 0; background: #f6f6f7; padding: 10px 0;
           border-bottom: 1px solid #ddd; z-index: 10; margin-bottom: 16px; }
.toolbar label { margin-right: 16px; font-size: 13px; cursor: pointer; }
.card { background: #fff; border-radius: 8px; padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 16px; }
.card-head { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.badge { padding: 2px 10px; border-radius: 12px; color: #fff; font-size: 11px;
         font-weight: 600; text-transform: uppercase; }
.crop-id { font-family: ui-monospace, "SF Mono", Menlo, monospace;
           font-size: 12px; color: #888; }
.grid { display: grid; grid-template-columns: 350px 1fr 1fr 1fr 1fr; gap: 14px; }
@media (max-width: 1400px) { .grid { grid-template-columns: 280px 1fr 1fr 1fr 1fr; } }
.col { min-width: 0; }
.col-img img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
.col-title { font-size: 11px; font-weight: 600; color: #666;
             text-transform: uppercase; margin-bottom: 4px; display: flex;
             justify-content: space-between; align-items: baseline; }
.cer-pill { background: #eee; padding: 1px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 500; color: #333; }
.cer-good { background: #d4f4dd; color: #1d6c33; }
.cer-mid  { background: #fef0c4; color: #8a6500; }
.cer-bad  { background: #fbd5d5; color: #9b1c1c; }
.text {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word;
  background: #fafafa; padding: 10px; border: 1px solid #eee; border-radius: 4px;
  max-height: 600px; overflow-y: auto;
}
.text .ok  { color: #222; }
.text .sub { background: #ffd6d6; color: #9b1c1c; text-decoration: underline; }
.text .del { background: #ffd6d6; color: #9b1c1c; text-decoration: line-through; }
.text .ins { background: #c9f7d4; color: #1d6c33; }
.meta { font-size: 11px; color: #888; margin-top: 4px; }
.hide { display: none; }
"""

TOOLBAR_JS = """
<script>
function applyFilters() {
  const types = new Set(
    Array.from(document.querySelectorAll('input[name=type]:checked'))
         .map(x => x.value)
  );
  document.querySelectorAll('.card').forEach(card => {
    const t = card.dataset.label;
    card.classList.toggle('hide', !types.has(t));
  });
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('input[name=type]').forEach(el =>
    el.addEventListener('change', applyFilters)
  );
});
</script>
"""


def cer_pill_class(cer: float) -> str:
    if cer < 0.05:
        return "cer-good"
    if cer < 0.15:
        return "cer-mid"
    return "cer-bad"


def render_revue(
    revue: str,
    crops: List[Path],
    crops_dir: Path,
    ocr_outputs_dir: Path,
    engines: List[str],
    out_html: Path,
) -> Dict:
    rows_html: List[str] = []
    stats = defaultdict(lambda: {"sum_edits": 0, "sum_chars": 0, "n": 0})

    for crop_txt in crops:
        meta = load_meta(crop_txt)
        label = meta.get("label", "?")
        page_stem = meta.get("page_stem", crop_txt.stem.split("__", 1)[0])
        region_id = meta.get("region_id", crop_txt.stem.split("__", 1)[-1])
        png_path = crop_txt.with_suffix(".png")
        # path relatif depuis le HTML vers le png (gère le cas siblings)
        img_rel = os.path.relpath(png_path, out_html.parent)

        ref_raw = crop_txt.read_text(encoding="utf-8")
        ref_norm_diff = normalize_for_diff(ref_raw)
        ref_norm_cer = normalize_for_cer(ref_raw)

        color = LABEL_COLORS.get(label, "#888")
        row = []
        row.append(f'<div class="card" data-label="{html.escape(label)}">')
        row.append('<div class="card-head">')
        row.append(f'<span class="badge" style="background:{color}">{html.escape(label)}</span>')
        row.append(f'<span class="crop-id">{html.escape(page_stem)} :: {html.escape(region_id)}</span>')
        row.append(f'<span class="meta">{len(ref_norm_cer)} chars GT</span>')
        row.append('</div>')

        row.append('<div class="grid">')

        # Col 1 : image
        row.append('<div class="col col-img">')
        row.append('<div class="col-title">Crop</div>')
        row.append(f'<img src="{html.escape(str(img_rel))}" alt="crop">')
        row.append('</div>')

        # Col 2 : GT (référence pour le diff)
        row.append('<div class="col">')
        row.append('<div class="col-title">GT (référence)</div>')
        row.append(f'<div class="text">{html.escape(ref_norm_diff)}</div>')
        row.append('</div>')

        # Cols 3+ : chaque moteur
        for engine in engines:
            hp = hyp_path(ocr_outputs_dir, engine, crop_txt, crops_dir)
            if not hp.exists():
                row.append('<div class="col">')
                row.append(f'<div class="col-title">{html.escape(engine)}<span class="cer-pill">—</span></div>')
                row.append('<div class="text" style="color:#aaa">(sortie manquante)</div>')
                row.append('</div>')
                continue
            hyp_raw = hp.read_text(encoding="utf-8")
            hyp_norm_diff = normalize_for_diff(hyp_raw)
            hyp_norm_cer = normalize_for_cer(hyp_raw)

            cer = cer_for(ref_norm_cer, hyp_norm_cer)
            edits = jiwer.process_characters([ref_norm_cer], [hyp_norm_cer])
            n_edits = edits.substitutions + edits.insertions + edits.deletions

            stats[engine]["sum_edits"] += n_edits
            stats[engine]["sum_chars"] += len(ref_norm_cer)
            stats[engine]["n"] += 1

            _ref_html, hyp_html = char_diff_html(ref_norm_diff, hyp_norm_diff)
            pill_class = cer_pill_class(cer)
            row.append('<div class="col">')
            row.append(
                f'<div class="col-title">{html.escape(engine)}'
                f'<span class="cer-pill {pill_class}">CER {cer*100:.1f}%</span></div>'
            )
            row.append(f'<div class="text">{hyp_html}</div>')
            row.append('</div>')

        row.append('</div></div>')
        rows_html.append("".join(row))

    # Bandeau de résumé
    summary_parts: List[str] = ["<h2>Résumé moteur (CER pondéré global pour cette revue)</h2>"]
    summary_parts.append('<div style="font-family:monospace;font-size:13px;">')
    for engine in engines:
        s = stats[engine]
        if s["sum_chars"] > 0:
            cer = s["sum_edits"] / s["sum_chars"]
            summary_parts.append(
                f"<div>{html.escape(engine):<14} n={s['n']:>3}  "
                f"chars={s['sum_chars']:>6}  CER={cer*100:.2f}%</div>"
            )
        else:
            summary_parts.append(f"<div>{html.escape(engine):<14} (aucune sortie)</div>")
    summary_parts.append("</div>")
    summary_html = "".join(summary_parts)

    # Toolbar de filtre
    toolbar = ['<div class="toolbar"><strong>Filtrer :</strong> ']
    for label_name in ["corps", "titre", "auteur", "autre"]:
        toolbar.append(
            f'<label><input type="checkbox" name="type" value="{label_name}" checked> '
            f'{label_name}</label>'
        )
    toolbar.append('</div>')
    toolbar_html = "".join(toolbar)

    html_out = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Comparaison OCR — {html.escape(revue)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>Comparaison qualitative OCR — {html.escape(revue)} ({len(crops)} régions)</h1>
{summary_html}
{toolbar_html}
{"".join(rows_html)}
{TOOLBAR_JS}
</body></html>"""
    out_html.write_text(html_out, encoding="utf-8")
    return stats


def render_index(out_dir: Path, revue_files: Dict[str, Path]) -> None:
    parts = [
        '<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">',
        '<title>Comparaison OCR — Index</title>',
        f'<style>{CSS}</style></head><body>',
        '<h1>Rapports de comparaison qualitative OCR</h1>',
        '<ul style="font-size:14px; line-height:2;">',
    ]
    for revue, path in revue_files.items():
        rel = path.relative_to(out_dir)
        parts.append(f'<li><a href="{html.escape(str(rel))}">{html.escape(revue)}</a></li>')
    parts.append('</ul></body></html>')
    (out_dir / "comparison_index.html").write_text("".join(parts), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Rapport HTML qualitatif OCR.")
    p.add_argument("--crops-dir", required=True, type=Path)
    p.add_argument("--ocr-outputs-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--engines", default="tesseract,paddleocr,pero")
    args = p.parse_args()

    crops_dir = args.crops_dir.expanduser().resolve()
    ocr_dir = args.ocr_outputs_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    crops = list_crops(crops_dir)
    if not crops:
        print(f"Aucun crop sous {crops_dir}", file=sys.stderr)
        return 1

    # Grouper par revue
    by_revue: Dict[str, List[Path]] = defaultdict(list)
    for crop in crops:
        meta = load_meta(crop)
        revue = meta.get("revue", crop.parent.name)
        by_revue[revue].append(crop)

    revue_files: Dict[str, Path] = {}
    for revue, items in by_revue.items():
        out_html = out_dir / f"comparison_{revue}.html"
        print(f"Génération : {out_html.name} ({len(items)} régions)")
        render_revue(revue, items, crops_dir, ocr_dir, engines, out_html)
        revue_files[revue] = out_html

    render_index(out_dir, revue_files)
    index = out_dir / "comparison_index.html"
    print(f"\nIndex : file://{index}")
    print("Ouvre ce fichier dans ton navigateur (les images sont liées en relatif).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
