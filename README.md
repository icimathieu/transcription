# Transcription OCR pour mémoire de master

Ce dépôt contient les scripts et résultats d'expérimentation OCR réalisés dans le cadre de mon mémoire de master.

Objectif : transcrire efficacement des pages imprimées du XIXᵉ siècle (mise en page souvent en colonnes), sous contrainte de temps de calcul sur machine CPU.

## Structure du dépôt

```text
transcription/
├─ data/
│  ├─ data_to_git/             # jeu d'images de démonstration publié avec le dépôt
│  ├─ data_not_to_git/         # données locales non publiées
│  └─ benchmark_ocr/           # corpus, annotations et sorties du benchmark multi-moteurs
│     ├─ benchmark_review.md   # méthodologie, dataset, résultats détaillés
│     ├─ ocr_outputs/          # sorties textuelles par moteur (Tesseract, Paddle, Pero…)
│     ├─ results/              # comparaisons HTML, CSV, résumés CER/WER, timings
│     ├─ truth_dataset_ls/     # vérité terrain (export Label Studio)
│     ├─ images_dataset/       # (non publié — corpus d'images source)
│     └─ crops/                # (non publié — crops par zone)
├─ output/                     # généré localement à l'exécution, non publié
├─ scripts_notebooks/
│  ├─ scripts_benchmark_ocr/   # scripts du benchmark multi-moteurs
│  └─ archives/                # scripts d'exploration / pipelines antérieures
│     ├─ tesseract_boxes.py        # pipeline page-entière multi-colonnes
│     ├─ paddleocr_boxes.py        # exploration Paddle + reconstruction colonnes
│     ├─ pdf2image.ipynb           # notebook d'exploration PDF -> images -> OCR
│     ├─ paddleocr_cpu_simple.py   (non publié)
│     ├─ paddleocr_vl_test.py      (non publié)
│     └─ pdf2image.py              (non publié)
├─ notes_ocr_memoire.md        (non publié)
└─ requirements
```

## Workflows testés

Cinq workflows OCR ont été explorés au fil du projet :

1. **Notebook d'exploration PDF → images → OCR** — `archives/pdf2image.ipynb`, `archives/pdf2image.py`. Conversion PDF, premiers essais Paddle / PP-StructureV3.
2. **PaddleOCR-VL** — `archives/paddleocr_vl_test.py`. Vision-language model, abandonné rapidement (trop coûteux sur CPU local).
3. **PaddleOCR classique + bounding boxes** — `archives/paddleocr_cpu_simple.py`, `archives/paddleocr_boxes.py`. Qualité correcte mais temps de calcul élevé (~1 min par page CPU).
4. **Tesseract + reconstruction de colonnes** — `archives/tesseract_boxes.py`. Pipeline page-entière, quelques secondes par page CPU.
5. **Pero-OCR** — intégré plus tard, dans le cadre du benchmark. Modèle `pero_eu_cz_print_newspapers_2022-09-26`. Cf. `scripts_notebooks/scripts_benchmark_ocr/pero_engine.py`.

## Benchmark multi-moteurs

Un benchmark formel a été conduit sur **60 pages** issues de 3 numéros de presse XIXᵉ (*L'Année scientifique et industrielle* 1876, *Bulletin de l'Académie de médecine* 1883, *Revue scientifique* 1891), avec annotation Label Studio bloc par bloc (~300 zones GT).

Méthodologie inspirée des compétitions ICDAR (PRImA Research Lab) : OCR effectué sur **crops par zone** (et non sur la page entière), CER/WER calculés via `jiwer`, agrégation pondérée par longueur de référence.

Moteurs évalués : Tesseract, PaddleOCR (server), PaddleOCR mobile, PaddleOCR mobile full-res, Pero-OCR.

Synthèse complète (dataset, méthodologie d'annotation, contraintes hardware, métriques, projections temps) : [`data/benchmark_ocr/benchmark_review.md`](data/benchmark_ocr/benchmark_review.md). Sorties brutes et tables d'évaluation sous [`data/benchmark_ocr/ocr_outputs/`](data/benchmark_ocr/ocr_outputs/) et [`data/benchmark_ocr/results/`](data/benchmark_ocr/results/).

Scripts du benchmark : [`scripts_notebooks/scripts_benchmark_ocr/`](scripts_notebooks/scripts_benchmark_ocr/).

## Reproduire

### Prérequis système
- Python 3.9+
- `tesseract` installé (accessible dans le `PATH`)

### Installation
```bash
git clone https://github.com/icimathieu/transcription
cd transcription
python -m venv .venv
source .venv/bin/activate
pip install -r requirements
```

### Pipeline page-entière (Tesseract, `archives/`)
```bash
source .venv/bin/activate
python scripts_notebooks/archives/tesseract_boxes.py \
  --image data/data_to_git/bitonal_1024.png \
  --tesseract-bin "$(command -v tesseract)"
```

Sorties écrites dans `output/tesseract_boxes/` (non versionné), suffixées `_raw_lines.json`, `_ordered_lines.json`, `_full_text.txt`, `_meta.json`.

Pour traiter l'ensemble du corpus de démonstration :

```bash
for img in data/data_to_git/*.png; do
  python scripts_notebooks/archives/tesseract_boxes.py \
    --image "$img" \
    --tesseract-bin "$(command -v tesseract)"
done
```

### Relancer le benchmark multi-moteurs
Voir [`scripts_notebooks/scripts_benchmark_ocr/run_benchmark.py`](scripts_notebooks/scripts_benchmark_ocr/run_benchmark.py) et la section *Pipeline d'évaluation* de [`benchmark_review.md`](data/benchmark_ocr/benchmark_review.md) pour les commandes et l'installation des moteurs additionnels (PaddleOCR, pero-ocr).

## Données

- `data/data_to_git/` — corpus de démonstration pour la pipeline page-entière.
- `data/benchmark_ocr/` — dataset, annotations, sorties OCR et résultats du benchmark. Les sous-dossiers `images_dataset/` (images source) et `crops/` (crops par zone) ne sont pas versionnés.
- `output/` — sorties générées localement à l'exécution, non versionné.

## Licence

Apache 2.0 — voir [`LICENSE`](LICENSE).
