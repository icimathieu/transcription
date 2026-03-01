# Transcription OCR pour mémoire de master

Ce dépôt contient des scripts et des résultats d'expérimentation OCR réalisés dans le cadre de mon mémoire de master.

Objectif: transcrire efficacement des pages imprimées du XIXe siècle (mise en page en colonnes), avec une contrainte forte de temps de calcul sur machine CPU.

## Structure du dépôt

```text
transcription/
├─ data/
│  ├─ data_to_git/        # jeux d'images publiés avec le dépôt
│  └─ data_not_to_git/    # données locales non publiées
├─ output/
│  ├─ tesseract_boxes/    # sorties OCR Tesseract + colonnes reconstruites
│  ├─ paddleocr_boxes/    # sorties OCR PaddleOCR + colonnes reconstruites
│  ├─ paddleocr_simple/   # sorties pipeline Paddle + PP-StructureV3 (non publiées)
│  └─ paddleocr_vl_test/  # sorties PaddleOCR-VL (non publiées)
├─ scripts_notebooks/
│  ├─ tesseract_boxes.py
│  ├─ paddleocr_boxes.py
│  ├─ paddleocr_cpu_simple.py (non publié)
│  ├─ paddleocr_vl_test.py (non publié)
│  ├─ pdf2image.py
│  └─ pdf2image.ipynb
├─ notes_ocr_memoire.md (non publié)
├─ requirements_min
└─ requirements
```


## Workflows testés

### 0) Workflow exploratoire notebook/script (PDF -> images -> OCR)
- Fichiers: `scripts_notebooks/pdf2image.ipynb` et `scripts_notebooks/pdf2image.py`.
- Rôle dans le projet:
  - conversion PDF -> images,
  - premiers essais PaddleOCR,
  - essais PP-StructureV3,
  - essais PaddleOCR-VL,
  - export intermédiaire JSON/JSONL.
- Ce workflow a servi de base d'exploration avant la stabilisation des scripts dédiés `paddleocr_*` et `tesseract_boxes.py`.
- Limite observée : nous avons décidés de scraper directement des .png ou .jpg et non de convertir le pdf en images.

### 1) Pipeline PaddleOCR-VL
- Script: `scripts_notebooks/paddleocr_vl_test.py`
- Intérêt: compréhension de documents avancée.
- Limite observée: trop lourd pour un usage massif sur CPU local (temps d'inférence élevé, overhead important). Abandonné très vite.

### 2) Pipeline PaddleOCR "classique" + PP-StructureV3 / bounding boxes
- Scripts: `scripts_notebooks/paddleocr_cpu_simple.py`, puis `scripts_notebooks/paddleocr_boxes.py`
- Intérêt: meilleure qualité OCR que des moteurs plus légers sur cas difficiles.
- Limite observée: temps de calcul important sur CPU (environ ~1 minute sur certaines images 1024 px, plus sur images plus grandes).
- Utiliser PP-StructureV3 ou de la reconstruction manuelle avec bounding box est assez équivalent dans le résultat mais PP-Structure alourdit le process en calcul et en temps.

### 3) Pipeline Tesseract + bounding boxes + reconstruction de colonnes
- Script: `scripts_notebooks/tesseract_boxes.py`
- Principe:
  - OCR Tesseract en TSV (mots + boîtes),
  - agrégation en lignes,
  - détection automatique des colonnes,
  - reconstruction de l'ordre de lecture.
- Résultat pratique: vitesse nettement meilleure sur CPU -- quelques secondes par page selon la taille : de 2 secondes pour une image à deux colonnes 512 pixels de largeur à 3 secondes pour la même image en full res. Semble adéquat pour de gros volumes.

## Conclusion méthodologique

La pipeline Tesseract a été retenue comme solution principale pour ce corpus, pour des raisons pragmatiques:
- exécution beaucoup plus rapide sur CPU;
- moteur bien optimisé pour des documents imprimés;
- corpus majoritairement simple du point de vue visuel (imprimé XIXe, peu d'éléments graphiques complexes);
- besoin de traiter un grand nombre de pages dans les délais d'un mémoire.
- L'utilisation d'un modèle IA ne semble pas nécessaire, ni *a fortiori*, d'un VLM.

## Suite

1. J'envisage de faire une pipeline pour trouver la qualité d'image à scraper optimale.
2. Il faut que j'industrialise un peu la chaîne de traitement.
3. Il faut que je fasse des tests de qualité d'OCR en annotant manuellement certaines données.
2. Il va falloir industrialiser le processus, en faire un script au sein d'une pipeline qui tourne en back sur mon ordinateur pour océriser mon (grand) volume de données.

## Choix qualité d'image

Des tests ont été réalisés sur différentes qualités/résolutions d'image (dont versions bitonales).

Constats:
- qualité trop basse => OCR dégradé (erreurs de caractères, perte de lignes);
- meilleure qualité => meilleur OCR;
- conversion bitonale utile sur ce type de pages imprimées car permet de réduire l'information sans impacter le reste de la pipeline. 
- Tesseract est plus sensible que PaddleOCR aux variations de qualité.
- Conclusion : Étant donné que mes données proviennent de Gallica et que celle-ci indistingue le scraping sur les images de plus de 1000x, j'envisage de scraper en full res. 
- J'envisage malgré tout de faire une pipeline simple pour évaluer la qualité d'image optimale par rapport à 1 - des contraintes de consommation internet et 2 - de qualité d'OCR.

## Reproduire rapidement

## Prérequis système
- Python 3.9+ avec venv
- `tesseract` installé sur la machine (accessible dans le `PATH`)

## Installation
```bash
git clone https://github.com/icimathieu/transcription
cd transcription
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_min
```

## Lancer la pipeline recommandée (Tesseract)
```bash
source .venv/bin/activate
python scripts_notebooks/tesseract_boxes.py \
  --image data/data_to_git/bitonal_1024.png \
  --tesseract-bin "$(command -v tesseract)"
```

Sorties:
- `output/tesseract_boxes/*_raw_lines.json`
- `output/tesseract_boxes/*_ordered_lines.json`
- `output/tesseract_boxes/*_full_text.txt`
- `output/tesseract_boxes/*_meta.json`

## Option: reproduire les tests Paddle
Installer dépendances complètes:
```bash
pip install -r requirements
```

Puis utiliser:
- `scripts_notebooks/paddleocr_boxes.py`
- `scripts_notebooks/paddleocr_cpu_simple.py`
- `scripts_notebooks/paddleocr_vl_test.py`

## Reproduire les sorties publiées

Pour générer les mêmes types de sorties que celles versionnées dans `output/`:

```bash
source .venv/bin/activate
for img in data/data_to_git/*.png; do
  python scripts_notebooks/tesseract_boxes.py \
    --image "$img" \
    --tesseract-bin "$(command -v tesseract)"
done
```

Les fichiers seront écrits dans `output/tesseract_boxes/` avec le même schéma de nommage:
- `<nom_image>_raw_lines.json`
- `<nom_image>_ordered_lines.json`
- `<nom_image>_full_text.txt`
- `<nom_image>_meta.json`

## Données et sorties

Les dossiers `data/` et `output/` utiles à la reproductibilité ont été rendus disponibles sur le dépôt GitHub.

## Confidentialité et usage d'outils IA

Ce travail a été réalisé en mode **opt-out** pour la confidentialité des données.

Les assistants **Codex** et **ChatGPT** ont été utilisés comme tiers aide technique (scripting, tests, documentation), sous supervision humaine.

## Licence

Le projet est distribué sous la licence Apache 2.0 ajoutée au dépôt (voir le fichier de licence à la racine du repo).
