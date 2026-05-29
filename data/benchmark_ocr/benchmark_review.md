# Jeu de données — benchmark OCR

Dataset constitué pour le benchmark OCR (Tesseract / PaddleOCR / pero-ocr). 3 numéros, 20 premières pages de chacun, soit **60 images au total**.

Source des images : `scraping_pdf/images_process/<revue>/<numero_id>/page_NNNN.png` (pages PNG converties depuis le PDF Gallica).
Source des métadonnées : `scraping_pdf/input/arks_numeros.json` et `scraping_pdf/input/arks_revues.json`.

## Numéros retenus

### 1. *L'Année scientifique et industrielle* — 1876

- **Numero_id** : `annee_scientifique_industrielle1876001_bpt6k201105v`
- **ARK numéro** : `ark:/12148/bpt6k201105v`
- **URL Gallica** : https://gallica.bnf.fr/ark:/12148/bpt6k201105v
- **ARK revue** : `ark:/12148/cb326950838` — https://gallica.bnf.fr/ark:/12148/cb326950838
- **Précision date** : 1876 (jour de l'année = 1 → numéro annuel)
- **Pages disponibles localement** : 583
- **Statut pipeline scraping** : `error / pdf_not_found` dans `arks_numeros.json` (les images sont pourtant bien présentes — métadonnée à rafraîchir)
- **Description** : annuaire de vulgarisation scientifique fondé par Louis Figuier (1857), publication annuelle qui synthétise les progrès des sciences et techniques de l'année écoulée. Mise en page imprimée XIXᵉ, généralement à colonne unique avec illustrations gravées.

### 2. *Bulletin de l'Académie de médecine* — 1883

- **Numero_id** : `bulletin_academie_medecine1883001_bpt6k408673d`
- **ARK numéro** : `ark:/12148/bpt6k408673d`
- **URL Gallica** : https://gallica.bnf.fr/ark:/12148/bpt6k408673d
- **ARK revue** : `ark:/12148/cb34348109k` — https://gallica.bnf.fr/ark:/12148/cb34348109k
- **Précision date** : 1883 (volume annuel)
- **Pages disponibles localement** : 1589
- **Statut pipeline scraping** : `ok / done`
- **Description** : organe officiel de l'Académie nationale de médecine (Paris), publication des séances, mémoires et rapports. Texte dense, pleine page ou deux colonnes selon les rubriques, vocabulaire médical et latinismes fréquents.

### 3. *Revue scientifique* (« Revue rose ») — 1ᵉʳ juillet 1891

- **Numero_id** : `revue_scientifique1891182_bpt6k215115j`
- **ARK numéro** : `ark:/12148/bpt6k215115j`
- **URL Gallica** : https://gallica.bnf.fr/ark:/12148/bpt6k215115j
- **ARK revue** : `ark:/12148/cb34378388w` — https://gallica.bnf.fr/ark:/12148/cb34378388w
- **Précision date** : 01 juillet 1891 (jour de l'année = 182 → semestre 2)
- **Pages disponibles localement** : 835
- **Statut pipeline scraping** : `error / pdf_not_found` dans `arks_numeros.json` (images présentes — métadonnée à rafraîchir)
- **Description** : revue hebdomadaire de vulgarisation scientifique fondée en 1863 (« Revue des cours scientifiques » puis *Revue scientifique*), surnommée « Revue rose » à cause de sa couverture. Mise en page typique deux colonnes serrées, articles de fond + chronique scientifique.

## Note sur les métadonnées

Pour deux des trois numéros (`annee_scientifique_industrielle 1876` et `revue_scientifique 1891182`), le champ `status` dans `arks_numeros.json` est resté à `error / pdf_not_found` alors que les images sont bien présentes dans `images_process/`. Le PDF intermédiaire a vraisemblablement été supprimé après conversion sans mise à jour du JSON. À fixer côté pipeline scraping si besoin (les images, elles, sont exploitables sans souci pour le benchmark).

---

# Structure du dossier `data/benchmark_ocr/`

```
data/benchmark_ocr/
├─ benchmark_review.md           # ce fichier (jeu de données + méthodo + pipeline)
├─ images_dataset/               # 60 pages source, organisées par numéro
│  ├─ annee_scientifique_industrielle1876_bpt6k201105v/   # 20 pages
│  ├─ bulletin_academie_medecine1883_bpt6k408673d/        # 20 pages
│  └─ revue_scientifique1891182_bpt6k215115j/             # 20 pages
├─ label_studio/                 # config + exports d'annotation
│  ├─ config.xml                 # config Label Studio (cf. § Annotation)
│  └─ exports/                   # exports JSON après annotation
├─ crops/                        # crops par zone (générés depuis l'export)
│  └─ <revue>/page_NNNN__zone_K.png
│  └─ <revue>/page_NNNN__zone_K.txt   # transcription GT correspondante
├─ ocr_outputs/                  # sorties brutes des moteurs (générées)
│  ├─ tesseract/                 # Tesseract `fra` --psm 6
│  ├─ paddleocr/                 # PaddleOCR server_det, max_side=2000
│  ├─ paddleocr_mobile/          # PaddleOCR mobile_det, max_side=2000
│  ├─ paddleocr_mobile_fullres/  # PaddleOCR mobile_det, sans downscale
│  │                              # (subprocess-per-crop, plus lent mais robuste)
│  └─ pero/                      # pero-ocr european_printed (config_cpu.ini)
└─ results/                      # tables d'évaluation (générées)
   └─ results.csv
```

Les sous-dossiers `crops/`, `ocr_outputs/` et `results/` sont produits par les scripts d'évaluation, pas annotés à la main.

---

# Méthodologie d'annotation

## Outil et schéma

- Outil : **Label Studio** (en local, http://localhost:8080).
- Une seule tâche par page image. Annotation **bloc par bloc** (= zone logique : colonne, encadré, légende, tableau, titre d'article, etc.). Pas d'annotation au niveau ligne ni article.
- 3 à 8 blocs par page typique → ~300 blocs au total sur les 60 pages.
- Granularité = grain `TextRegion` ICDAR.

## Catégories de blocs

| Label | Usage |
|---|---|
| `corps` | colonne(s) de corps d'article |
| `titre` | **titre d'article uniquement** (pas le titre de revue, pas les rubriques publicitaires) |
| `auteur` | nom de l'auteur d'un article (souvent juste sous le titre, en italique) |
| `autre` | légendes de figures, cellules de tableaux, et tout autre texte éditorial qu'on veut océriser sans le ranger ailleurs |

## Config Label Studio

À coller dans *Labeling Interface → Code* :

```xml
<View>
  <Image name="image" value="$ocr"/>

  <Labels name="label" toName="image">
    <Label value="corps" background="#3338fb"/>
    <Label value="titre" background="#d50008"/>
    <Label value="auteur" background="#FFC069"/>
    <Label value="autre" background="#00aa2a"/>
  </Labels>

  <Rectangle name="bbox" toName="image" strokeWidth="3"/>
  <Polygon name="poly" toName="image" strokeWidth="3"/>

  <TextArea name="transcription" toName="image" editable="true" perRegion="true"
            required="true" maxSubmissions="1" rows="5"
            placeholder="Recognized Text" displayMode="region-list"/>
</View>
```

## Conventions de transcription

- **Fidélité absolue** au texte imprimé : pas de correction silencieuse des coquilles d'époque.
- **Césures conservées** : tirets de fin de ligne tapés tels quels, suivis directement d'un saut de ligne (`Entrée`).
- **Pas d'espace en bout de ligne** avant le saut.
- **Saut de ligne typographique** (sans tiret) : `Entrée` direct, sans espace avant.
- **Titre sur plusieurs lignes** (mise en page) : un seul bloc, retours à la ligne dans la TextArea (cf. ex. « L'ANNÉE / SCIENTIFIQUE / ET INDUSTRIELLE »).
- **Ligatures `œ`, `æ`** : conservées telles que.
- **`s` long historique (`ſ`)** : translittéré en `s` standard (sinon aucun OCR ne saura aligner).
- **Caractères illisibles** : `[?]` (1 marqueur par caractère manquant).

## Périmètre — ce qui s'annote vs ce qui s'ignore

| Annoter | Ignorer (= pas de bbox, pas de transcription) |
|---|---|
| Colonnes de corps d'article | En-têtes courants (titre courant en haut de page) |
| Titres d'articles | Pieds de page, numéros de page |
| Noms d'auteurs | Ex-libris, cachets de bibliothèque |
| Légendes de figures | Signatures manuscrites |
| Tableaux (cellule = bloc `autre`) | Illustrations sans texte |
| Colophons éditoriaux pertinents | Publicités, catalogues éditeur |
|  | Ornements typographiques |
|  | Titre de revue en page de garde (= `autre` si annoté, ou ignorer) |
|  | Texte disposé en cercle (cachets) ou rotation > 5° |

Règle générale : **ce qui n'est pas dans une box GT n'est pas évalué**. Les moteurs liront quand même tout, mais leur sortie hors zones est filtrée automatiquement (ou plutôt, jamais examinée — voir pipeline ci-dessous).

---

# Pipeline d'évaluation

## Approche : OCR sur crops par zone (style ICDAR région)

Plutôt que d'OCR-iser la page entière puis de filtrer la sortie par zone, on **découpe l'image originale en crops correspondant aux bbox annotées**, et on demande à chaque moteur de transcrire uniquement ce crop. Avantages :

- Isole la **qualité OCR pure** de la layout analysis.
- Élimine l'ambiguïté de reading order (1 bloc = 1 colonne = top-to-bottom trivial).
- Aucun risque que la sortie OCR capture du texte hors zone.
- Pipeline d'évaluation simplifié (1 crop ↔ 1 référence).

Limite assumée : on n'évalue pas la capacité des moteurs à *trouver* le texte. C'est un choix de scope cohérent avec l'objectif (« lequel transcrit le mieux ? »).

### Filiation méthodologique

Cette approche reprend la méthodologie d'évaluation OCR par région du PRImA Research Lab (groupe Antonacopoulos / Pletschacher / Clausner), telle que formalisée dans les compétitions ICDAR sur documents historiques. Référence centrale :

> Clausner, C., Antonacopoulos, A., Derrick, T., & Pletschacher, S. (2017). *ICDAR2017 Competition on Recognition of Early Indian Printed Documents – REID2017*. In *14th IAPR International Conference on Document Analysis and Recognition (ICDAR)*, Kyoto, Japan, pp. 1411–1416. DOI : [10.1109/ICDAR.2017.230](https://doi.org/10.1109/ICDAR.2017.230). PDF : [primaresearch.org](https://www.primaresearch.org/www/assets/papers/ICDAR2017_Clausner_REID2017.pdf).

Le papier justifie explicitement le découplage de l'évaluation OCR vis-à-vis de l'ordre de lecture global (Section IV.B, *Text Recognition*) :

> « A major problem for the evaluation is the influence of the reading order of text regions. For simple page layouts, the order is obvious, but for more complex layouts, the reading order can be ambiguous. In such cases, measures that are affected by the reading order are less meaningful. […] Special care was therefore taken when selecting the evaluation measures. »

Et la pondération par taille de zone (Section IV.A, *Layout Analysis*), qu'on transpose ici en pondération par longueur de texte :

> « the errors are also weighted by the size of the area affected […]. In this way, a missed region corresponding to a few characters will have less influence on the overall result than a miss of a whole paragraph. »

L'outillage de référence est le **PRImA Performance Evaluation Toolkit** (http://www.primaresearch.org/tools/PerformanceEvaluation), basé sur le format PAGE XML (Pletschacher & Antonacopoulos, *The PAGE (Page Analysis and Ground-truth Elements) Format Framework*, ICPR 2010, DOI [10.1109/ICPR.2010.72](https://doi.org/10.1109/ICPR.2010.72)). Notre pipeline ne réutilise pas directement cet outil — on calcule CER/WER avec `jiwer` après crop — mais la philosophie d'évaluation est la même : **région → métrique de distance d'édition → agrégation pondérée par longueur**.

## Schéma du pipeline

```
images_dataset/<revue>/page_NNNN.png       (60 pages)
        │
        │ Label Studio → export JSON
        ▼
crops/<revue>/page_NNNN__zone_K.png        (~300 crops, 1 par bloc)
crops/<revue>/page_NNNN__zone_K.txt        (transcription GT correspondante)
        │
        │ pour chaque crop, 3 moteurs
        ▼
ocr_outputs/{tesseract,paddleocr,pero}/<revue>/page_NNNN__zone_K.txt
        │
        │ jiwer (CER/WER) crop par crop
        ▼
results.csv : revue, page, zone, type_zone, moteur, n_chars_ref, n_edits, CER, WER
        │
        │ agrégation globale + par moteur + par type_zone (corps/titre/auteur/autre)
        ▼
3 chiffres CER, 1 par moteur, ± décomposition par type / par revue
```

## Précautions techniques

- **Marge** ~10-20 px autour de chaque bbox au moment du crop (éviter de couper hampes et jambages).
- **Résolution native** conservée au stockage. Les crops vont de 0,1 Mpx à ~15 Mpx (médiane ~2 Mpx).
- **Coordonnées Label Studio** : l'export JSON donne `x`, `y`, `width`, `height` en **pourcentage** de l'image originale → conversion en pixels au moment du crop.
- **Paramètres OCR adaptés au mono-bloc** :
  - **Tesseract** : `--psm 6` (uniform block of text), au lieu du `--psm 3` par défaut.
  - **PaddleOCR** : downscale interne à `max_side=2000` (cf. ci-dessous).
  - **pero-ocr** : `config_cpu.ini` du modèle `pero_eu_cz_print_newspapers_2022-09-26`, layout natif activé.

## Moteurs benchmarqués et contraintes hardware

| Moteur | Modèle / params | Downscale d'entrée | Run mode |
|---|---|---|---|
| `tesseract` | binaire `tesseract` v5, `-l fra --psm 6` | aucun | in-process |
| `paddleocr` | `PP-OCRv5_server_det` + `latin_PP-OCRv5_mobile_rec` | max_side=2000 px | in-process |
| `paddleocr_mobile` | `PP-OCRv5_mobile_det` + `latin_PP-OCRv5_mobile_rec` | max_side=2000 px | in-process |
| `paddleocr_mobile_fullres` | `PP-OCRv5_mobile_det` + `latin_PP-OCRv5_mobile_rec` | aucun (full-res) | **subprocess-par-crop** |
| `pero` | `pero_eu_cz_print_newspapers_2022-09-26`, ParseNet + OCR transformer | aucun (gère interne) | in-process |

### Note sur `paddleocr_fullres` (server sans downscale) — non viable sur ce Mac

On a tenté un 5ᵉ moteur `paddleocr_fullres` (= `PP-OCRv5_server_det` sans downscale). Résultat sur ce matériel :

- **3 premiers gros crops évalués → 3 échecs en chaîne** :
  - `page_0008__0mje1O4qnH.png` (corps, ~7 Mpx) → timeout 10 min, subprocess `UN` à 0.9 % CPU (swap intense).
  - `page_0009__obdsBqW7B8.png` (corps, ~6 Mpx) → idem timeout 10 min.
  - `page_0010__GTbEEobiL1.png` (corps, ~7 Mpx, > 4000 px côté max) → `exit -9` (OOM jetsam macOS).
- Sur les 20 premiers crops (petits, < 2 Mpx) le moteur fonctionnait ; il s'effondre dès que le crop excède ~4-5 Mpx.

**Conclusion** : le couple `PP-OCRv5_server_det` + full-res n'est pas exploitable sur un Mac Apple Silicon sans GPU pour ce corpus (crops jusqu'à 15 Mpx). Les sorties partielles ont été supprimées. La config retenue en production est `paddleocr` avec `max_side=2000 px` (compromis qualité / RAM documenté dans [paddleocr_ocr.py](../scripts_notebooks/scripts_benchmark_ocr/paddleocr_ocr.py)).

À titre comparatif on garde **`paddleocr_mobile_fullres`** (modèle mobile_det, full-res) qui est plus léger en RAM et passe en subprocess-par-crop : il permet de mesurer le coût/bénéfice d'absorber la résolution native avec un détecteur allégé.

## Métriques

- **CER** (Character Error Rate) = distance de Levenshtein normalisée par la longueur de la référence : `CER = (S + D + I) / N`. Calcul via `jiwer.process_characters`.
- **WER** (Word Error Rate) : idem au niveau mot. Plus sévère, plus parlant pour la lisibilité.
- **Agrégation pondérée par longueur** : `CER_global = Σ(S+D+I) / Σ N`, **pas** la moyenne des CER par bloc (ce qui biaiserait vers les petits blocs).

## Normalisations préalables au calcul

À appliquer **identiquement** sur référence et sortie OCR avant le calcul :

- Normalisation Unicode **NFC**.
- Espaces multiples → simple espace.
- Sauts de ligne `\n` → simple espace (variante 1) **ou** conservés (variante 2, pour mesurer la restitution de mise en page).
- Casse : **conservée** (les majuscules portent du sens en presse XIXᵉ).
- Césures : on calcule **deux variantes** — `brut` (avec `-\n`) et `joint` (`-\n` → `''`).

## Sortie attendue

Une table `results/results.csv` avec une ligne par (page × zone × moteur × variante), puis :

- **CER global par moteur** (4-5 chiffres principaux).
- **CER par moteur × type_zone** (4 colonnes : corps / titre / auteur / autre).
- **CER par moteur × revue** (3 colonnes : annee_scientifique / bulletin_medecine / revue_scientifique).
- Idem pour WER.
- **Temps d'inférence par moteur** : `results/timing_summary.txt` (médiane, p95, projections 1k/100k/1M crops).
- **Rapport HTML qualitatif** : `results/comparison_*.html` (1 par revue) — affiche pour chaque crop son image + GT + sortie de chaque moteur avec **diff caractère par caractère colorisé**.

C'est le tableau qu'on commentera dans le mémoire pour justifier le choix du moteur en production.
