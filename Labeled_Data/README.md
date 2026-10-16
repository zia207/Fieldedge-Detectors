# Updated analysis with labelled data: field-boundary model training and testing

Inputs, trained models and code of the training study, which compares the active-contour (snake)
pipeline, FTW Delineate Anything V2, SAM 2.1 and SAM v1, trained and tuned on labelled Google
satellite-image screenshots.

## Contents

| Path | What it is |
|---|---|
| `build_training_notebook.py` | Generates the notebook (single source of its cells) |
| `run_training.sh` | Builds, executes and renders the notebook (full retrain when `train_outputs/` is empty) |
| `render.sh` | Renders the HTML and the code-free PDF |
| `field_boundary_training.ipynb`, `.html`, `_report.pdf` | Executed notebook, report with code, report without code |
| **Input data** (`Data/`, local only) | |
| `Data/{Train,Valid,Test}/sw_xx/Tiles/*.TIF` | Georeferenced screenshots of Google satellite imagery (WGS 84 lon/lat, EPSG:4326, ≈ 1.8 cm/px; effective ≈ 0.2–0.3 m). Warped to BUTM-2010 (EPSG:9680) at 0.25 m when prepared; tiles already in a metric CRS are read directly |
| `Data/{Train,Valid,Test}/sw_xx/Polyline/*.shp` | Hand-digitised field-boundary polylines, one shapefile per tile (any CRS) |
| **Models** | |
| `checkpoints/DelineateAnythingv2.pt` | Pre-trained FTW Delineate Anything V2 (YOLO11x-seg) |
| `checkpoints/sam_vit_b_01ec64.pth` | Pre-trained SAM v1 ViT-B (SHA-256 `ec2df627…`) |
| `train_outputs/da/runs/finetune/weights/best.pt` | **Fine-tuned** Delineate Anything V2 (best epoch on Valid); used by `FieldEdge_App` |
| `train_outputs/da/runs/finetune/weights/last.pt` | Fine-tuned Delineate Anything V2, last epoch (`args.yaml`, `results.csv`: training settings and history) |
| `train_outputs/sam2/sam2_decoder_finetuned.pt` | **Fine-tuned** SAM 2.1 mask decoder + prompt encoder (load on top of `facebook/sam2.1-hiera-large`) |
| `train_outputs/sam1/sam1_decoder_finetuned.pt` | **Fine-tuned** SAM v1 mask decoder + prompt encoder (load on top of `sam_vit_b_01ec64.pth`) |

The selected snake-pipeline setting is H = 0.08, T_edge = 0.2, T_ΔE = 5.

Intermediate outputs in `train_outputs/` (`prep/`, `snake/`, `da/yolo_dataset/`, `pred/`, `vectors/`, `tables/`, `figures/`) are not in git. The final Test GeoJSONs used by the web map are copied to `../FieldEdge_App/web/data/`. `train_outputs_v1/` (local only) holds the models of the previous run on the BUTM copy of the same data.

To add new tiles, for example from the app's **Training data** export, put each `Tiles/` + `Polyline/` pair in a site folder under `Data/<split>/`, empty `train_outputs/`, and re-run.

SAM 2.1 base weights are fetched from Hugging Face (`facebook/sam2.1-hiera-large`) and are not copied here.

## Re-run

Uses the repository `.venv` (Python 3.12) with its Jupyter kernel `fieldedge`
(`.venv/bin/python -m ipykernel install --sys-prefix --name fieldedge`):

```bash
cd Labeled_Data
./run_training.sh > training_run.log 2>&1
```
Every stage is cached in `train_outputs/`; a re-run skips completed stages. Move or empty `train_outputs/`
to retrain all models. A full run takes about 2–3 h on a 16 GB laptop GPU.

Google satellite imagery is subject to Google's Terms of Service, which restrict bulk download and reuse.
