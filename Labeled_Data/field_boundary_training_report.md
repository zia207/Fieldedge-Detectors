# Training and Testing Field-Boundary Models on Google Satellite-Image Screenshots

Active contours · FTW Delineate Anything V2 · SAM · SAM 2

Author: Zia U. Ahmed, Ph.D. · Uppata Analytics LLC

## Objectives

1. Prepare a training / validation / test dataset from georeferenced Google satellite-image screenshots with hand-digitised field-boundary polylines (10 sites, 58 tiles).
2. Train or tune four field-delineation approaches on the Train sites and select settings and checkpoints on the Valid sites.
3. Evaluate every model before and after training on the held-out Test sites, using boundary accuracy (boundary F1 at 1 m and 2 m tolerance) and field-instance accuracy (precision / recall / F1 at IoU > 0.5).
4. Interpret what training changed, and which approach is best for operational field mapping.

| Model | Type | What is learned |
| --- | --- | --- |
| Active-contour (snake) pipeline | classical, unsupervised | segmentation parameters tuned by grid search |
| FTW Delineate Anything V2 | instance segmentation (YOLO11x-seg) | full network fine-tuned |
| SAM 2.1 (Hiera-Large) | promptable foundation model | mask decoder + prompt encoder fine-tuned |
| SAM (v1, ViT-B) | promptable foundation model | mask decoder + prompt encoder fine-tuned |

### Notebook map

| Part | Content |
| --- | --- |
| 0 | Environment and configuration |
| 1 | Data: inventory, preparation at 0.25 m, ground truth |
| 2 | Evaluation protocol |
| 3 | Model A: active-contour (snake) pipeline — tuning |
| 4 | Model B: FTW Delineate Anything V2 — fine-tuning |
| 5 | Model C: SAM 2.1 — decoder fine-tuning |
| 6 | Model D: SAM (v1) — decoder fine-tuning |
| 7 | Test-set evaluation, map export, then Steger lines and snake refinement |
| 8 | Interpretation and conclusions |

## Part 0 · Environment and configuration

Runs in the repository `.venv` environment (Python 3.12, PyTorch 2.10 + CUDA 12.8, `ftw-tools` 2.0.0b5, `ultralytics`, `sam2`, `segment-anything`). The labelled data are read from `../Labeled_Data/`. All outputs and models of this study live in `Updated_analysis_labeled_data/`. Every stage caches its outputs under `train_outputs/`: prepared tiles, tuning tables, trained weights and predictions. A re-run therefore skips completed training.

Python 3.12.15 · torch 2.10.0+cu128 (CUDA 12.8) · GPU: NVIDIA RTX A5500 Laptop GPU

| Package | Version |
| --- | --- |
| ftw-tools | 2.0.0b5 |
| ultralytics | 8.4.174 |
| sam2 | 1.1.0 |
| segment-anything | 1.0 |
| scikit-image | 0.26.0 |
| rasterio | 1.5.2 |
| geopandas | 1.2.0 |

Working GSD 0.25 m · minimum parcel 320 px.

## Part 1 · Data

### 1.1 Inventory

Each site folder `sw_xx` holds screenshots of Google satellite imagery (`Tiles/*.TIF`, RGB), georeferenced in WGS 84 longitude/latitude (EPSG:4326) and covering about 286 m × 206 m each. They are warped to BUTM-2010 (EPSG:9680) when prepared. The GeoTIFFs are stored at a nominal 1.8 cm/pixel, but that is the resolution of the screenshot export, not of the imagery: the underlying Google imagery is much coarser (see 1.2). For each tile there is a shapefile of hand-digitised field-boundary polylines (`Polyline/<tile>.shp`). The split is by site: no site appears in more than one split, so Valid and Test measure transfer to unseen locations.

Test/sw_09: shapefiles without an image (ignored): `sw_094`, `sw_095`.

| Split | Site | Tiles | Resolution (cm) | Extent (m) |
| --- | --- | --- | --- | --- |
| Test | sw_09 | 4 | 1.9 | 312 × 209 |
| Test | sw_10 | 6 | 1.9 | 312 × 209 |
| Train | sw_01 | 6 | 1.9 | 312 × 209 |
| Train | sw_02 | 6 | 1.9 | 312 × 209 |
| Train | sw_03 | 6 | 1.9 | 313 × 209 |
| Train | sw_04 | 6 | 1.9 | 312 × 209 |
| Train | sw_05 | 6 | 1.9 | 312 × 209 |
| Train | sw_06 | 6 | 1.9 | 312 × 209 |
| Valid | sw_07 | 6 | 1.9 | 312 × 209 |
| Valid | sw_08 | 6 | 1.9 | 313 × 209 |

### 1.2 Preparation at 0.25 m and ground truth

**Resampling.** The screenshots are stored at a nominal 1.8 cm/pixel (about 16 000 × 11 500 px per tile), but Google satellite imagery is far coarser, so most of these pixels are interpolated. Reducing a tile to 0.14–0.29 m and enlarging it back loses only about 0.03–0.2% of the image variance. Real detail starts to be lost only around 0.4 m (about 0.3–1%). The effective resolution is therefore about 0.2–0.3 m, and every tile is resampled by pixel averaging to 0.25 m (about 1 144 × 826 px) with essentially no loss of information. At 0.25 m a 15 m wide plot is 60 px across, and 0.25 m is inside Delineate Anything's training range (0.25–10 m). Nodata borders are masked.

**Ground-truth boundaries.** The polylines are clipped to the tile and rasterised (1 px, all touched). This raster is the reference for the boundary metrics.

**Ground-truth field instances.** Digitised lines do not always meet exactly and run off the tile edge, so polygonising them directly closes only a fraction of the fields. Instead the boundary raster is dilated by 1 px (closing gaps ≤ 0.5 m), the tile footprint edge is added, and every enclosed region ≥ 20 m² becomes one field instance. These instances are the training targets for the learned models and the reference for the instance metrics.

Variance lost when resampled to a coarser pixel size and back (nominal pixel 1.8 cm):

| Tile | 0.15 m | 0.30 m | 0.46 m | 0.61 m |
| --- | --- | --- | --- | --- |
| sw_010.TIF | 0.08% | 0.17% | 0.83% | 0.83% |
| sw_070.TIF | 0.07% | 0.17% | 0.85% | 0.73% |
| sw_090.TIF | 0.04% | 0.08% | 0.45% | 0.30% |

58 tiles prepared/loaded in 1 s.

| Split | Sites | Tiles | Fields | Median field (m²) | Boundary (km) |
| --- | --- | --- | --- | --- | --- |
| Train | 6 | 36 | 4169 | 374.3 | 240.5 |
| Valid | 2 | 12 | 1702 | 234.8 | 78.6 |
| Test | 2 | 10 | 815 | 527.7 | 55.0 |

<!-- fig:01 -->

*Figure 1. Example tiles from each split with ground truth. Yellow lines are the field instances built from the digitised boundaries.*

## Part 2 · Evaluation protocol

Every model outputs an instance label raster (one integer per field) on the 0.25 m tile grid. It is compared with the ground truth by:

| Metric | Definition |
| --- | --- |
| Boundary precision / recall / F1 @ τ | Predicted boundary pixels (edges of predicted fields) within τ of a digitised line ÷ all predicted boundary pixels, and digitised-line pixels within τ of a predicted boundary ÷ all digitised-line pixels. τ = 1 m (4 px) and 2 m (8 px). A 4-px band along the tile edge is excluded. |
| Instance precision / recall / F1 | A predicted field matches a reference field if IoU > 0.5 (one-to-one). Precision = matches ÷ predicted fields (≥ 20 m²); recall = matches ÷ reference fields. |
| Mean best IoU | For every reference field, the best IoU with any predicted field, averaged. This measures how well fields are captured, including partial matches. |

Scores are computed per tile and averaged over tiles (macro average).

Sanity check (reference vs itself): F1 1.0, precision 1.0, recall 1.0, mean best IoU 1.0, boundary F1 at 1 m 1.0, boundary F1 at 2 m 1.0.

## Part 3 · Model A: active-contour (snake) pipeline

The classical pipeline from the earlier notebooks (after Wagner & Oppelt 2020):

1. Pre-processing: gray-world white balance → percentile stretch → bilateral filter (×2) → sigmoid contrast on luminance, centred between the histogram peaks → CLAHE.
2. Augmented boundary-probability map: Sobel gradients (3 bands) + Sato ridge filter, averaged over 6 photometric/scale augmentations.
3. Segmentation: h-minima seeds → marker-controlled watershed → merge neighbouring regions (weakest border first) when the shared border is weak and their CIELAB colours are similar.
4. Active-contour refinement: each parcel outline is pulled onto boundary ridges by a parametric snake (α = 0.0005, β = 0.001).

The field / non-field mask of the earlier notebooks is not used: here the reference parcels include every enclosed area (also tree patches and homestead plots), so masking them would only remove correct parcels. All pixel parameters were defined at 0.55 m and are scaled by 0.55 / 0.25 = 2.2.

"Training" is parameter tuning. The pipeline has no learnable weights. Its three segmentation parameters, the h-minima depth H, the merge edge threshold T_edge and the colour threshold T_ΔE, are tuned by grid search on Train (2 tiles per site, 12 tiles), maximising instance F1. The best and the original default (H 0.04, T_edge 0.30, T_ΔE 8) are then compared on Valid.

Boundary maps for 58 tiles ready (3 s). Best on Train (instance F1): H 0.08, T_edge 0.2, T_ΔE 5.0.

<!-- fig:02 -->

*Instance F1 on the Train tuning tiles, for each h-minima depth and edge threshold, at three colour thresholds.*

Top settings on the Train tuning tiles (several nearby settings tie):

| H | T_edge | T_ΔE | F1 | Precision | Recall | Mean best IoU | Boundary F1 @ 1 m | Boundary F1 @ 2 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.08 | 0.20 | 5 | 0.389 | 0.382 | 0.414 | 0.439 | 0.554 | 0.726 |
| 0.08 | 0.10 | 5 | 0.389 | 0.382 | 0.414 | 0.439 | 0.554 | 0.726 |
| 0.08 | 0.10 | 10 | 0.389 | 0.382 | 0.414 | 0.439 | 0.554 | 0.726 |
| 0.08 | 0.10 | 20 | 0.389 | 0.382 | 0.414 | 0.439 | 0.554 | 0.726 |
| 0.08 | 0.20 | 10 | 0.389 | 0.382 | 0.414 | 0.439 | 0.554 | 0.726 |
| 0.08 | 0.20 | 20 | 0.389 | 0.382 | 0.413 | 0.438 | 0.554 | 0.726 |
| 0.08 | 0.30 | 5 | 0.376 | 0.383 | 0.386 | 0.420 | 0.550 | 0.721 |
| 0.08 | 0.30 | 10 | 0.356 | 0.374 | 0.356 | 0.398 | 0.540 | 0.708 |
| 0.12 | 0.30 | 5 | 0.356 | 0.453 | 0.307 | 0.363 | 0.532 | 0.694 |
| 0.12 | 0.10 | 10 | 0.355 | 0.450 | 0.307 | 0.364 | 0.532 | 0.695 |

Snake pipeline on Valid:

| Setting | F1 | Precision | Recall | Mean best IoU | Boundary F1 @ 1 m | Boundary F1 @ 2 m |
| --- | --- | --- | --- | --- | --- | --- |
| Default | 0.301 | 0.249 | 0.388 | 0.435 | 0.552 | 0.717 |
| Tuned | 0.403 | 0.399 | 0.412 | 0.440 | 0.572 | 0.737 |

Selected for Test: H 0.08, T_edge 0.2, T_ΔE 5.0.

## Part 4 · Model B: FTW Delineate Anything V2

Delineate Anything V2 (Lavreniuk et al.; shipped with FTW tools) is a YOLO11x-seg instance-segmentation network (62 M parameters) pre-trained on FBIS-22M (22.9 M field instances, 0.25–10 m imagery).

**Inference.** The 0.25 m tile is processed in overlapping square patches (patch/4 overlap discarded at each side). Each patch gives field polygons with a confidence score. Duplicates across patches are removed by score-ordered non-maximum suppression (IoU ≥ 0.3 or ≥ 60% containment), and overlaps go to the higher score. The patch size, resize factor and confidence threshold are chosen on Valid.

**Fine-tuning.**

- Dataset: 512 × 512 px crops (128 m) with stride 256 from the 0.25 m tiles. Crops with < 50% valid pixels are skipped. Every reference field is converted to a YOLO polygon (outer contour, simplified by 1 px).
- Training: all layers, AdamW, lr 0.001, 60 epochs (early stopping, patience 15), batch 8, image size 512, Ultralytics default augmentation (HSV jitter, flips, scale, mosaic for the first epochs). The best epoch is selected on the Valid crops' mask mAP.

Pre-trained Delineate Anything V2 — inference settings on Valid:

| Patch | Resize | Confidence | F1 | Precision | Recall | Mean best IoU | Boundary F1 @ 1 m | Boundary F1 @ 2 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 2 | 0.05 | 0.188 | 0.206 | 0.182 | 0.234 | 0.404 | 0.535 |
| 512 | 1 | 0.05 | 0.352 | 0.442 | 0.300 | 0.358 | 0.555 | 0.693 |
| 512 | 1 | 0.15 | 0.329 | 0.518 | 0.250 | 0.291 | 0.510 | 0.634 |
| 512 | 1 | 0.30 | 0.261 | 0.550 | 0.179 | 0.205 | 0.432 | 0.542 |
| 768 | 1 | 0.15 | 0.355 | 0.592 | 0.261 | 0.301 | 0.523 | 0.658 |

Pre-trained model, selected inference setting: patch 768, resize 1, confidence 0.15.

Epochs run: 34. Best epoch 19: mask mAP50 0.683, mask mAP50-95 0.398.

<!-- fig:03 -->

*Figure 3. Delineate Anything V2 fine-tuning curves.*

Fine-tuned Delineate Anything V2 — inference settings on Valid:

| Patch | Resize | Confidence | F1 | Precision | Recall | Mean best IoU | Boundary F1 @ 1 m | Boundary F1 @ 2 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 1 | 0.10 | 0.602 | 0.578 | 0.636 | 0.601 | 0.702 | 0.837 |
| 512 | 1 | 0.25 | 0.622 | 0.625 | 0.626 | 0.590 | 0.700 | 0.834 |
| 512 | 1 | 0.40 | 0.632 | 0.663 | 0.610 | 0.575 | 0.696 | 0.829 |
| 512 | 1 | 0.55 | 0.637 | 0.712 | 0.583 | 0.546 | 0.687 | 0.817 |
| 768 | 1 | 0.25 | 0.659 | 0.684 | 0.639 | 0.597 | 0.689 | 0.834 |

Fine-tuned model, selected inference setting: patch 768, resize 1, confidence 0.25.

## Part 5 · Model C: SAM 2.1 (Hiera-Large)

SAM 2 is a promptable segmentation foundation model: an image encoder (Hiera-L, 224 M parameters) produces image features once, and a light prompt encoder + mask decoder (about 4 M parameters) turns a prompt (points or a box) into masks. For field mapping SAM 2 is used in automatic mask generation (AMG) mode: a 64 × 64 grid of point prompts plus one crop layer, mask-to-mask refinement, `pred_iou_thresh` 0.65, `stability_score_thresh` 0.80. Masks are painted largest to smallest into an instance raster (masks > 25% of the tile and < 20 m² dropped).

**Fine-tuning (decoder only).** The image encoder stays frozen, so the large pre-trained representation is kept and training is cheap. The prompt encoder and mask decoder are trained on Train tiles:

- Each step: one tile, random flip / 90° rotation; 16 reference fields sampled; for each, 1–3 positive points drawn from the field interior (distance to edge ≥ 30% of the maximum).
- Loss: for each prompt, the best of the 3 multimask outputs under BCE + Dice, plus an MSE loss teaching the IoU head the true IoU (AMG uses that head to filter masks).
- AdamW, learning rate 2 × 10⁻⁵ with cosine decay, 2 000 steps, bfloat16 autocast.
- Validation every 250 steps on 6 Valid tiles: mean IoU of masks prompted by one point at the centre of every reference field. The best checkpoint is kept.

**Re-calibrating automatic mask generation (also on Valid).** AMG keeps a mask only if the decoder's IoU head predicts a high quality (`pred_iou_thresh`) and the mask is stable (`stability_score_thresh`). Fine-tuning changes how confident the IoU head is: the fine-tuned decoder rates almost every mask highly, so thresholds tuned for the pre-trained model let hundreds of fragments through. In a development run with the original thresholds, the fine-tuned SAM 2 dropped to F1 0.03 on Test. Both thresholds are therefore chosen on Valid for every model state (pre-trained and fine-tuned), as was done for Delineate Anything's confidence. To make the search affordable, masks are generated once per Valid tile at loose thresholds (0.5 / 0.7), then filtered post hoc for each candidate pair. The selected pair is used for an exact AMG run on Test.

SAM 2 decoder fine-tuning. The validation score is point-prompted mIoU on Valid tiles.

| Step | Loss | Valid mIoU |
| --- | --- | --- |
| 0 | — | 0.331 |
| 250 | 0.269 | 0.573 |
| 500 | 0.209 | 0.583 |
| 750 | 0.205 | 0.592 |
| 1000 | 0.202 | 0.579 |
| 1250 | 0.197 | 0.574 |
| 1500 | 0.191 | 0.582 |
| 1750 | 0.202 | 0.587 |
| 2000 | 0.192 | 0.587 |

The prompted mIoU measures the decoder alone. The pipeline used for mapping is automatic mask generation, so pre-trained and fine-tuned decoders are also compared in AMG mode on 4 Valid tiles.

SAM 2 AMG on Valid: instance F1 per threshold pair.

| pred_iou_thresh | stability_thresh | Fine-tuned F1 | Pre-trained F1 |
| --- | --- | --- | --- |
| 0.65 | 0.80 | 0.084 | 0.528 |
| 0.65 | 0.90 | 0.349 | 0.512 |
| 0.75 | 0.80 | 0.147 | 0.542 |
| 0.75 | 0.90 | 0.382 | 0.508 |
| 0.85 | 0.80 | 0.400 | 0.551 |
| 0.85 | 0.90 | 0.516 | 0.511 |
| 0.90 | 0.80 | 0.548 | 0.520 |
| 0.90 | 0.90 | 0.573 | 0.495 |
| 0.95 | 0.80 | 0.320 | 0.358 |
| 0.95 | 0.90 | 0.317 | 0.351 |

SAM 2 AMG on Valid at the selected thresholds:

| Model | F1 | Precision | Recall | Mean best IoU | Boundary F1 @ 1 m | Boundary F1 @ 2 m |
| --- | --- | --- | --- | --- | --- | --- |
| Pre-trained @ (0.85, 0.80) | 0.551 | 0.611 | 0.507 | 0.500 | 0.707 | 0.805 |
| Fine-tuned @ (0.90, 0.90) | 0.573 | 0.680 | 0.509 | 0.480 | 0.693 | 0.783 |

Selected `(pred_iou_thresh, stability_thresh)`: pre-trained (0.85, 0.80), fine-tuned (0.90, 0.90).

## Part 6 · Model D: SAM (v1, ViT-B)

The original Segment Anything model (Kirillov et al. 2023) has the same design: a ViT image encoder, a prompt encoder and a mask decoder. Here the ViT-B variant (91 M encoder parameters) is used. The ViT-H checkpoint (2.5 GB) could not be downloaded in reasonable time on this connection. Training and validation follow Part 5 exactly (same prompts, loss, optimiser, steps and validation), including the AMG threshold calibration on Valid (64 × 64 points, one crop layer, no mask-to-mask refinement, which SAM v1 lacks).

SAM (v1) decoder fine-tuning. The validation score is point-prompted mIoU on Valid tiles.

| Step | Loss | Valid mIoU |
| --- | --- | --- |
| 0 | — | 0.275 |
| 250 | 0.316 | 0.520 |
| 500 | 0.265 | 0.540 |
| 750 | 0.245 | 0.579 |
| 1000 | 0.230 | 0.562 |
| 1250 | 0.224 | 0.560 |
| 1500 | 0.222 | 0.567 |
| 1750 | 0.217 | 0.557 |
| 2000 | 0.223 | 0.559 |

<!-- fig:04 -->

*Figure 4. Decoder fine-tuning curves for SAM 2.1 and SAM v1. Solid blue is validation mIoU. Dashed red is training loss.*

SAM v1 AMG on Valid: instance F1 per threshold pair.

| pred_iou_thresh | stability_thresh | Fine-tuned F1 | Pre-trained F1 |
| --- | --- | --- | --- |
| 0.65 | 0.80 | 0.086 | 0.458 |
| 0.65 | 0.90 | 0.285 | 0.415 |
| 0.75 | 0.80 | 0.239 | 0.458 |
| 0.75 | 0.90 | 0.331 | 0.415 |
| 0.85 | 0.80 | 0.167 | 0.435 |
| 0.85 | 0.90 | 0.161 | 0.400 |
| 0.90 | 0.80 | 0.008 | 0.366 |
| 0.90 | 0.90 | 0.008 | 0.347 |
| 0.95 | 0.80 | 0.000 | 0.211 |
| 0.95 | 0.90 | 0.000 | 0.207 |

SAM v1 AMG on Valid at the selected thresholds:

| Model | F1 | Precision | Recall | Mean best IoU | Boundary F1 @ 1 m | Boundary F1 @ 2 m |
| --- | --- | --- | --- | --- | --- | --- |
| Pre-trained @ (0.65, 0.80) | 0.458 | 0.520 | 0.418 | 0.433 | 0.662 | 0.769 |
| Fine-tuned @ (0.75, 0.90) | 0.331 | 0.423 | 0.278 | 0.274 | 0.479 | 0.610 |

Selected `(pred_iou_thresh, stability_thresh)`: pre-trained (0.65, 0.80), fine-tuned (0.75, 0.90).

## Part 7 · Test-set evaluation

All eight configurations (four models × before / after training) are applied to every Test tile (sites sw_09 and sw_10, never seen during training or model selection), each with the inference setting selected on Valid: snake parameters, Delineate Anything patch size and confidence, and SAM AMG thresholds.

### Table 1 · Test performance (mean over 10 tiles)

| Model | Stage | F1 | Precision | Recall | Mean best IoU | Boundary F1 @ 1 m | Boundary F1 @ 2 m | Predicted fields | Reference fields |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Snake pipeline | default | 0.277 | 0.199 | 0.480 | 0.502 | 0.531 | 0.699 | 208 | 82 |
| Snake pipeline | tuned | 0.421 | 0.340 | 0.568 | 0.551 | 0.562 | 0.733 | 140 | 82 |
| FTW Delineate Anything V2 | pre-trained | 0.394 | 0.635 | 0.294 | 0.315 | 0.447 | 0.583 | 36 | 82 |
| FTW Delineate Anything V2 | fine-tuned | 0.641 | 0.553 | 0.771 | 0.691 | 0.651 | 0.809 | 114 | 82 |
| SAM 2.1 | pre-trained | 0.680 | 0.597 | 0.804 | 0.726 | 0.765 | 0.898 | 109 | 82 |
| SAM 2.1 | fine-tuned | 0.639 | 0.599 | 0.703 | 0.636 | 0.712 | 0.833 | 97 | 82 |
| SAM v1 | pre-trained | 0.622 | 0.537 | 0.742 | 0.689 | 0.740 | 0.876 | 112 | 82 |
| SAM v1 | fine-tuned | 0.388 | 0.299 | 0.562 | 0.550 | 0.597 | 0.765 | 155 | 82 |

### Table 2 · Change from training / tuning (after − before) on Test

| Model | F1 | Precision | Recall | Mean best IoU | Boundary F1 @ 1 m | Boundary F1 @ 2 m |
| --- | --- | --- | --- | --- | --- | --- |
| Snake pipeline | +0.144 | +0.141 | +0.088 | +0.049 | +0.031 | +0.034 |
| FTW Delineate Anything V2 | +0.247 | −0.082 | +0.477 | +0.376 | +0.204 | +0.225 |
| SAM 2.1 | −0.042 | +0.002 | −0.101 | −0.089 | −0.053 | −0.065 |
| SAM v1 | −0.234 | −0.238 | −0.180 | −0.139 | −0.143 | −0.111 |

<!-- fig:05 -->

*Figure 5. Test performance before and after training (mean ± SD over 10 tiles).*

Test site sw_09, before vs after training:

| Model | F1 before | F1 after | Boundary F1 @ 1 m before | Boundary F1 @ 1 m after | Mean best IoU before | Mean best IoU after |
| --- | --- | --- | --- | --- | --- | --- |
| Snake pipeline | 0.321 | 0.457 | 0.553 | 0.582 | 0.546 | 0.590 |
| FTW Delineate Anything V2 | 0.499 | 0.668 | 0.521 | 0.656 | 0.407 | 0.720 |
| SAM 2.1 | 0.693 | 0.672 | 0.787 | 0.740 | 0.776 | 0.699 |
| SAM v1 | 0.657 | 0.416 | 0.762 | 0.613 | 0.728 | 0.583 |

Test site sw_10, before vs after training:

| Model | F1 before | F1 after | Boundary F1 @ 1 m before | Boundary F1 @ 1 m after | Mean best IoU before | Mean best IoU after |
| --- | --- | --- | --- | --- | --- | --- |
| Snake pipeline | 0.248 | 0.396 | 0.516 | 0.548 | 0.472 | 0.525 |
| FTW Delineate Anything V2 | 0.324 | 0.623 | 0.398 | 0.648 | 0.254 | 0.672 |
| SAM 2.1 | 0.672 | 0.617 | 0.751 | 0.693 | 0.692 | 0.595 |
| SAM v1 | 0.599 | 0.369 | 0.725 | 0.586 | 0.664 | 0.528 |

### Table 3 · Paired test: after vs before training

Wilcoxon signed-rank test, n = 10 tiles.

| Model | Metric | Mean change | Tiles improved | n | p |
| --- | --- | --- | --- | --- | --- |
| Snake pipeline | F1 | +0.144 | 10 | 10 | 0.002 |
| Snake pipeline | Boundary F1 @ 1 m | +0.031 | 10 | 10 | 0.002 |
| FTW Delineate Anything V2 | F1 | +0.247 | 10 | 10 | 0.002 |
| FTW Delineate Anything V2 | Boundary F1 @ 1 m | +0.204 | 10 | 10 | 0.002 |
| SAM 2.1 | F1 | −0.042 | 3 | 10 | 0.049 |
| SAM 2.1 | Boundary F1 @ 1 m | −0.054 | 0 | 10 | 0.002 |
| SAM v1 | F1 | −0.234 | 0 | 10 | 0.002 |
| SAM v1 | Boundary F1 @ 1 m | −0.143 | 0 | 10 | 0.002 |

<!-- fig:06 -->

*Test tile sw_090: the 0.25 m image, the digitised reference, and each model before and after training.*

<!-- fig:07 -->

*Test tile sw_105: the 0.25 m image, the digitised reference, and each model before and after training.*

### 7.1 Predicted fields as map layers

The two models recommended for operational mapping (Part 8) are written out as field polygons so they can be opened in Google Earth, Google My Maps, geojson.io, QGIS, or any other web map. These files are the predictions before the Steger line extraction and the snake in 7.2. The final KML, after that refinement, is written in 7.3.

| Model | Why this file | Files |
| --- | --- | --- |
| SAM 2.1, pre-trained | Highest Test accuracy | `train_outputs/vectors/sam2_pretrained_fields.geojson` and `.kml` |
| FTW Delineate Anything V2, fine-tuned | Close second, and fast enough for large areas | `train_outputs/vectors/da_finetuned_fields.geojson` and `.kml` |

Each connected field in the instance raster (≥ 20 m²) becomes one polygon. Coordinates are transformed from BUTM-2010 (EPSG:9680) to WGS84 longitude/latitude (EPSG:4326), which is what Google Maps and KML expect. Boundaries are simplified by 0.5 m so the files stay small enough to import; the rasters in `train_outputs/pred/` remain the exact predictions. In Google Earth or Google My Maps, use Import and choose the `.kml`. On geojson.io or similar sites, drop in the `.geojson`.

| Model | Stage | Fields | Tiles | Area (ha) | Longitude min | Latitude min | Longitude max | Latitude max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM 2.1 | pre-trained | 1002 | 10 | 52.19 | 88.81665 | 23.56664 | 88.96672 | 23.61115 |
| FTW Delineate Anything V2 | fine-tuned | 1213 | 10 | 47.13 | 88.81667 | 23.56667 | 88.96666 | 23.61112 |

### 7.2 Missing boundaries (Steger 1998) and snake refinement

The predicted outlines are already close to the digitised lines (about 1.3 m on average for SAM 2.1 and 2.5 m for Delineate Anything), but two geometric faults remain: some dark bunds inside a predicted field were never cut, and the outline that was cut does not always sit on the centre of the bund.

**Line extraction (Steger 1998).** On the pre-processed luminance, Gaussian derivatives at scales σ = 1.0, 1.6 and 2.2 px (bunds about 0.6–1.4 m wide) give the Hessian. A pixel is a line centre where the first derivative across the line vanishes inside the pixel and the second derivative is positive — a dark line. Dark lines are the bunds and ditches: at σ = 1.2 their response on the digitised boundaries is about seven times the response in the field interior, whereas bright ridges are not. Centres are linked along the line into polylines (hysteresis, normals within about 50°).

A polyline is a candidate missing boundary when it runs through the interior of one predicted field (more than 1.25 m from the existing outline along most of its length), is at least 7 m long, is reasonably straight, and the CIELAB colour 2 m to either side differs by at least 7. Each candidate can be extended until both ends meet that field's outline (a topological cut) and fitted by an open snake with fixed ends. Doing so on sw_090, sw_091 and sw_100 lowered both instance F1 and boundary F1: most of these dark lines are crop rows, tree lines or ditches inside a digitised parcel, not a boundary the digitiser omitted. They are therefore saved as a line layer, `train_outputs/vectors/missing_boundaries.geojson` (194 lines on the SAM 2.1 predictions), and are not used to split the parcels. The web map can turn them on in yellow.

**Snakes on the existing outlines.** Every predicted outline is a closed snake (Kass, Witkin and Terzopoulos 1988). The external force acts only along the contour normal and only up to 0.75 m, and only towards a Steger centre: the outline can lock onto the bund it already follows, and it cannot jump to another line. Two smoothing passes are the internal energy (elasticity, and a little rigidity). A field is kept in its original shape unless the new region still overlaps it with IoU ≥ 0.92 and the new outline lies on a stronger Steger response than the old one. Those two tests are the topological constraint: the partition into fields does not change, only the geometry of the outline.

<!-- fig:08 -->

*Figure 7. Steger line extraction and snake refinement on Test. Green is the refined outline. Yellow marks Steger lines that were not used to split fields.*

### Table 4 · Test, before and after the snake

Instance F1 is unchanged. Boundary F1 moves slightly.

| Model | F1 before | F1 after | Boundary F1 @ 1 m before | Boundary F1 @ 1 m after | Boundary F1 @ 2 m before | Boundary F1 @ 2 m after |
| --- | --- | --- | --- | --- | --- | --- |
| FTW Delineate Anything V2 | 0.641 | 0.641 | 0.651 | 0.668 | 0.809 | 0.817 |
| SAM 2.1 | 0.680 | 0.682 | 0.765 | 0.772 | 0.898 | 0.901 |

Outlines updated on 84 fields per tile (mean). Candidate missing lines (SAM 2.1, not used to split): 194.

### 7.3 Final field polygons

The KML and GeoJSON written here are the final map layers: the Test fields after the snake in 7.2 has moved each outline onto the Steger line centre. They replace the pre-refinement files of 7.1 for Google Earth, Google My Maps and the local web map.

| Model | Files |
| --- | --- |
| SAM 2.1, pre-trained | `train_outputs/vectors/sam2_pretrained_final.geojson` and `.kml` |
| FTW Delineate Anything V2, fine-tuned | `train_outputs/vectors/da_finetuned_final.geojson` and `.kml` |

| Model | Stage | Fields | Tiles | Area (ha) |
| --- | --- | --- | --- | --- |
| SAM 2.1 | snake-refined | 1002 | 10 | 52.19 |
| FTW Delineate Anything V2 | snake-refined | 1192 | 10 | 47.47 |

The final KML is the snake-refined outline. Open the `.kml` in Google Earth or Google My Maps.

### 7.4 Final polygons and digitised lines

The final SAM 2.1 polygons (after the snake) are drawn on the image, beside the digitised Test lines on the same image. The three rows are the best, medium and worst Test tiles, ranked by instance F1. The medium tile is the one nearest the median F1.

<!-- fig:09 -->

*Figure 8. SAM 2.1 final polygons (blue) and digitised Test lines (red) on the best, medium and worst Test tiles.*

## Part 8 · Interpretation and conclusions

Numbers refer to the reference run of this notebook on `Labeled_Data/` (EPSG:4326 tiles warped to BUTM-2010): Test = 10 tiles at sites sw_09 and sw_10, 815 reference fields, mean over tiles. Paired changes were tested with the Wilcoxon signed-rank test (n = 10 tiles).

### 8.1 What training changed

| Model | Instance F1, before → after | Boundary F1 @ 1 m, before → after | Verdict |
| --- | --- | --- | --- |
| Snake pipeline (tuning) | 0.28 → 0.42 | 0.53 → 0.56 | Tuning on Train helps on every Test tile (10/10, p = 0.002). The default over-segments (about 210 polygons per tile vs 82 reference fields). The tuned setting (deeper h-minima, H = 0.08) suppresses spurious seeds, so precision rises from 0.20 to 0.34. |
| FTW Delineate Anything V2 (fine-tuning) | 0.39 → 0.64 | 0.45 → 0.65 | The largest gain from training (10/10 tiles, p = 0.002). Recall more than doubles (0.29 → 0.77): the pre-trained model misses many plots in this landscape; after fine-tuning it finds most of them. Validation picked the 768 px patch and confidence 0.25. |
| SAM 2.1 (decoder fine-tuning) | 0.68 → 0.64 | 0.77 → 0.71 | Decoder fine-tuning does not help automatic mapping (worse on 7/10 tiles, p = 0.049), even though point-prompted Valid mIoU rose from 0.33 to 0.59. On Valid the fine-tuned decoder calibrated slightly higher (AMG F1 0.57 vs 0.55), so a Valid-based selection would have deployed the weaker model here. |
| SAM v1 (decoder fine-tuning) | 0.62 → 0.39 | 0.74 → 0.60 | Same pattern, more strongly (worse on 10/10 tiles). Valid already preferred the pre-trained decoder (AMG F1 0.46 vs 0.33), so a Valid-based selection would not deploy the fine-tuned SAM v1. |

### 8.2 Why fine-tuning the SAM decoders helped prompts but hurt automatic mapping

**Training and use differ.** The decoder was trained to answer "segment the field under this point", and it got better at exactly that (prompted mIoU +0.26 for SAM 2, +0.30 for SAM v1). Automatic mask generation instead prompts a dense 64 × 64 grid, including points on bunds, paths and trees, and then ranks and filters the answers with the decoder's own IoU estimate and a stability score.

**Calibration shifts.** After fine-tuning, the IoU head rates almost every answer as high quality, so the pre-trained thresholds let hundreds of fragments through. Before threshold re-calibration on Valid, the fine-tuned SAM 2 fell to F1 0.03 on Test. Re-calibration recovers most of this (0.64), but not the pre-trained level. A development experiment that sampled training points anywhere inside fields (to mimic the grid) did not help either.

**Small, homogeneous training set.** 36 tiles from 6 nearby sites are enough to adapt a 4 M-parameter decoder to these prompts, but not to change the ranking behaviour that automatic generation relies on.

**Calibration also helps the pre-trained models.** Choosing thresholds on Valid raised pre-trained SAM 2 from F1 about 0.56 (default thresholds, first run) to 0.68 on Test. For promptable models, calibrating the inference settings was worth more than training the decoder.

### 8.3 Best models for field-boundary mapping on this imagery

1. **SAM 2.1, pre-trained, with Valid-calibrated AMG thresholds:** best on almost every Test metric: instance F1 0.68, recall 0.80, mean best IoU 0.73, boundary F1 0.77 at 1 m and 0.90 at 2 m. It needs no training, but it is the slowest (about 1 min per tile on an RTX A5500).
2. **FTW Delineate Anything V2, fine-tuned:** second (instance F1 0.64, precision 0.55, recall 0.77, boundary F1 0.65 at 1 m), close to SAM 2.1 at Test site sw_09 (F1 0.67 vs 0.69). It runs in about 1–2 s per tile (about 1/50 of SAM 2's time) and gains the most from local training data. It is the most practical choice for large areas.
3. **SAM v1 (ViT-B), pre-trained:** F1 0.62. A good fallback, but weaker than SAM 2.
4. **Snake pipeline, tuned:** F1 0.42. Needs no GPU or training labels, but its boundaries are the least precise and it under-segments small plots. Useful where labels and GPUs are unavailable.

Pre-trained SAM 2 is best at both Test sites: sw_09 (F1 0.69 vs 0.67 for fine-tuned Delineate Anything) and sw_10 (0.67 vs 0.62). Both models form the top group at both sites. The before/after changes are significant for every model (p ≤ 0.049). The snake tuning, Delineate Anything and SAM v1 changes go the same way on all 10 tiles; the SAM 2.1 decoder fine-tuning is less consistent (worse on 7 of 10).

### 8.4 Recommendations

- For operational mapping: fine-tune Delineate Anything V2 on a few dozen labelled tiles from the target region, or use pre-trained SAM 2 with calibrated thresholds where a GPU is available and boundary precision matters most.
- A promising combination: use the fine-tuned Delineate Anything detections as box prompts for SAM 2 (the detector finds the fields, SAM traces the edges). This avoids SAM's automatic grid and its calibration problem.
- If SAM is to be trained, train it for the deployment task: full fine-tuning with an automatic-generation objective (grid prompts including negatives) or box prompts, with checkpoint selection on automatic-mapping F1 on Valid, not on prompted IoU.

### 8.5 Limitations

- **One region, one season.** 10 sites from the same area. Transfer to other districts, seasons or sensors is untested.
- **Reference construction.** Field instances were derived from digitised lines (gaps ≤ 0.5 m closed, regions ≥ 20 m²). Enclosed non-field patches (trees, homesteads) count as parcels, and digitising errors affect all models equally.
- **Imagery source.** All tiles are screenshots of Google satellite imagery (effective resolution about 0.2–0.3 m; acquisition dates unknown and possibly mixed). Results may differ for other providers, dates or true very-high-resolution imagery. Google imagery is subject to Google's Terms of Service, which restrict bulk download and reuse.
- **SAM v1 used the ViT-B encoder.** The ViT-H checkpoint could not be downloaded in reasonable time. ViT-H would likely narrow the gap to SAM 2.
- **Limited hyper-parameter search.** Training schedules (2 000 decoder steps, ≤ 60 YOLO epochs) and the threshold grids were not exhaustively searched.
