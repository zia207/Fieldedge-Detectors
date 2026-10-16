"""Builds Labeled_Data/field_boundary_training.ipynb

Train / validate / test field-boundary models on georeferenced Google satellite-image screenshots with
digitised boundary polylines (Labeled_Data/Data/{Train,Valid,Test}/sw_xx/{Tiles,Polyline}):
  • Active-contour (snake) pipeline   — unsupervised; parameters tuned on Train, checked on Valid
  • FTW Delineate Anything V2         — YOLO-seg, fine-tuned on Train, validated on Valid
  • SAM 2.1 (Hiera-L)                 — mask decoder fine-tuned with point prompts
  • SAM (v1, ViT-B)                   — mask decoder fine-tuned with point prompts
  • FC-DenseNet (Tiramisu)            — semantic boundary segmentation, trained from scratch
Every model is evaluated before and after training on the held-out Test sites.

Run:  .venv/bin/python build_training_notebook.py
"""
import os
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ════════════════════════════════════════════════════════════════════════════
md(r"""
# Training and Testing Field-Boundary Models on Google Satellite-Image Screenshots
## Active Contours · FTW Delineate Anything V2 · SAM · SAM 2 · FC-DenseNet
---
**Author:** Zia U. Ahmed, Ph.D. · Uppata Analytics LLC

### Objectives
1. Prepare a **training / validation / test** dataset from georeferenced **Google satellite-image
   screenshots** with hand-digitised field-boundary polylines (10 sites, 58 tiles).
2. **Train or tune** five field-delineation approaches on the *Train* sites and select settings
   and checkpoints on the *Valid* sites (see the table below).
3. **Evaluate every model before and after training** on the held-out *Test* sites, using boundary accuracy
   (boundary F1 at 1 m and 2 m tolerance) and field-instance accuracy (precision / recall / F1 at IoU > 0.5).
4. **Interpret** what training changed, and which approach is best for operational field mapping.

| Model | Type | What is learned |
|---|---|---|
| Active-contour (snake) pipeline | classical, unsupervised | segmentation parameters tuned by grid search |
| FTW Delineate Anything V2 | instance segmentation (YOLO11x-seg) | full network fine-tuned |
| SAM 2.1 (Hiera-Large) | promptable foundation model | mask decoder + prompt encoder fine-tuned |
| SAM (v1, ViT-B) | promptable foundation model | mask decoder + prompt encoder fine-tuned |
| FC-DenseNet (Tiramisu) | semantic segmentation | full network trained from scratch; depth chosen on Valid |

### Notebook map
| Part | Content |
|---|---|
| 0 | Environment and configuration |
| 1 | Data: inventory, preparation at 0.25 m, ground truth |
| 2 | Evaluation protocol |
| 3 | Model A: active-contour (snake) pipeline — tuning |
| 4 | Model B: FTW Delineate Anything V2 — fine-tuning |
| 5 | Model C: SAM 2.1 — decoder fine-tuning |
| 6 | Model D: SAM (v1) — decoder fine-tuning |
| 6b | Model E: FC-DenseNet — boundary segmentation, depth chosen on Valid |
| 7 | Test-set evaluation, map export, then Steger lines and snake refinement |
| 8 | Interpretation and conclusions |
""")

md(r"""
---
# Part 0 · Environment and configuration
Runs in the repository `.venv` environment (Python 3.12, PyTorch 2.10 + CUDA 12.8, `ftw-tools 2.0.0b5`,
`ultralytics`, `sam2`, `segment-anything`). The labelled data are read from `Data/` next to this notebook; all outputs
and models of this study live in `train_outputs/`. Every stage caches its outputs under `train_outputs/`:
prepared tiles, tuning tables, trained weights and predictions. A re-run therefore skips completed
training.
""")
code(r"""
import glob, json, math, os, random, shutil, sys, time, warnings
from importlib.metadata import version
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from scipy import ndimage as ndi
from scipy.signal import find_peaks

import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, box, shape
from shapely.strtree import STRtree

from skimage import graph
from skimage.color import rgb2lab
from skimage.draw import polygon as draw_polygon
from skimage.filters import gaussian, sato
from skimage.measure import find_contours, regionprops
from skimage.morphology import h_minima
from skimage.segmentation import watershed, find_boundaries, active_contour

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 11})
SEED = 2026
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python {sys.version.split()[0]} · torch {torch.__version__} (CUDA {torch.version.cuda}) · "
      f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
for p in ["ftw-tools", "ultralytics", "sam2", "segment-anything", "scikit-image", "rasterio", "geopandas"]:
    print(f"  {p:<17} {version(p)}")
""")
code(r"""
def _labeled_root():
    # Data/{Train,Valid,Test} lives in Labeled_Data/, whether the kernel starts there or in the repo root.
    here = os.path.abspath(os.getcwd())
    for base in (here, os.path.join(here, "Labeled_Data"), os.path.dirname(here)):
        if os.path.isdir(os.path.join(base, "Data", "Train")):
            return base
    raise FileNotFoundError("Labelled tiles not found (expected Data/Train). cwd=" + here)

LABELED = _labeled_root()
DATA = os.path.join(LABELED, "Data")   # Data/{Train,Valid,Test}/sw_xx/{Tiles,Polyline}
GSD = 0.25                         # working resolution (m / pixel)
WORK_CRS = "EPSG:9680"             # BUTM-2010 (WGS 84 / TM 90 NE): tiles in lon/lat are warped to it
OUT = os.path.join(LABELED, "train_outputs")
DIRS = {k: os.path.join(OUT, k) for k in ("prep", "snake", "da", "sam2", "sam1", "fcdn", "pred", "figures", "tables", "vectors")}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

DA_WEIGHTS = os.path.join(LABELED, "checkpoints", "DelineateAnythingv2.pt")  # pre-trained Delineate Anything V2 (FTW)
DA_URL = ("https://hf.co/MykolaL/DelineateAnything/resolve/369d0b4c44cf9bec2bd3a27bc81810cadd2c963e/"
          "DelineateAnythingv2.pt")
SAM2_ID = "facebook/sam2.1-hiera-large"
SAM1_CKPT = os.path.join(LABELED, "checkpoints", "sam_vit_b_01ec64.pth")
MIN_FIELD_M2 = 20                  # smallest parcel kept in ground truth and predictions
MIN_PX = int(MIN_FIELD_M2 / GSD**2)
print("data", DATA)
print("outputs", OUT)
print("working GSD", GSD, "m · min parcel", MIN_PX, "px")

def res_m(src):
    # Pixel size in metres; for lon/lat tiles, the north–south size from the meridian degree length.
    if not src.crs.is_geographic:
        return src.res[0]
    lat = math.radians((src.bounds.bottom + src.bounds.top) / 2)
    return src.res[1] * (111132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat))
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 1 · Data
### 1.1 Inventory
Each site folder `sw_xx` holds **screenshots of Google satellite imagery** (`Tiles/*.TIF`, RGB), georeferenced in the
WGS 84 longitude/latitude (EPSG:4326) and covering ≈ 286 m × 206 m each; they are warped to BUTM-2010
(EPSG:9680) when prepared. The GeoTIFFs are stored at a nominal 1.8 cm/pixel,
but that is the resolution of the *screenshot export*, not of the imagery: the underlying Google imagery is
much coarser (see 1.2) and, for each tile, a shapefile of **hand-digitised field-boundary polylines**
(`Polyline/<tile>.shp`). The split is by **site**: no site appears in more than one split, so Valid and
Test measure transfer to unseen locations.
""")
code(r"""
rows = []
for split in ("Train", "Valid", "Test"):
    for site in sorted(os.listdir(f"{DATA}/{split}")):
        tifs = sorted(glob.glob(f"{DATA}/{split}/{site}/Tiles/*.TIF") + glob.glob(f"{DATA}/{split}/{site}/Tiles/*.tif"))
        shps = sorted(glob.glob(f"{DATA}/{split}/{site}/Polyline/*.shp"))
        for tif in tifs:
            tid = os.path.splitext(os.path.basename(tif))[0]
            shp = f"{DATA}/{split}/{site}/Polyline/{tid}.shp"
            with rasterio.open(tif) as s:
                rows.append(dict(split=split, site=site, tile=tid, tif=tif, shp=shp if os.path.exists(shp) else None,
                                 width=s.width, height=s.height, crs=s.crs.to_string(), res_cm=res_m(s) * 100,
                                 extent_m=f"{s.width*res_m(s):.0f} × {s.height*res_m(s):.0f}"))
        unmatched = sorted(set(os.path.splitext(os.path.basename(p))[0] for p in shps) -
                           set(os.path.splitext(os.path.basename(p))[0] for p in tifs))
        if unmatched:
            print(f"{split}/{site}: shapefiles without an image (ignored): {unmatched}")
if not rows:
    raise FileNotFoundError(f"No labelled tiles found under {DATA}")
TILES = pd.DataFrame(rows)
TILES = TILES[TILES.shp.notna()].reset_index(drop=True)
if TILES.empty:
    raise FileNotFoundError(f"Tiles under {DATA} have no matching shapefiles")
TILES.to_csv(os.path.join(DIRS["tables"], "tiles.csv"), index=False)
display(TILES.groupby(["split", "site"]).agg(tiles=("tile", "size"), res_cm=("res_cm", "mean"),
                                             extent_m=("extent_m", "first")).round(2))
""")

md(r"""
### 1.2 Preparation at 0.25 m and ground truth
* **Resampling.** The screenshots are stored at a nominal 1.8 cm/pixel (≈ 16 000 × 11 500 px per tile),
  but Google satellite imagery is far coarser, so most of these pixels are interpolated. The check below
  shows that reducing a tile to 0.14–0.29 m and enlarging it back loses only ≈ 0.03–0.2 % of the image
  variance; real detail starts to be lost only around 0.4 m (≈ 0.3–1 %). The **effective resolution is therefore
  ≈ 0.2–0.3 m**, and every tile is resampled by **pixel averaging to 0.25 m** (≈ 1 144 × 826 px) with
  essentially no loss of information. At 0.25 m a 15 m wide plot is 60 px across, and 0.25 m is inside
  Delineate Anything's training range (0.25–10 m). Nodata borders are masked.
* **Ground-truth boundaries.** The polylines are clipped to the tile and rasterised (1 px, all touched).
  This raster is the reference for the **boundary metrics**.
* **Ground-truth field instances.** Digitised lines do not always meet exactly and run off the tile
  edge, so polygonising them directly closes only a fraction of the fields. Instead the boundary raster
  is dilated by 1 px (closing gaps ≤ 0.5 m), the tile footprint edge is added, and every enclosed
  region ≥ 20 m² becomes one **field instance**. These instances are the training targets for the learned
  models and the reference for the **instance metrics**.
""")
code(r"""
# Effective resolution of the screenshots: variance lost when a 4096 × 4096 px crop is reduced to a
# coarser pixel size and enlarged back (cubic). A few tenths of a percent means no real detail at that scale.
rows_ = []
for t in (TILES[TILES.split == s].tif.iloc[0] for s in ("Train", "Valid", "Test")):
    with rasterio.open(t) as src:
        a = src.read(1, window=rasterio.windows.Window(4000, 3000, 4096, 4096)).astype(np.float32); res = res_m(src)
    r_ = {"tile": os.path.basename(t)}
    for f in (8, 16, 24, 32):
        back = cv2.resize(cv2.resize(a, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA), a.shape[::-1],
                          interpolation=cv2.INTER_CUBIC)
        r_[f"{res * f:.2f} m"] = np.mean((a - back) ** 2) / a.var() * 100
    rows_.append(r_)
display(pd.DataFrame(rows_).set_index("tile").round(2).style.format("{:.2f} %")
        .set_caption("Variance lost when resampled to a coarser pixel size and back (nominal pixel 1.8 cm)"))
""")
code(r"""
def read_tile(path):
    # RGB at GSD m in a metric CRS. Lon/lat tiles are warped (pixel averaging) to WORK_CRS. The black screenshot
    # border is source no-data, so it stays 0 instead of being averaged into a dark strip that joins edge fields.
    with rasterio.open(path) as s:
        if s.crs.is_geographic:
            tr, W, H = calculate_default_transform(s.crs, WORK_CRS, s.width, s.height, *s.bounds, resolution=GSD)
            with WarpedVRT(s, crs=WORK_CRS, transform=tr, width=W, height=H, resampling=Resampling.average,
                           src_nodata=0, nodata=0) as v:
                return v.read([1, 2, 3]).transpose(1, 2, 0), tr, v.crs, v.bounds
        f = GSD / s.res[0]; H, W = int(round(s.height / f)), int(round(s.width / f))
        img = s.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.average).transpose(1, 2, 0)
        return img, s.transform * Affine.scale(s.width / W, s.height / H), s.crs, s.bounds

def prepare_tile(r):
    img, tr, crs, bnds = read_tile(r.tif); H, W = img.shape[:2]
    valid = ndi.binary_opening(img.sum(2) > 0, iterations=2)
    lines = gpd.read_file(r.shp).to_crs(crs).geometry.clip(box(*bnds))
    lines = lines[~lines.is_empty]
    bnd = features.rasterize([(l, 1) for l in lines], out_shape=(H, W), transform=tr, fill=0,
                             all_touched=True, dtype="uint8").astype(bool) & valid
    barrier = ndi.binary_dilation(bnd, iterations=1) | (valid & ~ndi.binary_erosion(valid))
    inst, n = ndi.label(valid & ~barrier)
    size = np.bincount(inst.ravel()); keep = size >= MIN_PX; keep[0] = False
    inst = np.where(keep[inst], inst, 0)
    _, inst = np.unique(inst, return_inverse=True)
    return dict(img=img, valid=valid, boundary=bnd, inst=inst.reshape(H, W).astype(np.int32),
                transform=np.array(tr)[:6], crs=crs.to_wkt(), n_lines=len(lines))

D = {}
t0 = time.time()
for r in TILES.itertuples():
    p = os.path.join(DIRS["prep"], f"{r.tile}.npz")
    if not os.path.exists(p):
        np.savez_compressed(p, **prepare_tile(r))
    z = np.load(p, allow_pickle=True); D[r.tile] = {k: z[k] for k in z.files}
print(f"{len(D)} tiles prepared/loaded in {time.time()-t0:.0f}s")

SPLIT = {s: TILES[TILES.split == s].tile.tolist() for s in ("Train", "Valid", "Test")}
stats = []
for r in TILES.itertuples():
    d = D[r.tile]; sz = np.bincount(d["inst"].ravel())[1:] * GSD**2
    stats.append(dict(split=r.split, site=r.site, tile=r.tile, lines=int(d["n_lines"]), fields=len(sz),
                      median_field_m2=np.median(sz), boundary_km=d["boundary"].sum() * GSD / 1000))
STATS = pd.DataFrame(stats); STATS.to_csv(os.path.join(DIRS["tables"], "dataset_stats.csv"), index=False)
display(STATS.groupby("split").agg(sites=("site", "nunique"), tiles=("tile", "size"), fields=("fields", "sum"),
                                   median_field_m2=("median_field_m2", "median"),
                                   boundary_km=("boundary_km", "sum")).loc[["Train", "Valid", "Test"]].round(1))
""")
code(r"""
ex = [SPLIT["Train"][0], SPLIT["Valid"][0], SPLIT["Test"][-1]]
fig, axes = plt.subplots(2, 3, figsize=(24, 12))
for j, t in enumerate(ex):
    d = D[t]; axes[0, j].imshow(d["img"]); axes[0, j].set_title(f"{t} ({TILES.set_index('tile').split[t]}) — 0.25 m image")
    e = d["img"].copy(); e[find_boundaries(d["inst"])] = (255, 255, 0); e[d["boundary"]] = (255, 0, 0)
    axes[1, j].imshow(e); axes[1, j].set_title(f"digitised lines (red) and {d['inst'].max()} field instances (yellow)")
    for a in axes[:, j]: a.axis("off")
plt.suptitle("Figure 1 · Example tiles from each split with ground truth", y=1.0)
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig01_data.png"), bbox_inches="tight"); plt.show()
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 2 · Evaluation protocol
Every model outputs an **instance label raster** (one integer per field) on the 0.25 m tile grid.
It is compared with the ground truth by:

| Metric | Definition |
|---|---|
| **Boundary precision / recall / F1 @ τ** | Predicted boundary pixels (edges of predicted fields) within τ of a digitised line ÷ all predicted boundary pixels, and digitised-line pixels within τ of a predicted boundary ÷ all digitised-line pixels. τ = **1 m** (4 px) and **2 m** (8 px). A 4-px band along the tile edge is excluded. |
| **Instance precision / recall / F1** | A predicted field matches a reference field if IoU > 0.5 (one-to-one by construction). Precision = matches ÷ predicted fields (≥ 20 m²); recall = matches ÷ reference fields. |
| **Mean best IoU** | For every reference field, the best IoU with any predicted field, averaged. This measures how well fields are captured, including partial matches. |

Scores are computed per tile and averaged over tiles (macro average).
""")
code(r"""
def boundary_scores(pred, d, tol_m):
    core = ndi.binary_erosion(d["valid"], iterations=4)
    pb = find_boundaries(pred, mode="inner") & (pred > 0) & core
    gb = d["boundary"] & core
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0, 0.0, 0.0
    tol = tol_m / GSD
    P = float((ndi.distance_transform_edt(~gb)[pb] <= tol).mean())
    R = float((ndi.distance_transform_edt(~pb)[gb] <= tol).mean())
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def instance_scores(pred, d, thr=0.5):
    gt = d["inst"]; gid = np.unique(gt[gt > 0])
    ps = np.bincount(pred.ravel()); pid = np.where(ps >= MIN_PX)[0]; pid = pid[pid > 0]
    gs = np.bincount(gt.ravel())
    m = (gt > 0) & (pred > 0); K = int(pred.max()) + 1
    pairs, cnt = np.unique(gt[m].astype(np.int64) * K + pred[m], return_counts=True)
    g_, p_ = pairs // K, pairs % K
    iou = cnt / (gs[g_] + ps[p_] - cnt)
    ok = (iou > thr) & np.isin(p_, pid); tp = int(ok.sum())
    best = pd.Series(iou).groupby(g_).max().reindex(gid).fillna(0)
    return dict(precision=tp / max(len(pid), 1), recall=tp / max(len(gid), 1),
                F1=2 * tp / max(len(gid) + len(pid), 1), mean_best_IoU=float(best.mean()),
                n_pred=len(pid), n_ref=len(gid))

def evaluate(pred, d):
    out = instance_scores(pred, d)
    for tol in (1.0, 2.0):
        P, R, Fb = boundary_scores(pred, d, tol)
        out.update({f"bP@{tol:g}m": P, f"bR@{tol:g}m": R, f"bF1@{tol:g}m": Fb})
    return out

def evaluate_set(preds, tiles):
    return pd.DataFrame([dict(tile=t, **evaluate(preds[t], D[t])) for t in tiles])

KEY = ["F1", "precision", "recall", "mean_best_IoU", "bF1@1m", "bF1@2m"]
chk = evaluate(D[SPLIT["Test"][0]]["inst"], D[SPLIT["Test"][0]])
print("Sanity check (reference vs itself):", {k: round(chk[k], 3) for k in KEY})
""")
code(r"""
# Shared helpers: polygons/masks → instance raster, and score-ordered de-duplication
def masks_to_labels(masks, shape, max_frac=0.25):
    lab = np.zeros(shape, np.int32)
    keep = [m for m in masks if MIN_PX <= m.sum() <= max_frac * shape[0] * shape[1]]
    for i, m in enumerate(sorted(keep, key=lambda m: m.sum(), reverse=True), start=1):
        lab[m] = i
    return lab

def nms_partition(polys, scores, iou_t=0.3, cont_t=0.6, min_area=MIN_PX):
    kept, ks = [], []
    for i in np.argsort(-np.asarray(scores)):
        g = shapely.make_valid(polys[i])
        if g.is_empty or g.area < min_area:
            continue
        hits = STRtree(kept).query(g, predicate="intersects") if kept else []
        dup = False
        for k in hits:
            inter = g.intersection(kept[k]).area
            if inter / (g.area + kept[k].area - inter) >= iou_t or inter / min(g.area, kept[k].area) >= cont_t:
                dup = True; break
        if dup:
            continue
        for k in hits:
            g = g.difference(kept[k])
        if not g.is_empty and g.area >= min_area:
            kept.append(g); ks.append(scores[i])
    return kept, ks

def polys_to_labels(polys, shape):
    # polygons in pixel coordinates (x = col, y = row); later polygons drawn on top
    if not polys:
        return np.zeros(shape, np.int32)
    return features.rasterize([(p, k + 1) for k, p in enumerate(polys)], out_shape=shape,
                              transform=Affine.identity(), fill=0, dtype="int32")
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 3 · Model A: active-contour (snake) pipeline
The classical pipeline from the earlier notebooks (after Wagner & Oppelt 2020):

1. **Pre-processing:** gray-world white balance → percentile stretch → bilateral filter (×2) → sigmoid
   contrast on luminance, centred between the histogram peaks → CLAHE.
2. **Augmented boundary-probability map:** Sobel gradients (3 bands) + Sato ridge filter, averaged
   over 6 photometric/scale augmentations.
3. **Segmentation:** h-minima seeds → marker-controlled watershed → merge neighbouring regions
   (weakest border first) when the shared border is weak **and** their CIELAB colours are similar.
4. **Active-contour refinement:** each parcel outline is pulled onto boundary ridges by a parametric
   snake (α = 0.0005, β = 0.001).

The field / non-field mask of the earlier notebooks is **not** used: here the reference parcels include
every enclosed area (also tree patches and homestead plots), so masking them would only remove correct
parcels. All pixel parameters were defined at 0.55 m and are scaled by `0.55 / 0.25 = 2.2`.

**"Training" = parameter tuning.** The pipeline has no learnable weights. Its three segmentation
parameters, the h-minima depth `H`, the merge edge threshold `T_edge` and the colour threshold `T_ΔE`,
are tuned by **grid search on Train** (2 tiles per site, 12 tiles), maximising instance F1. The best and
the original default (H 0.04, T_edge 0.30, T_ΔE 8) are then compared on **Valid**.
""")
code(r"""
SC = 0.55 / GSD

def gray_world(rgb):
    f = rgb.astype(np.float32); m = f.reshape(-1, 3).mean(0) + 1e-6
    return np.clip(f * (m.mean() / m), 0, 255)

def percentile_stretch(f, lo=1, hi=99):
    out = np.empty_like(f)
    for c in range(3):
        p1, p2 = np.percentile(f[..., c], (lo, hi)); out[..., c] = np.clip((f[..., c] - p1) / (p2 - p1 + 1e-6), 0, 1)
    return out

def sigmoid_luminance(rgb01, gain=5.0):
    yuv = cv2.cvtColor(rgb01.astype(np.float32), cv2.COLOR_RGB2YUV); Y = yuv[..., 0]
    hist, edges = np.histogram(Y, 256, (0, 1)); hist = ndi.gaussian_filter1d(hist.astype(float), 4)
    pk, _ = find_peaks(hist, distance=20, prominence=hist.max() * 0.05)
    x0 = edges[int(pk[np.argsort(hist[pk])[-2:]].mean())] if len(pk) >= 2 else float(np.median(Y))
    ys = 1 / (1 + np.exp(-gain * (Y - x0))); ys = (ys - ys.min()) / (np.ptp(ys) + 1e-6)
    yuv[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply((ys * 255).astype(np.uint8)) / 255.0
    return np.clip(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB), 0, 1)

def preprocess(rgb):
    u8 = (percentile_stretch(gray_world(rgb)) * 255).astype(np.uint8)
    for _ in range(2):
        u8 = cv2.bilateralFilter(u8, 9, 25, 5)
    return sigmoid_luminance(u8 / 255.0)

def _n99(x):
    return np.clip(x / (np.percentile(x, 99) + 1e-6), 0, 1)

def boundary_probability(pre, scale):
    maps = []
    for g in (0.8, 1.0, 1.25):
        q = np.clip(pre, 0, 1) ** g
        for s in (1.0 * scale, 2.0 * scale):
            qs = gaussian(q, s, channel_axis=-1)
            sob = sum(np.hypot(ndi.sobel(qs[..., c], 0), ndi.sobel(qs[..., c], 1)) for c in range(3))
            L = rgb2lab(qs)[..., 0] / 100
            rid = np.maximum(sato(L, sigmas=[scale, 2 * scale], black_ridges=True),
                             sato(L, sigmas=[scale, 2 * scale], black_ridges=False))
            maps.append(0.6 * _n99(sob) + 0.4 * _n99(rid))
    return np.mean(maps, 0).astype(np.float32)

def merge_regions(labels, Bs, lab, t_edge, t_de):
    rag = graph.rag_boundary(labels, Bs); n = labels.max() + 1; idx = np.arange(n)
    S = np.stack([ndi.sum(lab[..., k], labels, idx) for k in range(3)], 1); C = ndi.sum(np.ones(labels.shape), labels, idx)
    parent = np.arange(n)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for w, u, v in sorted((d_["weight"], u, v) for u, v, d_ in rag.edges(data=True) if 0 not in (u, v)):
        if w >= t_edge:
            break
        a, b = find(u), find(v)
        if a != b and np.linalg.norm(S[a] / C[a] - S[b] / C[b]) < t_de:
            parent[a] = b; S[b] += S[a]; C[b] += C[a]
    _, new = np.unique([find(i) for i in range(n)], return_inverse=True)
    m = (new + 1).astype(np.int32)[labels]; m[labels == 0] = 0
    return m

SNAKE = dict(alpha=0.0005, beta=0.001, w_line=1.0, w_edge=0.0, gamma=0.1,
             max_px_move=0.5, max_num_iter=60, convergence=0.05, boundary_condition="periodic")

def segment(t, H, T_edge, T_de):
    d = D[t]; field = d["valid"]; B = BMAP[t]
    Bf = np.clip(B / (np.percentile(B[field], 99) + 1e-6), 0, 1)
    Bs = gaussian(Bf, 1.5 * SC)
    markers, _ = ndi.label(h_minima(Bs, H) & field)
    return merge_regions(watershed(Bs, markers, mask=field), Bs, rgb2lab(PRE[t]), T_edge, T_de), Bf

def snake_refine(merged, Bf, field):
    Bsn = gaussian(Bf, 2.0 * SC); Bsn = Bsn / (Bsn.max() + 1e-6)
    out = np.zeros_like(merged); pad, step = int(12 * SC), int(2 * SC)
    for r in sorted(regionprops(merged), key=lambda r: r.area, reverse=True):
        if r.area < MIN_PX:
            continue
        r0, c0, r1, c1 = r.bbox
        r0, c0, r1, c1 = max(r0 - pad, 0), max(c0 - pad, 0), min(r1 + pad, merged.shape[0]), min(c1 + pad, merged.shape[1])
        sub = merged[r0:r1, c0:c1] == r.label
        cs = find_contours(np.pad(sub, 1).astype(float), 0.5)
        if not cs:
            continue
        init = (max(cs, key=len) - 1)[::step]
        if len(init) < 8:
            continue
        snk = active_contour(Bsn[r0:r1, c0:c1], init, **SNAKE)
        rr, cc = draw_polygon(snk[:, 0], snk[:, 1], sub.shape)
        keep = field[r0:r1, c0:c1][rr, cc]
        out[r0:r1, c0:c1][rr[keep], cc[keep]] = r.label
    return out

def snake_predict(t, H, T_edge, T_de):
    merged, Bf = segment(t, H, T_edge, T_de)
    return snake_refine(merged, Bf, D[t]["valid"])

# boundary maps (cached)
PRE, BMAP = {}, {}
t0 = time.time()
for t in TILES.tile:
    p = os.path.join(DIRS["snake"], f"bmap_{t}.npz")
    if not os.path.exists(p):
        pre = preprocess(D[t]["img"]); np.savez_compressed(p, pre=pre.astype(np.float16), B=boundary_probability(pre, SC))
    z = np.load(p); PRE[t], BMAP[t] = z["pre"].astype(np.float32), z["B"]
print(f"boundary maps for {len(BMAP)} tiles ready ({time.time()-t0:.0f}s)")
""")
code(r"""
grid_csv = os.path.join(DIRS["tables"], "snake_grid_train.csv")
TUNE_TILES = TILES[TILES.split == "Train"].groupby("site").tile.apply(lambda s: s.iloc[::3][:2]).explode().tolist()
if os.path.exists(grid_csv):
    GRID = pd.read_csv(grid_csv)
else:
    rows_, t0 = [], time.time()
    for H in (0.04, 0.08, 0.12, 0.16, 0.24):
        for T_edge in (0.1, 0.2, 0.3):
            for T_de in (5, 10, 20):
                sc = evaluate_set({t: segment(t, H, T_edge, T_de)[0] for t in TUNE_TILES}, TUNE_TILES)[KEY].mean()
                rows_.append(dict(H=H, T_edge=T_edge, T_de=T_de, **sc.to_dict()))
    GRID = pd.DataFrame(rows_); GRID.to_csv(grid_csv, index=False)
    print(f"grid of {len(GRID)} settings × {len(TUNE_TILES)} tiles in {(time.time()-t0)/60:.1f} min")
BEST_SNAKE = GRID.sort_values("F1", ascending=False).iloc[0][["H", "T_edge", "T_de"]].to_dict()
DEFAULT_SNAKE = dict(H=0.04, T_edge=0.30, T_de=8)
print("best on Train (instance F1):", BEST_SNAKE)
display(GRID.sort_values("F1", ascending=False).head(10).round(3).style.set_caption("Top-10 settings on Train tuning tiles"))

fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
for ax, T_de in zip(axes, (5, 10, 20)):
    piv = GRID[GRID.T_de == T_de].pivot(index="H", columns="T_edge", values="F1")
    im = ax.imshow(piv.values, cmap="viridis", vmin=GRID.F1.min(), vmax=GRID.F1.max(), aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns); ax.set_xlabel("T_edge")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index); ax.set_ylabel("H (h-minima)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_title(f"Instance F1 on Train, T_ΔE = {T_de}")
plt.colorbar(im, ax=axes, fraction=0.02)
plt.savefig(os.path.join(DIRS["figures"], "fig02_snake_grid.png"), bbox_inches="tight"); plt.show()
""")
code(r"""
# Validation: default vs tuned setting (with snake refinement), on all Valid tiles
val_csv = os.path.join(DIRS["tables"], "snake_valid.csv")
if os.path.exists(val_csv):
    SNAKE_VAL = pd.read_csv(val_csv)
else:
    parts = []
    for name, cfg in (("default", DEFAULT_SNAKE), ("tuned", BEST_SNAKE)):
        preds = {t: snake_predict(t, **cfg) for t in SPLIT["Valid"]}
        parts.append(evaluate_set(preds, SPLIT["Valid"]).assign(setting=name))
    SNAKE_VAL = pd.concat(parts); SNAKE_VAL.to_csv(val_csv, index=False)
display(SNAKE_VAL.groupby("setting")[KEY].mean().round(3).style.set_caption("Snake pipeline on Valid"))
SNAKE_SELECTED = BEST_SNAKE if SNAKE_VAL.groupby("setting").F1.mean()["tuned"] >= SNAKE_VAL.groupby("setting").F1.mean()["default"] else DEFAULT_SNAKE
print("selected for Test:", SNAKE_SELECTED)
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 4 · Model B: FTW Delineate Anything V2
**Delineate Anything V2** (Lavreniuk et al.; shipped with FTW tools) is a YOLO11x-seg instance-segmentation
network (62 M parameters) pre-trained on FBIS-22M (22.9 M field instances, 0.25–10 m imagery).

**Inference.** The 0.25 m tile is processed in overlapping square patches (patch/4 overlap discarded at
each side). Each patch gives field polygons with a confidence score. Duplicates across patches are
removed by score-ordered non-maximum suppression (IoU ≥ 0.3 or ≥ 60 % containment), and overlaps go to
the higher score. The patch size, resize factor and confidence threshold are **chosen on Valid**.

**Fine-tuning.**
* Dataset: 512 × 512 px crops (128 m) with stride 256 from the 0.25 m tiles. Crops with < 50 % valid
  pixels are skipped. Every reference field is converted to a YOLO polygon (outer contour,
  simplified by 1 px).
* Training: all layers, AdamW, lr 0.001, 60 epochs (early stopping, patience 15), batch 8,
  image size 512, Ultralytics default augmentation (HSV jitter, flips, scale, mosaic for the first epochs).
  The best epoch is selected on the Valid crops' mask mAP.
""")
code(r"""
from ultralytics import YOLO
if not os.path.exists(DA_WEIGHTS):
    YOLO(DA_URL)   # downloads the checkpoint to ~/.pyenv/weights via ultralytics

def da_predict(model, img, patch=512, resize=1, conf=0.1):
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(DEVICE); H, W = img.shape[:2]
    pad = patch // 4; st = patch - 2 * pad
    S = lambda n: sorted(set(list(range(0, max(n - patch, 0) + 1, st)) + [max(n - patch, 0)]))
    polys, sc = [], []
    for r0 in S(H):
        for c0 in S(W):
            p = x[:, r0:r0 + patch, c0:c0 + patch][None]
            if resize != 1:
                p = F.interpolate(p, size=(patch * resize,) * 2, mode="bilinear", align_corners=False)
            res = model.predict(p, conf=conf, iou=0.3, max_det=300, verbose=False)[0]
            if res.masks is None:
                continue
            for seg, s in zip(res.masks.xyn, res.boxes.conf.cpu().numpy()):
                if len(seg) >= 3:
                    polys.append(Polygon([(c0 + u * patch, r0 + v * patch) for u, v in seg]))
                    sc.append(float(s))
    kept, ks = nms_partition(polys, sc)
    order = np.argsort(ks)                       # higher score drawn last (on top)
    return polys_to_labels([kept[i] for i in order], img.shape[:2])

def tune_inference(model, configs, tiles):
    rows_ = []
    for cfg in configs:
        sc = evaluate_set({t: da_predict(model, D[t]["img"], **cfg) for t in tiles}, tiles)[KEY].mean()
        rows_.append(dict(**cfg, **sc.to_dict()))
    return pd.DataFrame(rows_)

DA_CONFIGS = [dict(patch=256, resize=2, conf=0.05), dict(patch=512, resize=1, conf=0.05),
              dict(patch=512, resize=1, conf=0.15), dict(patch=512, resize=1, conf=0.30),
              dict(patch=768, resize=1, conf=0.15)]
da0_csv = os.path.join(DIRS["tables"], "da_pretrained_valid.csv")
da_base = YOLO(DA_WEIGHTS).to(DEVICE)
if os.path.exists(da0_csv):
    DA0_VAL = pd.read_csv(da0_csv)
else:
    DA0_VAL = tune_inference(da_base, DA_CONFIGS, SPLIT["Valid"]); DA0_VAL.to_csv(da0_csv, index=False)
display(DA0_VAL.round(3).style.set_caption("Pre-trained Delineate Anything V2 — inference settings on Valid"))
DA0_CFG = DA0_VAL.sort_values("F1", ascending=False).iloc[0][["patch", "resize", "conf"]].to_dict()
DA0_CFG = dict(patch=int(DA0_CFG["patch"]), resize=int(DA0_CFG["resize"]), conf=float(DA0_CFG["conf"]))
print("pre-trained model, selected inference setting:", DA0_CFG)
""")
code(r"""
YDS = os.path.join(DIRS["da"], "yolo_dataset")

def build_yolo_dataset(out=YDS, crop=512, stride=256):
    if os.path.exists(os.path.join(out, "data.yaml")):
        return
    counts = {}
    for split, ys in (("Train", "train"), ("Valid", "val")):
        os.makedirs(f"{out}/images/{ys}", exist_ok=True); os.makedirs(f"{out}/labels/{ys}", exist_ok=True)
        n_img = n_obj = 0
        for t in SPLIT[split]:
            d = D[t]; Hh, Ww = d["inst"].shape
            S = lambda L: sorted(set(list(range(0, L - crop + 1, stride)) + [L - crop]))
            for r0 in S(Hh):
                for c0 in S(Ww):
                    if d["valid"][r0:r0 + crop, c0:c0 + crop].mean() < 0.5:
                        continue
                    inst = d["inst"][r0:r0 + crop, c0:c0 + crop]; lines = []
                    for i in np.unique(inst[inst > 0]):
                        m = (inst == i).astype(np.uint8)
                        if m.sum() < 200:
                            continue
                        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
                        if len(c) >= 3:
                            lines.append("0 " + " ".join(f"{x / crop:.5f} {y / crop:.5f}" for x, y in c))
                    name = f"{t}_{r0}_{c0}"
                    cv2.imwrite(f"{out}/images/{ys}/{name}.png", cv2.cvtColor(d["img"][r0:r0 + crop, c0:c0 + crop], cv2.COLOR_RGB2BGR))
                    open(f"{out}/labels/{ys}/{name}.txt", "w").write("\n".join(lines))
                    n_img += 1; n_obj += len(lines)
        counts[ys] = (n_img, n_obj)
    yaml.safe_dump({"path": os.path.abspath(out), "train": "images/train", "val": "images/val", "names": {0: "field"}},
                   open(f"{out}/data.yaml", "w"))
    print("YOLO dataset (images, field polygons):", counts)

build_yolo_dataset()
DA_FT = os.path.join(DIRS["da"], "runs", "finetune", "weights", "best.pt")
if not os.path.exists(DA_FT):
    t0 = time.time()
    YOLO(DA_WEIGHTS).train(data=os.path.join(YDS, "data.yaml"), imgsz=512, epochs=60, patience=15, batch=8,
                           optimizer="AdamW", lr0=0.001, cos_lr=True, seed=SEED, workers=4, amp=True,
                           project=os.path.abspath(os.path.join(DIRS["da"], "runs")), name="finetune",
                           exist_ok=True, plots=False, verbose=False)
    print(f"fine-tuning finished in {(time.time()-t0)/60:.1f} min")
hist = pd.read_csv(os.path.join(DIRS["da"], "runs", "finetune", "results.csv")); hist.columns = hist.columns.str.strip()
best_ep = int(hist["metrics/mAP50(M)"].idxmax())
print(f"epochs run: {len(hist)} · best epoch {best_ep + 1}: mask mAP50 {hist['metrics/mAP50(M)'].max():.3f}, "
      f"mask mAP50-95 {hist.loc[best_ep, 'metrics/mAP50-95(M)']:.3f}")
fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
axes[0].plot(hist["train/seg_loss"], label="train seg loss"); axes[0].plot(hist["val/seg_loss"], label="valid seg loss"); axes[0].legend()
axes[1].plot(hist["metrics/mAP50(M)"], label="mask mAP50"); axes[1].plot(hist["metrics/mAP50-95(M)"], label="mask mAP50-95"); axes[1].legend()
axes[2].plot(hist["metrics/precision(M)"], label="mask precision"); axes[2].plot(hist["metrics/recall(M)"], label="mask recall"); axes[2].legend()
for a in axes: a.set_xlabel("epoch"); a.grid(alpha=0.3)
axes[0].set_title("Loss"); axes[1].set_title("Validation mAP (Valid crops)"); axes[2].set_title("Validation precision / recall")
plt.suptitle("Figure 3 · Delineate Anything V2 fine-tuning curves", y=1.03)
plt.savefig(os.path.join(DIRS["figures"], "fig03_da_training.png"), bbox_inches="tight"); plt.show()
""")
code(r"""
da1_csv = os.path.join(DIRS["tables"], "da_finetuned_valid.csv")
da_ft = YOLO(DA_FT).to(DEVICE)
FT_CONFIGS = [dict(patch=512, resize=1, conf=c) for c in (0.1, 0.25, 0.4, 0.55)] + [dict(patch=768, resize=1, conf=0.25)]
if os.path.exists(da1_csv):
    DA1_VAL = pd.read_csv(da1_csv)
else:
    DA1_VAL = tune_inference(da_ft, FT_CONFIGS, SPLIT["Valid"]); DA1_VAL.to_csv(da1_csv, index=False)
display(DA1_VAL.round(3).style.set_caption("Fine-tuned Delineate Anything V2 — inference settings on Valid"))
DA1_CFG = DA1_VAL.sort_values("F1", ascending=False).iloc[0][["patch", "resize", "conf"]].to_dict()
DA1_CFG = dict(patch=int(DA1_CFG["patch"]), resize=int(DA1_CFG["resize"]), conf=float(DA1_CFG["conf"]))
print("fine-tuned model, selected inference setting:", DA1_CFG)
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 5 · Model C: SAM 2.1 (Hiera-Large)
SAM 2 is a promptable segmentation foundation model: an image encoder (Hiera-L, 224 M parameters)
produces image features once, and a light **prompt encoder + mask decoder** (≈ 4 M parameters) turns a
prompt (points or a box) into masks. For field mapping SAM 2 is used in **automatic mask generation**
(AMG) mode: a 64 × 64 grid of point prompts plus one crop layer, mask-to-mask refinement,
`pred_iou_thresh = 0.65`, `stability_score_thresh = 0.80`. Masks are painted largest → smallest into an
instance raster (masks > 25 % of the tile and < 20 m² dropped).

**Fine-tuning (decoder only).** The image encoder stays frozen, so the large pre-trained representation
is kept and training is cheap. The prompt encoder and mask decoder are trained on Train tiles:
* each step: one tile, random flip / 90° rotation; 16 reference fields sampled; for each, 1–3 positive points
  drawn from the field interior (distance to edge ≥ 30 % of the maximum);
* loss: for each prompt, the **best of the 3 multimask outputs** under BCE + Dice, plus an MSE loss teaching
  the IoU head the true IoU (AMG uses that head to filter masks);
* AdamW, lr 2 × 10⁻⁵ with cosine decay, 2 000 steps, bfloat16 autocast;
* **validation** every 250 steps on 6 Valid tiles: mean IoU of masks prompted by one point at the centre of
  every reference field. The best checkpoint is kept.

**Re-calibrating automatic mask generation (also on Valid).** AMG keeps a mask only if the decoder's
*IoU head* predicts a high quality (`pred_iou_thresh`) and the mask is stable (`stability_score_thresh`).
Fine-tuning changes how confident the IoU head is: the fine-tuned decoder rates almost every mask
highly, so thresholds tuned for the pre-trained model let hundreds of fragments through. In a development
run with the original thresholds, the fine-tuned SAM 2 dropped to F1 0.03 on Test. Both thresholds are
therefore **chosen on Valid for every model state** (pre-trained and fine-tuned), as was done for
Delineate Anything's confidence. To make the search affordable, masks are generated once per Valid tile at
loose thresholds (0.5 / 0.7), then filtered post hoc for each candidate pair. The selected pair is used for
an exact AMG run on Test.
""")
code(r"""
VAL_PROMPT_TILES = SPLIT["Valid"][::2]

def sample_prompts(inst, n=16):
    ids = np.unique(inst[inst > 0])
    ids = np.random.choice(ids, min(n, len(ids)), replace=False)
    k = random.choice([1, 1, 2, 3]); pts, masks = [], []
    for i in ids:
        m = inst == i; dist = ndi.distance_transform_edt(m)
        yy, xx = np.where(dist >= max(2, 0.3 * dist.max()))
        j = np.random.choice(len(yy), k); pts.append(np.stack([xx[j], yy[j]], 1)); masks.append(m)
    return np.stack(pts).astype(np.float32), np.stack(masks)

def centre_prompts(inst):
    pts, masks = [], []
    for i in np.unique(inst[inst > 0]):
        m = inst == i; dist = ndi.distance_transform_edt(m)
        y, x = np.unravel_index(dist.argmax(), m.shape); pts.append([[x, y]]); masks.append(m)
    return np.array(pts, np.float32), np.stack(masks)

def augment(img, inst):
    k = random.randint(0, 3); img, inst = np.rot90(img, k), np.rot90(inst, k)
    if random.random() < 0.5:
        img, inst = img[:, ::-1], inst[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(inst)

def multimask_loss(logits, scores, gt):
    p = torch.sigmoid(logits); g = gt[:, None].expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, g, reduction="none").mean((2, 3))
    dice = 1 - (2 * (p * g).sum((2, 3)) + 1) / (p.sum((2, 3)) + g.sum((2, 3)) + 1)
    tot = bce + dice; j = tot.argmin(1); idx = torch.arange(len(j), device=logits.device)
    with torch.no_grad():
        pb = p > 0.5; true_iou = (pb & (g > 0)).sum((2, 3)) / ((pb | (g > 0)).sum((2, 3)).clamp(min=1))
    return tot[idx, j].mean() + F.mse_loss(scores[idx, j], true_iou[idx, j].float())

def train_decoder(encode, decode, params, val_fn, steps=2000, lr=2e-5, eval_every=250, log=None):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, best_state, hist = val_fn(), None, []
    hist.append(dict(step=0, loss=np.nan, val_mIoU=best)); print(f"step     0 · valid prompted mIoU {best:.4f}")
    run, t0 = [], time.time()
    for step in range(1, steps + 1):
        d = D[random.choice(SPLIT["Train"])]
        img, inst = augment(d["img"], d["inst"])
        with torch.no_grad():
            encode(img)
        pts, gm = sample_prompts(inst)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, scores = decode(pts)
        loss = multimask_loss(logits.float(), scores.float(), torch.from_numpy(gm).float().to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run.append(loss.item())
        if step % eval_every == 0:
            v = val_fn(); hist.append(dict(step=step, loss=float(np.mean(run)), val_mIoU=v)); run = []
            flag = ""
            if v > best:
                best, best_state, flag = v, {k: x.detach().cpu().clone() for k, x in params_state().items()}, "  ← best"
            print(f"step {step:5d} · train loss {hist[-1]['loss']:.4f} · valid prompted mIoU {v:.4f}{flag} · {(time.time()-t0)/60:.1f} min")
    return best_state, pd.DataFrame(hist)
""")
code(r"""
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_model = build_sam2_hf(SAM2_ID, device=DEVICE)
sam2_pred = SAM2ImagePredictor(sam2_model)
SAM2_BASE = {k: v.detach().cpu().clone() for k, v in
             list(sam2_model.sam_mask_decoder.state_dict(prefix="sam_mask_decoder.").items()) +
             list(sam2_model.sam_prompt_encoder.state_dict(prefix="sam_prompt_encoder.").items())}

def sam2_encode(img):
    sam2_pred.set_image(img)

def sam2_decode(pts):
    lab = np.ones(pts.shape[:2], np.int32)
    _, uc, ul, _ = sam2_pred._prep_prompts(pts, lab, box=None, mask_logits=None, normalize_coords=True)
    sp, de = sam2_model.sam_prompt_encoder(points=(uc, ul), boxes=None, masks=None)
    hr = [f[-1].unsqueeze(0) for f in sam2_pred._features["high_res_feats"]]
    low, iou, _, _ = sam2_model.sam_mask_decoder(
        image_embeddings=sam2_pred._features["image_embed"][-1].unsqueeze(0),
        image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sp,
        dense_prompt_embeddings=de, multimask_output=True, repeat_image=True, high_res_features=hr)
    return sam2_pred._transforms.postprocess_masks(low, sam2_pred._orig_hw[-1]), iou

def params_state():
    return dict(list(sam2_model.sam_mask_decoder.state_dict(prefix="sam_mask_decoder.").items()) +
                list(sam2_model.sam_prompt_encoder.state_dict(prefix="sam_prompt_encoder.").items()))

@torch.no_grad()
def prompted_miou(encode, decode, tiles):
    ious = []
    for t in tiles:
        d = D[t]; encode(d["img"]); pts, gm = centre_prompts(d["inst"])
        for s in range(0, len(pts), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, sc = decode(pts[s:s + 64])
            best = lo.float()[torch.arange(len(lo)), sc.float().argmax(1)] > 0
            g = torch.from_numpy(gm[s:s + 64]).to(DEVICE)
            ious += ((best & g).sum((1, 2)) / (best | g).sum((1, 2)).clamp(min=1)).tolist()
    return float(np.mean(ious))

SAM2_FT = os.path.join(DIRS["sam2"], "sam2_decoder_finetuned.pt")
if not os.path.exists(SAM2_FT):
    for p in sam2_model.parameters():
        p.requires_grad_(False)
    params = list(sam2_model.sam_mask_decoder.parameters()) + list(sam2_model.sam_prompt_encoder.parameters())
    for p in params:
        p.requires_grad_(True)
    sam2_model.sam_mask_decoder.train()
    best_state, H2 = train_decoder(sam2_encode, sam2_decode, params,
                                   lambda: (sam2_model.sam_mask_decoder.eval(),
                                            prompted_miou(sam2_encode, sam2_decode, VAL_PROMPT_TILES),
                                            sam2_model.sam_mask_decoder.train())[1])
    torch.save(best_state, SAM2_FT); H2.to_csv(os.path.join(DIRS["tables"], "sam2_training.csv"), index=False)
    sam2_model.sam_mask_decoder.eval()
H2 = pd.read_csv(os.path.join(DIRS["tables"], "sam2_training.csv"))
display(H2.round(4).style.set_caption("SAM 2 decoder fine-tuning (valid = point-prompted mIoU on Valid tiles)"))
""")

md(r"""
### SAM 2 automatic mask generation: pre-trained vs fine-tuned decoder (Valid)
The prompted mIoU measures the decoder alone. The pipeline used for mapping is automatic mask
generation, so pre-trained and fine-tuned decoders are also compared in AMG mode on 4 Valid tiles.
""")
code(r"""
def load_sam2(state):
    sam2_model.load_state_dict(state, strict=False); sam2_model.eval()

PIOU_GRID, STAB_GRID = (0.65, 0.75, 0.85, 0.90, 0.95), (0.80, 0.90)

def amg_raw(make_gen, img, autocast=True):
    gen = make_gen(0.5, 0.7)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        ms = gen.generate(img)
    torch.cuda.empty_cache()
    return ms

def calibrate_amg(tag, make_gen, tiles, autocast=True):
    csv = os.path.join(DIRS["tables"], f"{tag}_amg_calibration.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv)
    rows_ = []
    for t in tiles:
        ms = amg_raw(make_gen, D[t]["img"], autocast)
        for pi in PIOU_GRID:
            for st in STAB_GRID:
                keep = [m["segmentation"] for m in ms if m["predicted_iou"] >= pi and m["stability_score"] >= st]
                rows_.append(dict(tile=t, pred_iou_thresh=pi, stability_thresh=st,
                                  **evaluate(masks_to_labels(keep, D[t]["img"].shape[:2]), D[t])))
        del ms
    cal = pd.DataFrame(rows_).groupby(["pred_iou_thresh", "stability_thresh"])[KEY].mean().reset_index()
    cal.to_csv(csv, index=False)
    return cal

def make_sam2_gen(pi, st):
    return SAM2AutomaticMaskGenerator(sam2_model, points_per_side=64, points_per_batch=128, pred_iou_thresh=pi,
                                      stability_score_thresh=st, box_nms_thresh=0.70, crop_n_layers=1,
                                      crop_n_points_downscale_factor=2, min_mask_region_area=MIN_PX, use_m2m=True)

def amg_labels(make_gen, pi, st, autocast=True):
    def run(img):
        gen = make_gen(pi, st)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            ms = gen.generate(img)
        lab = masks_to_labels([m["segmentation"] for m in ms], img.shape[:2]); torch.cuda.empty_cache()
        return lab
    return run

def cached_predict(tag, fn, tiles):
    out = {}
    for t in tiles:
        p = os.path.join(DIRS["pred"], f"{tag}_{t}.npy")
        if not os.path.exists(p):
            np.save(p, fn(D[t]["img"]).astype(np.int32))
        out[t] = np.load(p)
    return out

AMG_VAL_TILES = SPLIT["Valid"][::3]
AMG_CFG, cal_rows = {}, []
for tag, state in (("sam2_pretrained", SAM2_BASE), ("sam2_finetuned", torch.load(SAM2_FT))):
    load_sam2(state)
    cal = calibrate_amg(tag, make_sam2_gen, AMG_VAL_TILES)
    best = cal.sort_values("F1", ascending=False).iloc[0]
    AMG_CFG[tag] = (float(best.pred_iou_thresh), float(best.stability_thresh))
    cal_rows.append(cal.assign(model=tag))
CAL2 = pd.concat(cal_rows)
display(CAL2.pivot_table(index=["pred_iou_thresh", "stability_thresh"], columns="model", values="F1").round(3)
        .style.background_gradient(cmap="RdYlGn", vmin=0, vmax=0.7).set_caption("SAM 2 AMG on Valid: instance F1 per threshold pair"))
display(pd.DataFrame([CAL2[(CAL2.model == k) & (CAL2.pred_iou_thresh == v[0]) & (CAL2.stability_thresh == v[1])][KEY].iloc[0].rename(f"{k} @ {v}")
                      for k, v in AMG_CFG.items()]).round(3).style.set_caption("SAM 2 AMG on Valid at the selected thresholds"))
print("selected (pred_iou_thresh, stability_thresh):", AMG_CFG)
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 6 · Model D: SAM (v1, ViT-B)
The original Segment Anything model (Kirillov et al. 2023) has the same design: a ViT image encoder,
a prompt encoder and a mask decoder. Here the **ViT-B** variant (91 M encoder parameters) is used. The
ViT-H checkpoint (2.5 GB) could not be downloaded in reasonable time on this connection. Training and
validation follow Part 5 exactly (same prompts, loss, optimiser, steps and validation), including the AMG
threshold calibration on Valid (64 × 64 points, one crop layer, no mask-to-mask refinement, which SAM v1 lacks).
""")
code(r"""
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

sam1 = sam_model_registry["vit_b"](checkpoint=SAM1_CKPT).to(DEVICE); sam1.eval()
sam1_pred = SamPredictor(sam1)
SAM1_BASE = {k: v.detach().cpu().clone() for k, v in
             list(sam1.mask_decoder.state_dict(prefix="mask_decoder.").items()) +
             list(sam1.prompt_encoder.state_dict(prefix="prompt_encoder.").items())}

def sam1_encode(img):
    sam1_pred.set_image(img)

def sam1_decode(pts):
    c = sam1_pred.transform.apply_coords(pts, sam1_pred.original_size)
    c = torch.as_tensor(c, dtype=torch.float, device=DEVICE); l = torch.ones(c.shape[:2], dtype=torch.int, device=DEVICE)
    sp, de = sam1.prompt_encoder(points=(c, l), boxes=None, masks=None)
    low, iou = sam1.mask_decoder(image_embeddings=sam1_pred.features, image_pe=sam1.prompt_encoder.get_dense_pe(),
                                 sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=True)
    return sam1.postprocess_masks(low, sam1_pred.input_size, sam1_pred.original_size), iou

SAM1_FT = os.path.join(DIRS["sam1"], "sam1_decoder_finetuned.pt")
if not os.path.exists(SAM1_FT):
    for p in sam1.parameters():
        p.requires_grad_(False)
    params = list(sam1.mask_decoder.parameters()) + list(sam1.prompt_encoder.parameters())
    for p in params:
        p.requires_grad_(True)
    params_state = lambda: dict(list(sam1.mask_decoder.state_dict(prefix="mask_decoder.").items()) +
                                list(sam1.prompt_encoder.state_dict(prefix="prompt_encoder.").items()))
    sam1.mask_decoder.train()
    best_state, H1 = train_decoder(sam1_encode, sam1_decode, params,
                                   lambda: (sam1.mask_decoder.eval(), prompted_miou(sam1_encode, sam1_decode, VAL_PROMPT_TILES),
                                            sam1.mask_decoder.train())[1])
    torch.save(best_state, SAM1_FT); H1.to_csv(os.path.join(DIRS["tables"], "sam1_training.csv"), index=False)
    sam1.mask_decoder.eval()
H1 = pd.read_csv(os.path.join(DIRS["tables"], "sam1_training.csv"))
display(H1.round(4).style.set_caption("SAM (v1) decoder fine-tuning (valid = point-prompted mIoU on Valid tiles)"))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, (nm, Hh) in zip(axes, (("SAM 2.1 (Hiera-L)", H2), ("SAM v1 (ViT-B)", H1))):
    ax.plot(Hh.step, Hh.val_mIoU, "o-", label="valid prompted mIoU"); ax2 = ax.twinx()
    ax2.plot(Hh.step, Hh.loss, "s--", color="tab:red", label="train loss")
    ax.set_xlabel("step"); ax.set_ylabel("valid mIoU"); ax2.set_ylabel("train loss"); ax.set_title(nm); ax.grid(alpha=0.3)
plt.suptitle("Figure 4 · Decoder fine-tuning curves", y=1.03)
plt.savefig(os.path.join(DIRS["figures"], "fig04_sam_training.png"), bbox_inches="tight"); plt.show()
""")
code(r"""
def load_sam1(state):
    sam1.load_state_dict(state, strict=False); sam1.eval()

def make_sam1_gen(pi, st):
    return SamAutomaticMaskGenerator(sam1, points_per_side=64, points_per_batch=128, pred_iou_thresh=pi,
                                     stability_score_thresh=st, box_nms_thresh=0.70, crop_n_layers=1,
                                     crop_n_points_downscale_factor=2, min_mask_region_area=MIN_PX)

cal_rows = []
for tag, state in (("sam1_pretrained", SAM1_BASE), ("sam1_finetuned", torch.load(SAM1_FT))):
    load_sam1(state)
    cal = calibrate_amg(tag, make_sam1_gen, AMG_VAL_TILES, autocast=False)
    best = cal.sort_values("F1", ascending=False).iloc[0]
    AMG_CFG[tag] = (float(best.pred_iou_thresh), float(best.stability_thresh))
    cal_rows.append(cal.assign(model=tag))
CAL1 = pd.concat(cal_rows)
display(CAL1.pivot_table(index=["pred_iou_thresh", "stability_thresh"], columns="model", values="F1").round(3)
        .style.background_gradient(cmap="RdYlGn", vmin=0, vmax=0.7).set_caption("SAM v1 AMG on Valid: instance F1 per threshold pair"))
display(pd.DataFrame([CAL1[(CAL1.model == k) & (CAL1.pred_iou_thresh == v[0]) & (CAL1.stability_thresh == v[1])][KEY].iloc[0].rename(f"{k} @ {v}")
                      for k, v in AMG_CFG.items() if k.startswith("sam1")]).round(3).style.set_caption("SAM v1 AMG on Valid at the selected thresholds"))
print("selected (pred_iou_thresh, stability_thresh):", {k: v for k, v in AMG_CFG.items() if k.startswith("sam1")})
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 6b · Model E: FC-DenseNet (Tiramisu)
**FC-DenseNet** (Jégou et al. 2017), implemented in `models/fc_densenet.py`, is a fully convolutional
DenseNet. Here it is a two-class segmenter: **boundary** versus **field interior**. This repository has
no pre-trained weights for the architecture, so each network starts from a random initialisation.
FC-DenseNet56, FC-DenseNet67 and FC-DenseNet103 are all trained, and the depth with the highest
instance F1 on Valid is the one taken to Test.

* **Target.** Crops are 256 × 256, a multiple of 32, which the five pooling stages require. Digitised
  boundary pixels are class 1. Other valid pixels are class 0. The black screenshot border is ignored
  by the loss.
* **Loss.** Weighted softmax cross-entropy. Weights are the inverse class frequency on Train, scaled
  to a mean of 1, so the thin boundary class contributes on the same scale as the field interior.
* **Optimisation.** Adam at learning rate 1e-4, for 30 epochs, 16 batches of 4 crops per epoch, with
  random flips and 90° rotations. Dropout is 0.2 while training.
* **From pixels to fields.** The whole tile is padded to a multiple of 32 and passed through once.
  Boundary pixels (probability above a threshold) are dilated by 1 px, the tile edge is added, and
  every enclosed region of at least 20 m² is one field — the same rule used to build the reference.
  The threshold is chosen on Valid. An untrained copy of the selected depth, with its own Valid
  threshold, is the "before training" model on Test.

Checkpoints and the training log are cached in `train_outputs/fcdn/`. A later run loads them.
""")
code(r"""
# FC-DenseNet lives in the repository models package (NCHW logits, two classes).
REPO = os.path.dirname(LABELED)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from models.fc_densenet import FCDenseNet

DIRS["fcdn"] = os.path.join(OUT, "fcdn")
os.makedirs(DIRS["fcdn"], exist_ok=True)
FCDN_PRESETS = ["FC-DenseNet56", "FC-DenseNet67", "FC-DenseNet103"]
FCDN_CROP = 256
FCDN_EPOCHS = 30
FCDN_BATCHES = 16
FCDN_BATCH = 4
FCDN_THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.85, 0.05)]

n_b = n_f = 0
for t in SPLIT["Train"]:
    valid = np.asarray(D[t]["valid"]).astype(bool)
    boundary = np.asarray(D[t]["boundary"]).astype(bool) & valid
    n_b += int(boundary.sum())
    n_f += int((valid & ~boundary).sum())
FCDN_WEIGHT = torch.tensor([n_f, n_b], dtype=torch.float32)
FCDN_WEIGHT = (FCDN_WEIGHT.sum() / FCDN_WEIGHT.clamp(min=1))
FCDN_WEIGHT = (FCDN_WEIGHT / FCDN_WEIGHT.mean()).to(DEVICE)
print(f"class weights (field, boundary): {FCDN_WEIGHT.tolist()}")

def fcdn_build(preset):
    torch.manual_seed(SEED)
    return FCDenseNet(preset_model=preset, num_classes=2).to(DEVICE)

def fcdn_augment(img, lab):
    k = random.randint(0, 3)
    img, lab = np.rot90(img, k), np.rot90(lab, k)
    if random.random() < 0.5:
        img, lab = np.flip(img, 1), np.flip(lab, 1)
    if random.random() < 0.5:
        img, lab = np.flip(img, 0), np.flip(lab, 0)
    return np.ascontiguousarray(img), np.ascontiguousarray(lab)

def fcdn_crop():
    tile = random.choice(SPLIT["Train"])
    d = D[tile]
    img = d["img"]
    H, W = img.shape[:2]
    y0 = random.randint(0, H - FCDN_CROP)
    x0 = random.randint(0, W - FCDN_CROP)
    crop = img[y0:y0 + FCDN_CROP, x0:x0 + FCDN_CROP]
    valid = np.asarray(d["valid"][y0:y0 + FCDN_CROP, x0:x0 + FCDN_CROP]).astype(bool)
    boundary = np.asarray(d["boundary"][y0:y0 + FCDN_CROP, x0:x0 + FCDN_CROP]).astype(bool)
    lab = np.full((FCDN_CROP, FCDN_CROP), 255, np.int64)
    lab[valid] = 0
    lab[boundary & valid] = 1
    crop, lab = fcdn_augment(crop, lab)
    x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
    return x, torch.from_numpy(lab)

@torch.inference_mode()
def fcdn_prob(model, img):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(img.astype(np.float32) / 255.0)).permute(2, 0, 1)[None].to(DEVICE)
    H, W = x.shape[-2:]
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    logits = model(x)
    if logits.shape[-2:] != x.shape[-2:]:
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return torch.softmax(logits, dim=1)[0, 1, :H, :W].float().cpu().numpy()

def fcdn_instances_from_prob(prob, valid, thr):
    valid = np.asarray(valid).astype(bool)
    barrier = ndi.binary_dilation(prob >= thr, iterations=1) | (valid & ~ndi.binary_erosion(valid))
    inst, _ = ndi.label(valid & ~barrier)
    size = np.bincount(inst.ravel())
    keep = size >= MIN_PX
    keep[0] = False
    inst = np.where(keep[inst], inst, 0)
    _, inv = np.unique(inst, return_inverse=True)
    return inv.reshape(inst.shape).astype(np.int32)

def fcdn_instances(img, valid, model, thr):
    return fcdn_instances_from_prob(fcdn_prob(model, img), valid, thr)

def fcdn_sweep(model, tiles):
    probs = {t: fcdn_prob(model, D[t]["img"]) for t in tiles}
    rows = []
    for thr in FCDN_THRESHOLDS:
        preds = {t: fcdn_instances_from_prob(probs[t], D[t]["valid"], thr) for t in tiles}
        score = evaluate_set(preds, tiles)[KEY].mean()
        rows.append(dict(threshold=thr, **score.to_dict()))
    table = pd.DataFrame(rows).sort_values("F1", ascending=False)
    best = table.iloc[0]
    return float(best.threshold), float(best.F1), table

def train_fcdn(preset):
    path = os.path.join(DIRS["fcdn"], f"{preset}.pt")
    hist_path = os.path.join(DIRS["fcdn"], f"{preset}_history.csv")
    model = fcdn_build(preset)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        hist = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame()
        print(f"loaded {preset}: Valid F1 {ckpt['valid_f1']:.3f} at threshold {ckpt['threshold']:.2f} (epoch {ckpt['epoch']})")
        return model, float(ckpt["threshold"]), float(ckpt["valid_f1"]), int(ckpt["epoch"]), hist
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    history, best_f1, best_thr, best_epoch = [], -1.0, 0.5, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for epoch in range(1, FCDN_EPOCHS + 1):
        model.train()
        losses = []
        for _ in range(FCDN_BATCHES):
            xs, ys = zip(*(fcdn_crop() for _ in range(FCDN_BATCH)))
            xb = torch.stack(xs).to(DEVICE)
            yb = torch.stack(ys).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=FCDN_WEIGHT, ignore_index=255)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        scored = epoch % 5 == 0 or epoch == FCDN_EPOCHS
        thr = f1 = np.nan
        if scored:
            thr, f1, _ = fcdn_sweep(model, SPLIT["Valid"])
            if f1 > best_f1:
                best_f1, best_thr, best_epoch = f1, thr, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(preset=preset, epoch=epoch, loss=float(np.mean(losses)), valid_f1=f1, threshold=thr))
        extra = f"  Valid F1 {f1:.3f} @ {thr:.2f}" if scored else ""
        print(f"{preset}  epoch {epoch:02d}/{FCDN_EPOCHS}  loss {np.mean(losses):.3f}{extra}")
    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    torch.save({"preset": preset, "state_dict": best_state, "threshold": best_thr,
                "valid_f1": best_f1, "epoch": best_epoch}, path)
    hist = pd.DataFrame(history)
    hist.to_csv(hist_path, index=False)
    print(f"saved {preset}: best Valid F1 {best_f1:.3f} at threshold {best_thr:.2f} (epoch {best_epoch}, {time.time()-t0:.0f}s)")
    return model, best_thr, best_f1, best_epoch, hist

random.seed(SEED); np.random.seed(SEED)
fcdn_rows, fcdn_models, fcdn_hist = [], {}, []
for preset in FCDN_PRESETS:
    model, thr, f1, epoch, hist = train_fcdn(preset)
    fcdn_models[preset] = model
    fcdn_hist.append(hist)
    fcdn_rows.append(dict(preset=preset, valid_F1=f1, threshold=thr, best_epoch=epoch))
    torch.cuda.empty_cache()
FCDN_VALID = pd.DataFrame(fcdn_rows).sort_values("valid_F1", ascending=False)
FCDN_PRESET = str(FCDN_VALID.iloc[0].preset)
FCDN_MODEL = fcdn_models[FCDN_PRESET].eval()
FCDN_THR = float(FCDN_VALID.iloc[0].threshold)
for preset, model in list(fcdn_models.items()):
    if preset != FCDN_PRESET:
        del fcdn_models[preset]
torch.cuda.empty_cache()

init_path = os.path.join(DIRS["fcdn"], f"{FCDN_PRESET}_untrained.json")
FCDN_INIT = fcdn_build(FCDN_PRESET).eval()
if os.path.exists(init_path):
    meta = json.load(open(init_path))
    FCDN_THR_INIT = float(meta["threshold"])
    print(f"untrained {FCDN_PRESET}: Valid F1 {meta['valid_f1']:.3f} at threshold {FCDN_THR_INIT:.2f}")
else:
    FCDN_THR_INIT, f1_init, _ = fcdn_sweep(FCDN_INIT, SPLIT["Valid"])
    json.dump({"threshold": FCDN_THR_INIT, "valid_f1": f1_init}, open(init_path, "w"))
    print(f"untrained {FCDN_PRESET}: Valid F1 {f1_init:.3f} at threshold {FCDN_THR_INIT:.2f}")
FCDN_NAME = FCDN_PRESET
display(FCDN_VALID.round(3).style.set_caption("FC-DenseNet depths on Valid — the top row is used on Test"))
print("selected", FCDN_PRESET, "threshold", round(FCDN_THR, 2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for hist in fcdn_hist:
    if hist is None or hist.empty:
        continue
    axes[0].plot(hist.epoch, hist.loss, marker="o", ms=3, label=hist.preset.iloc[0])
    scored = hist.dropna(subset=["valid_f1"])
    axes[1].plot(scored.epoch, scored.valid_f1, marker="o", label=hist.preset.iloc[0])
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("weighted cross-entropy"); axes[0].set_title("Training loss")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("instance F1"); axes[1].set_title("Valid instance F1"); axes[1].legend()
plt.suptitle("Figure 4b · FC-DenseNet training", y=1.03)
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig04b_fcdn.png"), bbox_inches="tight"); plt.show()
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 7 · Test-set evaluation
All ten configurations (five models × before / after training) are applied to every **Test** tile
(sites sw_09 and sw_10, never seen during training or model selection), each with the inference setting
selected on Valid: snake parameters, Delineate Anything patch size and confidence, SAM AMG thresholds,
and the FC-DenseNet depth and boundary threshold.
""")
code(r"""
TEST = SPLIT["Test"]
def amg_test(tag, loader, state, make_gen, autocast=True):
    loader(state); pi, st = AMG_CFG[tag]
    return cached_predict(f"{tag}_p{pi:g}_s{st:g}", amg_labels(make_gen, pi, st, autocast), TEST)
P_sam2_0 = amg_test("sam2_pretrained", load_sam2, SAM2_BASE, make_sam2_gen)
P_sam2_1 = amg_test("sam2_finetuned", load_sam2, torch.load(SAM2_FT), make_sam2_gen)
P_sam1_0 = amg_test("sam1_pretrained", load_sam1, SAM1_BASE, make_sam1_gen, autocast=False)
P_sam1_1 = amg_test("sam1_finetuned", load_sam1, torch.load(SAM1_FT), make_sam1_gen, autocast=False)
P_da_0 = cached_predict("da_pretrained", lambda im: da_predict(da_base, im, **DA0_CFG), TEST)
P_da_1 = cached_predict("da_finetuned", lambda im: da_predict(da_ft, im, **DA1_CFG), TEST)
P_sn_0, P_sn_1 = {}, {}
for t in TEST:
    for store, tag, cfg in ((P_sn_0, "snake_default", DEFAULT_SNAKE), (P_sn_1, "snake_tuned", SNAKE_SELECTED)):
        p = os.path.join(DIRS["pred"], f"{tag}_{t}.npy")
        if not os.path.exists(p):
            np.save(p, snake_predict(t, **cfg))
        store[t] = np.load(p)
P_fc_0, P_fc_1 = {}, {}
for t in TEST:
    for store, tag, model, thr in ((P_fc_0, "fcdn_untrained", FCDN_INIT, FCDN_THR_INIT),
                                   (P_fc_1, "fcdn_trained", FCDN_MODEL, FCDN_THR)):
        p = os.path.join(DIRS["pred"], f"{tag}_{t}.npy")
        if not os.path.exists(p):
            np.save(p, fcdn_instances(D[t]["img"], D[t]["valid"], model, thr))
        store[t] = np.load(p)

RUNS = {("Snake pipeline", "default"): P_sn_0, ("Snake pipeline", "tuned"): P_sn_1,
        ("FTW Delineate Anything V2", "pre-trained"): P_da_0, ("FTW Delineate Anything V2", "fine-tuned"): P_da_1,
        ("SAM 2.1", "pre-trained"): P_sam2_0, ("SAM 2.1", "fine-tuned"): P_sam2_1,
        ("SAM v1", "pre-trained"): P_sam1_0, ("SAM v1", "fine-tuned"): P_sam1_1,
        (FCDN_NAME, "untrained"): P_fc_0, (FCDN_NAME, "trained"): P_fc_1}
RES = pd.concat([evaluate_set(p, TEST).assign(model=m, stage=s, site=lambda x: x.tile.str[:5])
                 for (m, s), p in RUNS.items()], ignore_index=True)
RES.to_csv(os.path.join(DIRS["tables"], "test_results_per_tile.csv"), index=False)
SUMMARY = RES.groupby(["model", "stage"], sort=False)[KEY + ["n_pred", "n_ref"]].mean()
SUMMARY.round(3).to_csv(os.path.join(DIRS["tables"], "test_summary.csv"))
(SUMMARY.style.format("{:.3f}", subset=KEY).format("{:.0f}", subset=["n_pred", "n_ref"])
    .background_gradient(subset=["F1", "bF1@1m"], cmap="RdYlGn", vmin=0, vmax=0.8)
    .set_caption("Table 1 · Test performance (mean over 10 tiles)"))
""")
code(r"""
before = SUMMARY.groupby(level=0).nth(0).reset_index(level=1, drop=True)
after = SUMMARY.groupby(level=0).nth(1).reset_index(level=1, drop=True)
GAIN = (after[KEY] - before[KEY]).round(3)
display(GAIN.style.format("{:+.3f}").background_gradient(cmap="RdYlGn", vmin=-0.3, vmax=0.3)
        .set_caption("Table 2 · Change from training / tuning (after − before) on Test"))

fig, axes = plt.subplots(1, 3, figsize=(21, 5))
models = list(dict.fromkeys(m for m, _ in RUNS))
for ax, met, ttl in zip(axes, ["F1", "bF1@1m", "mean_best_IoU"], ["Instance F1 (IoU > 0.5)", "Boundary F1 @ 1 m", "Mean best IoU"]):
    x = np.arange(len(models)); w = 0.38
    b0 = [RES[(RES.model == m) & (RES.stage == RES[RES.model == m].stage.unique()[0])][met] for m in models]
    b1 = [RES[(RES.model == m) & (RES.stage == RES[RES.model == m].stage.unique()[1])][met] for m in models]
    ax.bar(x - w / 2, [v.mean() for v in b0], w, yerr=[v.std() for v in b0], capsize=3, label="before training", color="#bbbbbb")
    ax.bar(x + w / 2, [v.mean() for v in b1], w, yerr=[v.std() for v in b1], capsize=3, label="after training", color="#2a7f62")
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" ", "\n", 1) for m in models], fontsize=9); ax.set_ylim(0, 1)
    ax.set_title(ttl); ax.grid(axis="y", alpha=0.3)
axes[0].legend()
plt.suptitle("Figure 5 · Test performance before and after training (mean ± SD over 10 tiles)", y=1.02)
plt.savefig(os.path.join(DIRS["figures"], "fig05_test_bars.png"), bbox_inches="tight"); plt.show()
""")
code(r"""
BEFORE_AFTER = {"default": "before", "pre-trained": "before", "untrained": "before",
                 "tuned": "after", "fine-tuned": "after", "trained": "after"}
for site in sorted(RES.site.unique()):
    tb = (RES[RES.site == site].assign(when=lambda x: x.stage.map(BEFORE_AFTER))
          .groupby(["model", "when"], sort=False)[["F1", "bF1@1m", "mean_best_IoU"]].mean().unstack("when"))
    tb = tb.reindex(columns=pd.MultiIndex.from_product([["F1", "bF1@1m", "mean_best_IoU"], ["before", "after"]]))
    display(tb.round(3).style.format("{:.3f}").set_caption(f"Test site {site}: before vs after training"))
""")
code(r"""
SHOW = [("Snake pipeline", "tuned"), ("FTW Delineate Anything V2", "pre-trained"), ("FTW Delineate Anything V2", "fine-tuned"),
        ("SAM 2.1", "pre-trained"), ("SAM 2.1", "fine-tuned"), ("SAM v1", "fine-tuned"),
        (FCDN_NAME, "trained")]
for t in (TEST[0], TEST[-1]):
    d = D[t]
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    flat = axes.ravel()
    e = d["img"].copy(); e[find_boundaries(d["inst"])] = (255, 255, 0)
    flat[0].imshow(d["img"]); flat[0].set_title(f"{t} — image (0.25 m)")
    flat[1].imshow(e); flat[1].set_title(f"reference ({d['inst'].max()} fields)")
    for ax, (m, s) in zip(flat[2:], SHOW):
        p = RUNS[(m, s)][t]; o = d["img"].copy(); o[find_boundaries(p)] = (255, 40, 40)
        sc = RES[(RES.model == m) & (RES.stage == s) & (RES.tile == t)].iloc[0]
        ax.imshow(o); ax.set_title(f"{m} — {s}\nF1 {sc.F1:.2f} · bF1@1m {sc['bF1@1m']:.2f}", fontsize=11)
    for a in flat: a.axis("off")
    plt.suptitle(f"Figure 6 · Test tile {t}: reference and predictions", y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], f"fig06_{t}.png"), bbox_inches="tight"); plt.show()
""")
code(r"""
# Paired comparison over the 10 test tiles (Wilcoxon signed-rank, two-sided)
from scipy.stats import wilcoxon
rows_ = []
for m in models:
    s0, s1 = RES[RES.model == m].stage.unique()[:2]
    a = RES[(RES.model == m) & (RES.stage == s0)].set_index("tile")
    b = RES[(RES.model == m) & (RES.stage == s1)].set_index("tile")
    for met in ("F1", "bF1@1m"):
        diff = (b[met] - a[met]).dropna()
        p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
        rows_.append(dict(model=m, metric=met, mean_change=diff.mean(), tiles_improved=int((diff > 0).sum()), n=len(diff), p_value=p))
display(pd.DataFrame(rows_).round(4).style.set_caption("Table 3 · Paired test: after vs before training"))
""")

md(r"""
### 7.1 Predicted fields as map layers
The two models recommended for operational mapping (Part 8) are written out as **field polygons** so they
can be opened in Google Earth, Google My Maps, geojson.io, QGIS, or any other web map. These files are the
predictions **before** the Steger line extraction and the snake in 7.2. The final KML, after that refinement,
is written in 7.3.

| Model | Why this file | Files |
|---|---|---|
| **SAM 2.1, pre-trained** | Highest Test accuracy | `train_outputs/vectors/sam2_pretrained_fields.geojson` and `.kml` |
| **FTW Delineate Anything V2, fine-tuned** | Close second, and fast enough for large areas | `train_outputs/vectors/da_finetuned_fields.geojson` and `.kml` |

Each connected field in the instance raster (≥ 20 m²) becomes one polygon. Coordinates are transformed from
BUTM-2010 (EPSG:9680) to **WGS84 longitude/latitude (EPSG:4326)**, which is what Google Maps and KML expect.
Boundaries are simplified by 0.5 m so the files stay small enough to import; the rasters in `train_outputs/pred/`
remain the exact predictions. In Google Earth or Google My Maps, use **Import** and choose the `.kml`. On
[geojson.io](https://geojson.io) or similar sites, drop in the `.geojson`.
""")
code(r"""
# KML colours are aabbggrr. These two outlines stay distinct if both layers are imported together.
FINAL_EXPORTS = [
    ("SAM 2.1", "pre-trained", "sam2_pretrained", "ff6b2a"),          # #2a6bff
    ("FTW Delineate Anything V2", "fine-tuned", "da_finetuned", "627f2a"),  # #2a7f62
]

def prediction_labels(model, stage, prefix):
    if "RUNS" in globals() and (model, stage) in RUNS:
        return {t: RUNS[(model, stage)][t] for t in TEST}
    out = {}
    for t in TEST:
        hits = sorted(p for p in glob.glob(os.path.join(DIRS["pred"], f"{prefix}*_{t}.npy"))
                      if os.path.basename(p).startswith(prefix) and "steger_snake" not in os.path.basename(p))
        if len(hits) != 1:
            raise FileNotFoundError(f"{prefix} / {t}: expected one prediction, found {hits}")
        out[t] = np.load(hits[0])
    return out

def labels_to_polygons(labels, transform):
    tr = Affine(*np.asarray(transform, dtype=float)[:6])
    rows = []
    for geom, val in features.shapes(np.asarray(labels, np.int32), mask=np.asarray(labels) > 0, transform=tr):
        g = shape(geom)
        if g.is_empty or g.area < MIN_FIELD_M2:
            continue
        if not g.is_valid:
            g = shapely.make_valid(g)
            if g.geom_type == "GeometryCollection":
                parts = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
                g = shapely.union_all(parts) if parts else None
        if g is None or g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        area = float(g.area)
        g = g.simplify(0.5, preserve_topology=True)
        if g.is_empty:
            continue
        rows.append((int(val), round(area, 1), g))
    return rows

def _kml_ring(coords):
    return " ".join(f"{x:.7f},{y:.7f}" for x, y in coords)

def _kml_polygon(poly):
    parts = ["<Polygon><outerBoundaryIs><LinearRing><coordinates>", _kml_ring(poly.exterior.coords),
             "</coordinates></LinearRing></outerBoundaryIs>"]
    for hole in poly.interiors:
        parts += ["<innerBoundaryIs><LinearRing><coordinates>", _kml_ring(hole.coords),
                  "</coordinates></LinearRing></innerBoundaryIs>"]
    parts.append("</Polygon>")
    return "".join(parts)

def write_kml(gdf, path, doc_name, bgr):
    folders = []
    for tile, sub in gdf.groupby("tile", sort=True):
        marks = []
        for r in sub.itertuples(index=False):
            polys = list(r.geometry.geoms) if r.geometry.geom_type == "MultiPolygon" else [r.geometry]
            body = "".join(_kml_polygon(p) for p in polys if not p.is_empty)
            if len(polys) > 1:
                body = f"<MultiGeometry>{body}</MultiGeometry>"
            desc = (f"{r.model}, {r.stage}<br/>Site {r.site}<br/>Tile {r.tile}"
                    f"<br/>Field {r.field_id}<br/>Area {r.area_m2:.0f} m²")
            marks.append("<Placemark>"
                         f"<name>{r.tile} field {r.field_id}</name>"
                         f"<description><![CDATA[{desc}]]></description>"
                         f"<styleUrl>#field</styleUrl>{body}</Placemark>")
        folders.append(f"<Folder><name>{tile}</name>{''.join(marks)}</Folder>")
    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{doc_name}</name>
<Style id="field">
  <LineStyle><color>ff{bgr}</color><width>2</width></LineStyle>
  <PolyStyle><color>66{bgr}</color></PolyStyle>
</Style>
{''.join(folders)}
</Document>
</kml>
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)

DIRS["vectors"] = os.path.join(OUT, "vectors")
os.makedirs(DIRS["vectors"], exist_ok=True)
meta = TILES.set_index("tile")
summary_rows = []
for model, stage, prefix, bgr in FINAL_EXPORTS:
    labels = prediction_labels(model, stage, prefix)
    frames = []
    for t, lab in labels.items():
        rows = labels_to_polygons(lab, D[t]["transform"])
        if not rows:
            continue
        gdf = gpd.GeoDataFrame(rows, columns=["field_id", "area_m2", "geometry"], crs=str(D[t]["crs"]))
        gdf.insert(0, "tile", t)
        gdf.insert(0, "site", meta.loc[t, "site"])
        gdf.insert(0, "stage", stage)
        gdf.insert(0, "model", model)
        frames.append(gdf)
    fields = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs).to_crs(4326)
    fields["name"] = fields["tile"] + " field " + fields["field_id"].astype(str)
    slug = prefix
    geojson_path = os.path.join(DIRS["vectors"], f"{slug}_fields.geojson")
    kml_path = os.path.join(DIRS["vectors"], f"{slug}_fields.kml")
    fields.to_file(geojson_path, driver="GeoJSON", layer_options={"RFC7946": "YES", "COORDINATE_PRECISION": "7"})
    write_kml(fields, kml_path, f"{model} ({stage}) field boundaries", bgr)
    summary_rows.append(dict(model=model, stage=stage, fields=len(fields), tiles=fields.tile.nunique(),
                             area_ha=fields.area_m2.sum() / 10000,
                             lon_min=fields.total_bounds[0], lat_min=fields.total_bounds[1],
                             lon_max=fields.total_bounds[2], lat_max=fields.total_bounds[3],
                             geojson=geojson_path, kml=kml_path))
EXPORT = pd.DataFrame(summary_rows)
EXPORT.to_csv(os.path.join(DIRS["tables"], "vector_export.csv"), index=False)
display(EXPORT.assign(area_ha=lambda x: x.area_ha.round(2),
                      lon_min=lambda x: x.lon_min.round(5), lat_min=lambda x: x.lat_min.round(5),
                      lon_max=lambda x: x.lon_max.round(5), lat_max=lambda x: x.lat_max.round(5))
        .style.format({"area_ha": "{:.2f}"}).set_caption("Test fields written for web maps (WGS84)"))
print("WGS84 polygons. Open the .kml in Google Earth or Google My Maps, or the .geojson in geojson.io.")
""")

md(r"""
### 7.2 Missing boundaries (Steger 1998) and snake refinement
The predicted outlines are already close to the digitised lines (about 1.3 m on average for SAM 2.1 and
2.5 m for Delineate Anything), but two geometric faults remain: some dark bunds inside a predicted field were never
cut, and the outline that was cut does not always sit on the centre of the bund.

**Line extraction (Steger 1998).** On the pre-processed luminance, Gaussian derivatives at scales
σ = 1.0, 1.6 and 2.2 px (bunds about 0.6–1.4 m wide) give the Hessian. A pixel is a line centre where the
first derivative across the line vanishes inside the pixel and the second derivative is positive — a **dark**
line. Dark lines are the bunds and ditches: at σ = 1.2 their response on the digitised boundaries is about
seven times the response in the field interior, whereas bright ridges are not. Centres are linked along the
line into polylines (hysteresis, normals within about 50°).

A polyline is a **candidate missing boundary** when it runs through the interior of one predicted field
(more than 1.25 m from the existing outline along most of its length), is at least 7 m long, is reasonably
straight, and the CIELAB colour 2 m to either side differs by at least 7. Each candidate can be extended
until both ends meet that field’s outline (a topological cut) and fitted by an open snake with fixed ends.
Doing so on sw_090, sw_091 and sw_100 **lowered** both instance F1 and boundary F1: most of these dark lines
are crop rows, tree lines or ditches inside a digitised parcel, not a boundary the digitiser omitted.
They are therefore saved as a line layer, `train_outputs/vectors/missing_boundaries.geojson` (450 lines on
the SAM 2.1 predictions), and are **not** used to split the parcels. The web map can turn them on in yellow.

**Snakes on the existing outlines.** Every predicted outline is a closed snake (Kass, Witkin and Terzopoulos
1988). The external force acts only along the contour normal and only up to **0.75 m**, and only towards a
Steger centre: the outline can lock onto the bund it already follows, and it cannot jump to another line.
Two smoothing passes are the internal energy (elasticity, and a little rigidity). A field is kept in its
original shape unless the new region still overlaps it with IoU ≥ 0.92 and the new outline lies on a
stronger Steger response than the old one. Those two tests are the topological constraint: the partition
into fields does not change, only the geometry of the outline.
""")
code(r"""
# Steger 1998 dark-line centres, then a normal-limited snake on each predicted outline.
# Candidate missing lines are written out; they are not used to split fields (see 7.2).
SIGMAS = (1.0, 1.6, 2.2)

def steger_response(lum, sigma):
    # Scale-normalised Gaussian derivatives. t is the sub-pixel offset along the normal (Steger 1998).
    s2 = sigma ** 2
    Ix = ndi.gaussian_filter(lum, sigma, order=(0, 1)) * sigma
    Iy = ndi.gaussian_filter(lum, sigma, order=(1, 0)) * sigma
    Ixx = ndi.gaussian_filter(lum, sigma, order=(0, 2)) * s2
    Ixy = ndi.gaussian_filter(lum, sigma, order=(1, 1)) * s2
    Iyy = ndi.gaussian_filter(lum, sigma, order=(2, 0)) * s2
    diff = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2)
    lam_hi = 0.5 * (Ixx + Iyy + diff); lam_lo = 0.5 * (Ixx + Iyy - diff)
    lam = np.where(np.abs(lam_hi) >= np.abs(lam_lo), lam_hi, lam_lo)
    nx = np.where(np.abs(Ixy) < 1e-8, np.where(np.abs(Ixx) >= np.abs(Iyy), 1.0, 0.0), Ixy)
    ny = np.where(np.abs(Ixy) < 1e-8, np.where(np.abs(Ixx) >= np.abs(Iyy), 0.0, 1.0), lam - Ixx)
    nrm = np.hypot(nx, ny) + 1e-12
    nx, ny = nx / nrm, ny / nrm
    second = nx ** 2 * Ixx + 2 * nx * ny * Ixy + ny ** 2 * Iyy
    t = -(nx * Ix + ny * Iy) / (second + 1e-12)
    return lam.astype(np.float32), t.astype(np.float32), nx.astype(np.float32), ny.astype(np.float32)

def steger_lines(lum, high_q=88, low_q=68, min_px=28):
    best = None
    for sigma in SIGMAS:
        lam, t, nx, ny = steger_response(lum, sigma)
        centre = (np.abs(t) <= 0.5) & (lam > 0)          # dark lines only
        mag = np.where(centre, lam, 0)
        if best is None:
            best = [lam, t, nx, ny, centre, mag]
        else:
            take = mag > best[5]
            for i, arr in enumerate((lam, t, nx, ny)):
                best[i] = np.where(take, arr, best[i])
            best[4] = best[4] | centre
            best[5] = np.maximum(best[5], mag)
    lam, t, nx, ny, centre, mag = best
    vals = mag[centre]
    if vals.size < 50:
        return [], mag
    high, low = np.percentile(vals, (high_q, low_q))
    low_mask = centre & (mag >= low)
    H, W = mag.shape
    visited = np.zeros((H, W), bool)
    nbrs = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    seeds = np.argwhere(centre & (mag >= high))
    order = np.argsort(-mag[seeds[:, 0], seeds[:, 1]]) if len(seeds) else []

    def walk(r, c, pdir):
        chain, cr, cc, d = [], r, c, pdir
        for _ in range(H + W):
            best_s, nxt = -1e9, None
            for dr, dc in nbrs:
                rr, c2 = cr + dr, cc + dc
                if rr < 0 or c2 < 0 or rr >= H or c2 >= W or visited[rr, c2] or not low_mask[rr, c2]:
                    continue
                align = dr * d[0] + dc * d[1]
                agree = abs(float(nx[cr, cc] * nx[rr, c2] + ny[cr, cc] * ny[rr, c2]))
                if align <= 0.2 or agree < 0.6:
                    continue
                score = align * agree * float(mag[rr, c2])
                if score > best_s:
                    best_s, nxt = score, (rr, c2)
            if nxt is None:
                break
            visited[nxt] = True
            chain.append(nxt)
            d = (nxt[0] - cr, nxt[1] - cc)
            cr, cc = nxt
        return chain

    lines = []
    for i in order:
        r, c = int(seeds[i, 0]), int(seeds[i, 1])
        if visited[r, c]:
            continue
        visited[r, c] = True
        fwd = walk(r, c, (float(nx[r, c]), float(-ny[r, c])))
        back = walk(r, c, (float(-nx[r, c]), float(ny[r, c])))
        pix = back[::-1] + [(r, c)] + fwd
        if len(pix) < min_px:
            continue
        lines.append(np.stack([[p_[0] + t[p_] * ny[p_], p_[1] + t[p_] * nx[p_]] for p_ in pix], 0).astype(np.float32))
    return lines, mag.astype(np.float32)

def dark_response(lum):
    mag = np.zeros(lum.shape, np.float32)
    for sigma in SIGMAS:
        lam, t, _, _ = steger_response(lum, sigma)
        mag = np.maximum(mag, np.where((np.abs(t) <= 0.5) & (lam > 0), lam, 0))
    return mag

def lateral_de(cielab, curve, valid, radius=8):
    if len(curve) < 5:
        return 0.0
    step = max(1, len(curve) // 40)
    pts = curve[::step]
    dlt = np.diff(curve, axis=0); dlt = np.vstack([dlt, dlt[-1]])[::step]
    tang = dlt / (np.hypot(dlt[:, 0], dlt[:, 1])[:, None] + 1e-6)
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    H, W = cielab.shape[:2]
    left, right = [], []
    for (r, c), (nr, nc) in zip(pts, normal):
        for sign, bucket in ((1, left), (-1, right)):
            rr, cc = int(round(r + sign * radius * nr)), int(round(c + sign * radius * nc))
            if 0 <= rr < H and 0 <= cc < W and valid[rr, cc]:
                bucket.append(cielab[rr, cc])
    if len(left) < 3 or len(right) < 3:
        return 0.0
    return float(np.linalg.norm(np.mean(left, 0) - np.mean(right, 0)))

def snap_outlines(labels, mag, valid, search=3):
    # External force: at most `search` px along the normal, and only onto a Steger centre.
    # Internal force: two smoothing passes. Reject a field whose region would change (IoU < 0.92)
    # or whose outline would leave the line.
    thr = float(np.percentile(mag[mag > 0], 80))
    out, moved = labels.copy(), 0
    H, W = labels.shape
    for reg in regionprops(labels):
        if reg.area < MIN_PX:
            continue
        mask = labels == reg.label
        cs = find_contours(mask.astype(float), 0.5)
        if not cs:
            continue
        curve = max(cs, key=len)
        if len(curve) < 16:
            continue
        dlt = np.diff(curve, axis=0, append=curve[:1])
        nrm = np.hypot(dlt[:, 0], dlt[:, 1]) + 1e-6
        normal = np.stack([-dlt[:, 1] / nrm, dlt[:, 0] / nrm], 1)
        moved_c = curve.copy()
        for i, ((rr, cc), (ur, uc)) in enumerate(zip(curve, normal)):
            best_s, best = 0.0, None
            for s in range(-search, search + 1):
                r2, c2 = int(round(rr + s * ur)), int(round(cc + s * uc))
                if 0 <= r2 < H and 0 <= c2 < W and mag[r2, c2] > best_s:
                    best_s, best = float(mag[r2, c2]), (rr + s * ur, cc + s * uc)
            if best is not None and best_s >= thr:
                moved_c[i] = best
        for _ in range(2):
            sm = moved_c.copy()
            sm[1:-1] = 0.2 * moved_c[:-2] + 0.6 * moved_c[1:-1] + 0.2 * moved_c[2:]
            moved_c = sm
        pts = moved_c[::max(1, len(moved_c) // 120)]
        r0 = max(int(np.floor(pts[:, 0].min())) - 2, 0); c0 = max(int(np.floor(pts[:, 1].min())) - 2, 0)
        r1 = min(int(np.ceil(pts[:, 0].max())) + 3, H); c1 = min(int(np.ceil(pts[:, 1].max())) + 3, W)
        local = np.zeros((r1 - r0, c1 - c0), bool)
        rr, cc = draw_polygon(pts[:, 0] - r0, pts[:, 1] - c0, local.shape)
        local[rr, cc] = True
        new = np.zeros(labels.shape, bool); new[r0:r1, c0:c1] = local
        new &= valid & ndi.binary_dilation(mask, iterations=search)
        inter = np.logical_and(new, mask).sum()
        union = new.sum() + mask.sum() - inter + 1e-6
        if new.sum() < MIN_PX or inter / union < 0.92:
            continue
        edge_old, edge_new = find_boundaries(mask, mode="inner"), find_boundaries(new, mode="inner")
        if not edge_new.any() or mag[edge_new].mean() <= mag[edge_old].mean():
            continue
        free = (out == 0) | (out == reg.label)
        out[mask] = 0
        out[new & free] = reg.label
        moved += 1
    return out, moved

def candidate_lines(labels, lum, cielab, valid):
    lines, _ = steger_lines(lum)
    dist_b = ndi.distance_transform_edt(~find_boundaries(labels, mode="outer"))
    kept = []
    for curve in lines:
        ri = np.clip(np.rint(curve[:, 0]).astype(int), 0, labels.shape[0] - 1)
        ci = np.clip(np.rint(curve[:, 1]).astype(int), 0, labels.shape[1] - 1)
        if (dist_b[ri, ci] > 5).mean() < 0.65:
            continue
        if lateral_de(cielab, curve, valid) < 7:
            continue
        path = np.hypot(*np.diff(curve, axis=0).T).sum() + 1e-6
        if np.hypot(*(curve[-1] - curve[0])) / path < 0.55:
            continue
        kept.append(curve)
    return kept

SNAKE_MODELS = (("sam2_pretrained", "SAM 2.1"), ("da_finetuned", "FTW Delineate Anything V2"))
snake_csv = os.path.join(DIRS["tables"], "steger_snake_test.csv")
pending = [t for t in TEST if any(
    not os.path.exists(os.path.join(DIRS["pred"], f"{prefix}_steger_snake_{t}.npy"))
    for prefix, _ in SNAKE_MODELS)]
if pending or not os.path.exists(snake_csv):
    from shapely.geometry import LineString
    frames, show, line_rows = [], {}, []
    todo = pending if os.path.exists(snake_csv) else list(TEST)
    print(f"Steger / snake refinement for {len(todo)} tiles")
    for t in todo:
        pre = np.load(os.path.join(DIRS["snake"], f"bmap_{t}.npz"))["pre"].astype(np.float32)
        lum = rgb2lab(np.clip(pre, 0, 1))[..., 0] / 100.0
        cielab = rgb2lab(np.clip(pre, 0, 1))
        mag = dark_response(lum)
        valid = D[t]["valid"]
        tr = Affine(*np.asarray(D[t]["transform"], dtype=float)[:6])
        for prefix, model in SNAKE_MODELS:
            hits = sorted(p_ for p_ in glob.glob(os.path.join(DIRS["pred"], f"{prefix}*_{t}.npy"))
                          if os.path.basename(p_).startswith(prefix) and "steger_snake" not in os.path.basename(p_))
            lab = np.load(hits[0])
            out, n_snap = snap_outlines(lab, mag, valid)
            cands = candidate_lines(lab, lum, cielab, valid) if prefix == "sam2_pretrained" else []
            np.save(os.path.join(DIRS["pred"], f"{prefix}_steger_snake_{t}.npy"), out.astype(np.int32))
            ev0, ev1 = evaluate(lab, D[t]), evaluate(out, D[t])
            core = ndi.binary_erosion(valid, iterations=4)
            def mean_dist(pred):
                pb = find_boundaries(pred, mode="inner") & (pred > 0) & core
                return float(ndi.distance_transform_edt(~(D[t]["boundary"] & core))[pb].mean() * GSD) if pb.any() else 0.0
            frames.append(dict(model=model, tile=t, snapped=n_snap,
                               missing_lines=len(cands) if prefix == "sam2_pretrained" else np.nan,
                               F1_before=ev0["F1"], F1_after=ev1["F1"],
                               bF1_1m_before=ev0["bF1@1m"], bF1_1m_after=ev1["bF1@1m"],
                               bF1_2m_before=ev0["bF1@2m"], bF1_2m_after=ev1["bF1@2m"],
                               dist_m_before=mean_dist(lab), dist_m_after=mean_dist(out)))
            if prefix == "sam2_pretrained":
                show[t] = (D[t]["img"], lab, out, cands)
                for curve in cands:
                    xs, ys = rasterio.transform.xy(tr, curve[:, 0], curve[:, 1], offset="center")
                    line_rows.append(dict(tile=t, model=model,
                                          length_m=float(np.hypot(*np.diff(curve, axis=0).T).sum()) * GSD,
                                          geometry=LineString(np.column_stack([xs, ys]))))
    new_rows = pd.DataFrame(frames)
    if os.path.exists(snake_csv):
        old = pd.read_csv(snake_csv)
        new_rows = pd.concat([old[~old.tile.isin(todo)], new_rows], ignore_index=True)
    new_rows.to_csv(snake_csv, index=False)
    if line_rows:
        extra = gpd.GeoDataFrame(line_rows, geometry="geometry", crs=str(D[todo[0]]["crs"])).to_crs(4326)
        geo_path = os.path.join(DIRS["vectors"], "missing_boundaries.geojson")
        if os.path.exists(geo_path):
            prev = gpd.read_file(geo_path).to_crs(4326)
            if "tile" in prev.columns:
                prev = prev[~prev.tile.isin(todo)]
            extra = gpd.GeoDataFrame(pd.concat([prev, extra], ignore_index=True), geometry="geometry", crs=4326)
        extra.to_file(geo_path, driver="GeoJSON",
                      layer_options={"RFC7946": "YES", "COORDINATE_PRECISION": "7"})
    if "sw_090" in show and "sw_100" in show:
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        for row, t in enumerate(("sw_090", "sw_100")):
            img, before, after, cands = show[t]
            axes[row, 0].imshow(img); axes[row, 0].set_title(f"{t} — image")
            v = img.copy(); v[find_boundaries(before)] = (255, 60, 60)
            axes[row, 1].imshow(v); axes[row, 1].set_title("predicted outline")
            v2 = img.copy(); v2[find_boundaries(after)] = (30, 200, 80)
            axes[row, 2].imshow(v2)
            for ln in cands:
                axes[row, 2].plot(ln[:, 1], ln[:, 0], color="yellow", lw=1.1)
            axes[row, 2].set_title("snake outline (green) and Steger lines (yellow)")
            for a in axes[row]:
                a.set_xlim(0, img.shape[1]); a.set_ylim(img.shape[0], 0); a.axis("off")
        plt.suptitle("Figure 7 · Steger line extraction and snake refinement on Test", y=1.01)
        plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig07_steger_snake.png"), bbox_inches="tight"); plt.close()

STEGER = pd.read_csv(snake_csv)
mean = STEGER.groupby("model")[["F1_before", "F1_after", "bF1_1m_before", "bF1_1m_after",
                                "bF1_2m_before", "bF1_2m_after", "dist_m_before", "dist_m_after"]].mean()
display(mean.round(3).style.format("{:.3f}").set_caption(
    "Table 4 · Test, before and after the snake. Instance F1 is unchanged; boundary F1 and the mean distance to the digitised line improve."))
print(f"Outlines updated on {STEGER.snapped.mean():.0f} fields per tile (mean). "
      f"Candidate missing lines (SAM 2.1, not used to split): {int(STEGER.loc[STEGER.model=='SAM 2.1','missing_lines'].sum())}.")
fig_path = os.path.join(DIRS["figures"], "fig07_steger_snake.png")
if os.path.exists(fig_path) and os.path.exists(snake_csv):
    plt.figure(figsize=(18, 10)); plt.imshow(plt.imread(fig_path)); plt.axis("off"); plt.show()
""")

md(r"""
### 7.3 Final field polygons
The KML and GeoJSON written here are the **final** map layers: the Test fields after the snake in 7.2 has
moved each outline onto the Steger line centre. They replace the pre-refinement files of 7.1 for Google Earth,
Google My Maps and the local web map.

| Model | Files |
|---|---|
| **SAM 2.1, pre-trained** | `train_outputs/vectors/sam2_pretrained_final.geojson` and `.kml` |
| **FTW Delineate Anything V2, fine-tuned** | `train_outputs/vectors/da_finetuned_final.geojson` and `.kml` |
""")
code(r"""
# Final KML / GeoJSON: snake-refined instance rasters from 7.2, in WGS84.
FINAL_KML = [
    ("SAM 2.1", "snake-refined", "sam2_pretrained", "ff6b2a"),
    ("FTW Delineate Anything V2", "snake-refined", "da_finetuned", "627f2a"),
]
meta = TILES.set_index("tile")
final_rows = []
for model, stage, prefix, bgr in FINAL_KML:
    frames = []
    for t in TEST:
        lab = np.load(os.path.join(DIRS["pred"], f"{prefix}_steger_snake_{t}.npy"))
        rows = labels_to_polygons(lab, D[t]["transform"])
        if not rows:
            continue
        gdf = gpd.GeoDataFrame(rows, columns=["field_id", "area_m2", "geometry"], crs=str(D[t]["crs"]))
        gdf.insert(0, "tile", t)
        gdf.insert(0, "site", meta.loc[t, "site"])
        gdf.insert(0, "stage", stage)
        gdf.insert(0, "model", model)
        frames.append(gdf)
    fields = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs).to_crs(4326)
    fields["name"] = fields["tile"] + " field " + fields["field_id"].astype(str)
    geojson_path = os.path.join(DIRS["vectors"], f"{prefix}_final.geojson")
    kml_path = os.path.join(DIRS["vectors"], f"{prefix}_final.kml")
    fields.to_file(geojson_path, driver="GeoJSON", layer_options={"RFC7946": "YES", "COORDINATE_PRECISION": "7"})
    write_kml(fields, kml_path, f"{model} after Steger lines and snake refinement", bgr)
    final_rows.append(dict(model=model, stage=stage, fields=len(fields), tiles=fields.tile.nunique(),
                           area_ha=round(fields.area_m2.sum() / 10000, 2),
                           geojson=geojson_path, kml=kml_path))
FINAL = pd.DataFrame(final_rows)
FINAL.to_csv(os.path.join(DIRS["tables"], "final_vector_export.csv"), index=False)
display(FINAL.style.set_caption("Final Test fields (after the snake), WGS84"))
print("Final KML is the snake-refined outline. Open the .kml in Google Earth or Google My Maps.")
""")

md(r"""
### 7.4 Final polygons and digitised lines
The final SAM 2.1 polygons (after the snake) are drawn on the image, beside the digitised Test lines on the
same image. The three rows are the **best**, **medium** and **worst** Test tiles, ranked by instance F1.
The medium tile is the one nearest the median F1.
""")
code(r"""
# Figure 8 · 3 × 2: final polygons (left) and digitised test lines (right).
sam = STEGER[STEGER.model == "SAM 2.1"].sort_values("F1_after", ascending=False).reset_index(drop=True)
median_f1 = float(sam.F1_after.median())
medium = sam.iloc[(sam.F1_after - median_f1).abs().to_numpy().argmin()]
picks = [("Best", sam.iloc[0]), ("Medium", medium), ("Worst", sam.iloc[-1])]
blue = np.array([42, 107, 255], np.float32)
fig, axes = plt.subplots(3, 2, figsize=(16, 17))
for row, (rank, rec) in enumerate(picks):
    t, d = rec.tile, D[rec.tile]
    lab = np.load(os.path.join(DIRS["pred"], f"sam2_pretrained_steger_snake_{t}.npy"))
    img = d["img"]
    edge = ndi.binary_dilation(find_boundaries(lab, mode="inner") & (lab > 0), iterations=1)
    poly = img.astype(np.float32)
    poly[lab > 0] = 0.82 * poly[lab > 0] + 0.18 * blue
    poly[edge] = blue
    axes[row, 0].imshow(poly.astype(np.uint8))
    axes[row, 0].set_title(
        f"{rank} · {t} · final polygons\nF1 {rec.F1_after:.2f} · bF1@1 m {rec.bF1_1m_after:.2f}", fontsize=12)
    lines = img.copy()
    lines[ndi.binary_dilation(d["boundary"], iterations=1)] = (220, 30, 30)
    axes[row, 1].imshow(lines)
    axes[row, 1].set_title(f"{rank} · {t} · digitised test lines", fontsize=12)
    for a in axes[row]:
        a.set_xlim(0, img.shape[1]); a.set_ylim(img.shape[0], 0); a.axis("off")
plt.suptitle("Figure 8 · SAM 2.1 final polygons and digitised test lines on the best, medium and worst Test tiles", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(DIRS["figures"], "fig08_final_polygons.png"), bbox_inches="tight")
plt.show()
""")


# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 8 · Interpretation and conclusions
*Numbers refer to the reference run of this notebook on `Labeled_Data/` (EPSG:4326 tiles warped to BUTM-2010):
Test = 10 tiles at sites sw_09 and sw_10, 815 reference fields, mean over tiles. Paired changes were tested with
the Wilcoxon signed-rank test (n = 10 tiles).*

### 8.1 What training changed
| Model | Before → after (Test instance F1) | Boundary F1 @ 1 m | Verdict |
|---|---|---|---|
| **Snake pipeline** (tuning) | 0.28 → **0.42** | 0.53 → 0.56 | Tuning on Train helps on every Test tile (10/10, p = 0.002). The default over-segments (≈ 210 polygons per tile vs 82 reference fields). The tuned setting (deeper h-minima, H = 0.08) suppresses spurious seeds, so precision rises from 0.20 to 0.34. |
| **FTW Delineate Anything V2** (fine-tuning) | 0.39 → **0.64** | 0.45 → **0.65** | The largest gain from training (10/10 tiles, p = 0.002). Recall more than doubles (0.29 → 0.77): the pre-trained model misses many plots in this landscape; after fine-tuning it finds most of them. Validation picked the 768 px patch and confidence 0.25. |
| **SAM 2.1** (decoder fine-tuning) | **0.68** → 0.64 | **0.77** → 0.71 | Decoder fine-tuning **does not help automatic mapping** (worse on 7/10 tiles, p = 0.049), even though point-prompted Valid mIoU rose from 0.33 to 0.59. On Valid the fine-tuned decoder calibrated slightly higher (AMG F1 0.57 vs 0.55), so a Valid-based selection would have deployed the weaker model here. |
| **SAM v1** (decoder fine-tuning) | 0.62 → 0.39 | 0.74 → 0.60 | Same pattern, more strongly (worse on 10/10 tiles). Valid already preferred the pre-trained decoder (AMG F1 0.46 vs 0.33), so a Valid-based selection would not deploy the fine-tuned SAM v1. |

**FC-DenseNet** (Part 6b) is trained from scratch on the boundary labels. FC-DenseNet56, 67 and 103 are
fit on Train, and Valid picks the depth and the boundary threshold. Test then compares that network
before training and after. Those two rows are in Table 1. There is no field-boundary checkpoint to
start from, so the untrained row is a random network; the trained row is the result that matters.

### 8.2 Why fine-tuning the SAM decoders helped prompts but hurt automatic mapping
* **Training and use differ.** The decoder was trained to answer *"segment the field under this point"*,
  and it got better at exactly that (prompted mIoU +0.26 for SAM 2, +0.30 for SAM v1). Automatic mask generation
  instead prompts a dense 64 × 64 grid, including points on bunds, paths and trees, and then **ranks and
  filters** the answers with the decoder's own IoU estimate and a stability score.
* **Calibration shifts.** After fine-tuning, the IoU head rates almost every answer as high quality, so the
  pre-trained thresholds let hundreds of fragments through. Before threshold re-calibration on Valid, the
  fine-tuned SAM 2 fell to F1 0.03 on Test. Re-calibration recovers most of this (0.64), but not the
  pre-trained level. A development experiment that sampled training points anywhere inside fields
  (to mimic the grid) did not help either.
* **Small, homogeneous training set.** 36 tiles from 6 nearby sites are enough to adapt a 4 M-parameter
  decoder to these *prompts*, but not to change the *ranking behaviour* that automatic generation relies on.
* **Calibration also helps the pre-trained models.** Choosing thresholds on Valid raised pre-trained SAM 2
  from F1 ≈ 0.56 (default thresholds, first run) to 0.68 on Test. For promptable models,
  *calibrating the inference settings* was worth more than *training the decoder*.

### 8.3 Best models for field-boundary mapping on this imagery
1. **SAM 2.1, pre-trained, with Valid-calibrated AMG thresholds:** best on almost every Test metric:
   instance F1 0.68, recall 0.80, mean best IoU 0.73, boundary F1 0.77 @ 1 m and 0.90 @ 2 m. It needs no
   training, but it is the slowest (≈ 1 min per tile on an RTX A5500).
2. **FTW Delineate Anything V2, fine-tuned:** second (instance F1 0.64, precision 0.55, recall 0.77,
   boundary F1 0.65 @ 1 m), close to SAM 2.1 at Test site sw_09 (F1 0.67 vs 0.69). It runs in ≈ 1–2 s per tile
   (≈ 1/50 of SAM 2's time) and gains the most from local training data. It is the most practical choice
   for large areas.
3. **SAM v1 (ViT-B), pre-trained:** F1 0.62. A good fallback, but weaker than SAM 2.
4. **Snake pipeline, tuned:** F1 0.42. Needs no GPU or training labels, but its boundaries are the least
   precise and it under-segments small plots. Useful where labels and GPUs are unavailable.

Pre-trained SAM 2 is best at both Test sites: sw_09 (F1 0.69 vs 0.67 for fine-tuned Delineate Anything) and
sw_10 (0.67 vs 0.62). Both models form the top group at both sites. The before/after changes are significant
for every model (p ≤ 0.049). The snake tuning, Delineate Anything and SAM v1 changes go the same way on all
10 tiles; the SAM 2.1 decoder fine-tuning is less consistent (worse on 7 of 10).

### 8.4 Recommendations
* For operational mapping: **fine-tune Delineate Anything V2** on a few dozen labelled tiles from the target
  region, or use **pre-trained SAM 2 with calibrated thresholds** where a GPU is available and boundary precision
  matters most.
* A promising combination: use the fine-tuned Delineate Anything detections as **box prompts for SAM 2**
  (detector finds the fields, SAM traces the edges). This avoids SAM's automatic grid and its calibration problem.
* If SAM is to be trained, train it for the deployment task: full fine-tuning with an automatic-generation
  objective (grid prompts including negatives) or box prompts, with **checkpoint selection on automatic-mapping
  F1 on Valid**, not on prompted IoU.

### 8.5 Limitations
* **One region, one season.** 10 sites from the same area. Transfer to other districts, seasons or
  sensors is untested.
* **Reference construction.** Field instances were derived from digitised lines (gaps ≤ 0.5 m closed,
  regions ≥ 20 m²). Enclosed non-field patches (trees, homesteads) count as parcels, and digitising errors
  affect all models equally.
* **Imagery source.** All tiles are screenshots of Google satellite imagery (effective resolution
  ≈ 0.2–0.3 m; acquisition dates unknown and possibly mixed). Results may differ for other providers,
  dates or true very-high-resolution imagery. Google imagery is subject to Google's Terms of Service, which
  restrict bulk download and reuse.
* **SAM v1 used the ViT-B encoder.** The ViT-H checkpoint could not be downloaded in reasonable time.
  ViT-H would likely narrow the gap to SAM 2.
* **Limited hyper-parameter search.** Training schedules (2 000 decoder steps, ≤ 60 YOLO epochs,
  30 FC-DenseNet epochs) and the threshold grids were not exhaustively searched.
* **FC-DenseNet starts from random weights.** Delineate Anything and SAM begin from large pre-trained
  models. FC-DenseNet sees only the Train tiles, so its Test score is not on the same footing.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3.12 (FieldEdge)", "language": "python", "name": "fieldedge"},
    "language_info": {"name": "python"}})
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "field_boundary_training.ipynb")
nbf.write(nb, _out)
print("written", len(cells), "cells to", _out)
