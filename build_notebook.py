"""Builds field_boundary_comparison.ipynb

Compares FTW tools (Delineate Anything V2, PRUE on Sentinel-2, FTW Global Map), SAM 2 and an
active-contour (snake) pipeline on Esri and Google screenshots captured at zoom 17, 18 and 19
over the same ground footprint at five Bangladesh sites.

Run:  .venv/bin/python build_notebook.py
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ════════════════════════════════════════════════════════════════════════════
md(r"""
# Agricultural Field Boundary Delineation in Bangladesh
## Fields of The World (FTW) tools vs SAM 2 vs Active Contours, on Esri and Google imagery at three zoom levels
---
**Author:** Zia U. Ahmed, Ph.D. · Uppata Analytics LLC

### Objectives
1. **Capture** satellite "screenshots" of five smallholder farming sites from two free basemaps (Esri World
   Imagery and Google Maps) at **three zoom levels** (z17 ≈ 1.1 m, z18 ≈ 0.55 m, z19 ≈ 0.28 m per pixel)
   over the **same ground footprint**, and find which zoom shows field boundaries best.
2. **Delineate field boundaries** with three families of methods:
   * **FTW tools** (Microsoft AI for Good Lab, Taylor Geospatial and partners):
     *Delineate Anything* (v1 and V2) on the VHR screenshots, the *PRUE* U-Net on Sentinel-2, and the published
     *FTW Global Map*;
   * **SAM 2**, a foundation segmentation model, alone and combined with an unsupervised field mask;
   * a classical **active-contour (snake) pipeline**: image enhancement → unsupervised field mask →
     watershed → snakes.
3. **Compare** the methods across imagery source and zoom: field-detection accuracy against
   independent reference points, polygon statistics, edge alignment, and consistency across zooms and sources.
4. **Interpret** the results and recommend a workflow for smallholder field mapping in Bangladesh.

### Study design at a glance
| Factor | Levels |
|---|---|
| Sites | Bhola · Faridpur · Patuakhali · Rajbari · Barisal (south-central Bangladesh) |
| Imagery source | Esri World Imagery · Google Maps satellite (RGB, Web-Mercator tiles) |
| Zoom (same 565 m × 565 m footprint) | z17 = 512 px · z18 = 1024 px · z19 = 2048 px |
| Methods | Snake pipeline · SAM 2 · SAM 2 + field mask · FTW Delineate Anything v1 & V2 · FTW PRUE-B7 (Sentinel-2, 10 m) · FTW Global Map 2024/2025 |
| Reference | 141 independent random points, visually interpreted (field / non-field) |

### Notebook map
| Part | Content |
|---|---|
| 0 | Environment and configuration |
| 1 | Imagery capture at three zooms, quality checks, best-zoom selection |
| 2 | Reference data |
| 3 | Method A: active-contour (snake) pipeline |
| 4 | Method B: SAM 2 and SAM 2 + field mask |
| 5 | Method C: FTW Delineate Anything v1 / V2 (`ftw_tools`) |
| 6 | Method D: FTW PRUE-B7 on Sentinel-2 (`ftw inference` CLI) and FTW Global Map |
| 7 | Results: maps, accuracy, effect of zoom and source, agreement |
| 8 | Interpretation, conclusions, limitations |
""")

md(r"""
---
# Part 0 · Environment and configuration
This notebook runs in a dedicated virtual environment (the repository `.venv`, Python 3.12) because
`ftw-tools` requires Python ≥ 3.12 and its own PyTorch / torchvision versions.

```bash
uv venv -p 3.12 .venv
uv pip install --prerelease=allow "ftw-tools[delineate-anything]==2.0.0b5" "sam2>=1.1" duckdb \
    huggingface_hub opencv-python-headless scikit-learn ipykernel nbconvert
```
`ftw-tools 2.0.0b5` is the release that ships the **PRUE** models used for the FTW Global Map
(EfficientNet-B3/B5/B7 U-Nets) and **Delineate Anything V2**. All outputs are written under
`outputs/` and cached, so re-running the notebook skips completed steps.
""")
code(r"""
import io, json, math, os, subprocess, sys, time, warnings
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import find_peaks

import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, box, shape
from shapely.strtree import STRtree

from skimage import graph
from skimage.color import rgb2lab
from skimage.draw import polygon as draw_polygon
from skimage.filters import gaussian, sato
from skimage.measure import find_contours, regionprops
from skimage.morphology import h_minima, disk, opening, remove_small_objects, remove_small_holes
from skimage.segmentation import slic, watershed, find_boundaries, active_contour
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score

import torch

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 11})

print(f"Python {sys.version.split()[0]}  ·  torch {torch.__version__} (CUDA {torch.version.cuda}, "
      f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'})")
for pkg in ["ftw-tools", "sam2", "ultralytics", "torchgeo", "scikit-image", "geopandas", "rasterio", "duckdb"]:
    print(f"  {pkg:<13} {version(pkg)}")
""")
code(r"""
# ── Study configuration ─────────────────────────────────────────────────────
SITES = {
    "Bhola"     : (22.315167, 90.154757),
    "Faridpur"  : (23.482984, 89.764569),
    "Patuakhali": (22.313024, 90.330621),
    "Rajbari"   : (23.608727, 89.671955),
    "Barisal"   : (22.613982, 90.396066),
}
SOURCES = ["Esri", "Google"]
ZOOMS = [17, 18, 19]
REF_ZOOM, REF_SIZE = 18, 1024          # footprint = the 1024-px window at zoom 18 (≈ 565 m)
UTM = "EPSG:32646"                     # metric CRS (UTM 46N) for areas and lengths
TILE_URLS = {
    "Esri"  : "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT = "outputs"
DIRS = {k: os.path.join(OUT, k) for k in
        ("images", "ftw_input", "ftw_s2", "vectors", "rasters", "figures", "tables")}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

def size_at(z):            # pixels needed at zoom z to cover the reference footprint
    return int(REF_SIZE * 2 ** (z - REF_ZOOM))

def scale_at(z):           # pixel-parameter scale (parameters were tuned at zoom 18)
    return float(max(1, 2 ** (z - REF_ZOOM)))

def write_raster(path, arr, transform, crs="EPSG:3857", nodata=None):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                       dtype=arr.dtype, crs=crs, transform=transform, compress="deflate", nodata=nodata) as dst:
        dst.write(arr)

print({z: f"{size_at(z)} px" for z in ZOOMS})
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 1 · Imagery: screenshots at three zoom levels
### 1.1 Capture on a fixed footprint
Web basemaps are pyramids of 256 × 256 px tiles in the Web-Mercator projection (EPSG:3857); each zoom
level halves the pixel size. For every site the **footprint is fixed**: the 1024 × 1024 px window at
zoom 18 centred on the site (≈ 565 m × 565 m). The same footprint is captured at

| Zoom | Pixels | Nominal pixel size at 22–24 °N |
|---|---|---|
| 17 | 512 × 512 | ≈ 1.10 m |
| 18 | 1024 × 1024 | ≈ 0.55 m |
| 19 | 2048 × 2048 | ≈ 0.28 m |

so differences between zooms come only from **resolution and from which scene the provider serves at
that zoom**, not from a changing area. Tiles are stitched, cropped to the exact window, and written as
GeoTIFF (EPSG:3857).

> Google's Terms of Service restrict bulk download and reuse of map tiles. Google imagery is used here
> for research comparison only. Esri World Imagery is the more permissive default.
""")
code(r"""
TILE = 256
ORIGIN = math.pi * 6378137.0

def lonlat_to_global_px(lon, lat, z):
    n = TILE * 2 ** z
    s = math.sin(math.radians(lat))
    return (lon + 180.0) / 360.0 * n, (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n

def global_px_to_3857(px, py, z):
    res = 2 * ORIGIN / (TILE * 2 ** z)
    return px * res - ORIGIN, ORIGIN - py * res

def fetch_tile(url, retries=4):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (field-boundary research notebook)"})
    for k in range(retries):
        try:
            with urlopen(req, timeout=20) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            time.sleep(0.5 * (k + 1))
    return Image.new("RGB", (TILE, TILE), (0, 0, 0))

def capture(lat, lon, z, source):
    # Fixed footprint: centre the zoom-18 window, then scale its pixel origin to zoom z
    cx18, cy18 = lonlat_to_global_px(lon, lat, REF_ZOOM)
    f = 2 ** (z - REF_ZOOM); size = size_at(z)
    x0 = int(round(cx18 - REF_SIZE / 2) * f); y0 = int(round(cy18 - REF_SIZE / 2) * f)
    tx0, ty0, tx1, ty1 = x0 // TILE, y0 // TILE, (x0 + size - 1) // TILE, (y0 + size - 1) // TILE
    jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        tiles = list(ex.map(lambda t: fetch_tile(TILE_URLS[source].format(x=t[0], y=t[1], z=z)), jobs))
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    for (tx, ty), im in zip(jobs, tiles):
        mosaic.paste(im, ((tx - tx0) * TILE, (ty - ty0) * TILE))
    ox, oy = x0 - tx0 * TILE, y0 - ty0 * TILE
    arr = np.asarray(mosaic.crop((ox, oy, ox + size, oy + size)))
    west, north = global_px_to_3857(x0, y0, z)
    east, south = global_px_to_3857(x0 + size, y0 + size, z)
    return arr, from_bounds(west, south, east, north, size, size)

IMG, TR = {}, {}
for name, (lat, lon) in SITES.items():
    for s in SOURCES:
        for z in ZOOMS:
            tif = os.path.join(DIRS["images"], f"{s}_{name}_z{z}.tif")
            if not os.path.exists(tif):
                arr, tr = capture(lat, lon, z, s)
                write_raster(tif, arr.transpose(2, 0, 1).copy(), tr)
            with rasterio.open(tif) as src:
                IMG[(s, name, z)], TR[(s, name, z)] = src.read().transpose(1, 2, 0), src.transform

# Footprint polygon per site (EPSG:3857) and the common evaluation grid (zoom 18)
AOI ={n: box(TR[("Esri", n, 18)].c, TR[("Esri", n, 18)].f - REF_SIZE * TR[("Esri", n, 18)].a,
              TR[("Esri", n, 18)].c + REF_SIZE * TR[("Esri", n, 18)].a, TR[("Esri", n, 18)].f) for n in SITES}
for n in SITES:
    w = (AOI[n].bounds[2] - AOI[n].bounds[0]) * math.cos(math.radians(SITES[n][0]))
    print(f"{n:<11} footprint ≈ {w:.0f} m × {w:.0f} m   images: " +
          ", ".join(f"{s} z{z} {IMG[(s, n, z)].shape[0]}px" for s in SOURCES for z in ZOOMS))
""")

md(r"""
### 1.2 Image-quality checks and boundary visibility
Three diagnostics for every source × site × zoom:

* **Placeholder fraction:** share of 32 × 32 px blocks that are flat (std < 1.5). Providers serve grey
  *"Map data not yet available"* tiles where no imagery exists at that zoom; these images are 85–90 % flat.
  Images with > 50 % flat blocks are **invalid** and excluded. A lower threshold would wrongly reject scenes
  with large uniform crop blocks.
* **Genuine detail (not upsampled):** fine-scale energy (above half the Nyquist frequency, relative to
  total variance) of the z19 image, divided by that of the **z18 image upsampled 2× (cubic)**. That
  upsampled image is exactly what a provider serves when it only stretches z18. A ratio < 3 flags
  **upsampled** imagery; genuine z19 imagery here has ratios of 12–36.
* **Boundary density:** share of pixels on field-scale edge chains (Canny on the bilateral-filtered
  L\* channel, fixed thresholds, chains ≥ 30 px × scale). This measures how visible field boundaries
  are, and drives the **dynamic best-zoom choice**: the highest zoom that is valid, not upsampled,
  and has boundary density ≥ 0.02.
""")
code(r"""
def flat_fraction(rgb, block=32):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = (g.shape[0] // block) * block, (g.shape[1] // block) * block
    return (g[:h, :w].reshape(h // block, block, w // block, block).std(axis=(1, 3)) < 1.5).mean()

def hf_energy(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                       g.shape[::-1], interpolation=cv2.INTER_CUBIC)
    return float(np.mean((g - small) ** 2) / (g.var() + 1e-6))

def boundary_density(rgb, scale=1.0):
    L = cv2.bilateralFilter(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0], 9, 25, 5)
    edges = cv2.Canny(L, 15, 35, L2gradient=True) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(edges.astype(np.uint8), connectivity=8)
    return float(np.isin(lab, np.where(st[:, cv2.CC_STAT_AREA] >= 30 * scale)[0][1:]).mean())

rows = []
for (s, n, z), rgb in IMG.items():
    rows.append({"source": s, "site": n, "zoom": z, "flat_frac": flat_fraction(rgb),
                 "hf_energy": hf_energy(rgb), "boundary_density": boundary_density(rgb, scale_at(z))})
Q = pd.DataFrame(rows)
Q["valid"] = Q.flat_frac < 0.5
def hf_of_upsampled_z18(s, n):
    up = cv2.resize(IMG[(s, n, 18)], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return hf_energy(up)
Q["detail_ratio_z19"] = [r.hf_energy / hf_of_upsampled_z18(r.source, r.site) if r.zoom == 19 else np.nan
                         for r in Q.itertuples()]
Q["upsampled"] = (Q.zoom == 19) & Q.valid & (Q.detail_ratio_z19 < 3)
Q["usable"] = Q.valid & ~Q.upsampled

BEST = {}
for (s, n), g in Q.groupby(["source", "site"]):
    ok = g[g.usable & (g.boundary_density >= 0.02)]
    BEST[(s, n)] = int(ok.zoom.max()) if len(ok) else int(g[g.usable].zoom.max())
Q["best"] = [BEST[(r.source, r.site)] == r.zoom for r in Q.itertuples()]
Q.to_csv(os.path.join(DIRS["tables"], "image_quality.csv"), index=False)

show = Q.pivot_table(index=["source", "site"], columns="zoom",
                     values=["boundary_density", "flat_frac", "detail_ratio_z19"]).round(3)
display(show)
RUN_IMAGES = [(r.source, r.site, r.zoom) for r in Q[Q.usable].itertuples()]
print(f"Usable images: {len(RUN_IMAGES)} of {len(Q)}")
print("Excluded:", [(r.source, r.site, r.zoom, "placeholder" if not r.valid else "upsampled")
                    for r in Q[~Q.usable].itertuples()])
print("Best zoom (dynamic rule):", {f"{s}/{n}": z for (s, n), z in BEST.items()})
""")
code(r"""
fig, axes = plt.subplots(len(SITES), 6, figsize=(24, 4.2 * len(SITES)))
for i, n in enumerate(SITES):
    for j, (s, z) in enumerate([(s, z) for s in SOURCES for z in ZOOMS]):
        ax = axes[i, j]; ax.imshow(IMG[(s, n, z)]); ax.set_xticks([]); ax.set_yticks([])
        r = Q[(Q.source == s) & (Q.site == n) & (Q.zoom == z)].iloc[0]
        tag = "PLACEHOLDER" if not r.valid else ("UPSAMPLED" if r.upsampled else f"bd {r.boundary_density:.3f}")
        ax.set_title(f"{n} · {s} z{z} · {tag}", fontsize=9)
        if BEST[(s, n)] == z:
            for sp in ax.spines.values(): sp.set_edgecolor("#e6194b"); sp.set_linewidth(4)
plt.suptitle("Figure 1 · Same footprint at zoom 17 / 18 / 19 from Esri and Google (red frame = best zoom)", y=1.0)
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig01_imagery.png"), bbox_inches="tight"); plt.show()
""")
md(r"""
**Reading Figure 1 and the table.** Look at three things: (i) which zoom levels actually hold imagery
(placeholders are excluded), (ii) whether z19 adds real detail (detail ratio ≫ 3) or is an enlarged z18
scene, and (iii) that Esri and Google often show **different acquisition dates**:
different crop stages, flooding, or dry-season bare soil. This changes boundary visibility as much as
resolution does.
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 2 · Reference data
There are no surveyed field boundaries for these sites. Accuracy is therefore assessed as
**field-detection accuracy** at independent random points:

* **146 simple-random points**, about 29 per site, drawn uniformly inside the footprints *independently of every
  method* (`outputs/random_reference_points.csv`, copied from the project root).
* Each point was labelled by **visual interpretation** of the Esri z18 image: crop or fallow/bare →
  **field**; trees/homestead, water, road/path → **non-field**; 5 ambiguous points (on a boundary) are excluded.
* A method "detects" a point as field if the point falls inside one of its field polygons.

Reported per run: **field recall** (share of field points inside a polygon), **false-field rate** (share
of non-field points inside a polygon), **overall accuracy**, and **Cohen's κ**.
""")
code(r"""
src_ref = "random_reference_points.csv" if os.path.exists("random_reference_points.csv") else "../outputs/random_reference_points.csv"
REF = pd.read_csv(src_ref).dropna(subset=["label"])
REF["label"] = REF.label.astype(int)
REF = gpd.GeoDataFrame(REF, geometry=gpd.points_from_xy(REF.x, REF.y), crs=UTM).to_crs(3857)
REF = REF[[AOI[s].contains(p) for s, p in zip(REF.site, REF.geometry)]].reset_index(drop=True)
REF["is_field"] = (REF.label <= 1).astype(int)
REF[["pid", "site", "x", "y", "label"]].to_csv(os.path.join(DIRS["tables"], "reference_points_used.csv"), index=False)
CLASS = {0: "crop", 1: "fallow / bare", 2: "water", 3: "trees / homestead", 4: "road / path"}
display(pd.crosstab(REF.site, REF.label.map(CLASS), margins=True))

fig, axes = plt.subplots(1, 5, figsize=(25, 5.4))
for ax, n in zip(axes, SITES):
    b = AOI[n].bounds; ax.imshow(IMG[("Esri", n, 18)], extent=(b[0], b[2], b[1], b[3]))
    r = REF[REF.site == n]
    ax.scatter(r.geometry.x, r.geometry.y, c=np.where(r.is_field, "#ffe119", "#e6194b"), s=45, edgecolor="k")
    ax.set_title(f"{n}: {len(r)} points"); ax.axis("off")
fig.legend(handles=[mpatches.Patch(color="#ffe119", label="field"), mpatches.Patch(color="#e6194b", label="non-field")],
           loc="lower center", ncol=2, frameon=False)
plt.suptitle("Figure 2 · Independent random reference points (Esri z18)", y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig02_reference.png"), bbox_inches="tight"); plt.show()
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 3 · Method A: active-contour (snake) pipeline
A classical, fully **unsupervised** pipeline adapted from Wagner & Oppelt (2020, *Remote Sensing* 12:1205).
It runs on every usable image. All pixel-based parameters were tuned at zoom 18 and scale with
`scale = max(1, 2^(zoom−18))`, so smoothing widths, windows and minimum areas keep the same ground
size at zoom 19.

**A1 · Pre-processing.** Gray-world white balance → 1–99 % percentile stretch → two passes of an
edge-preserving **bilateral filter** (σ_colour 25, σ_space 5) → **sigmoid contrast** on luminance (YUV),
centred between the two main histogram peaks (dark crop vs bright soil), gain 5 → **CLAHE**, which restores
local contrast inside large uniform blocks.

**A2 · Augmented boundary-probability map.** Six augmented copies (gamma 0.8 / 1.0 / 1.25 × Gaussian
σ 1 / 2 px) → per copy, **Sobel gradient summed over the 3 bands** (0.6) + **Sato ridge filter** for thin
dark and bright lines such as bunds and ditches (0.4) → average. A boundary scores high only if it is
consistent across brightness and scale.

**A3 · Unsupervised field / non-field mask.** SLIC superpixels → per superpixel *medians* of L\*a\*b\*,
ExG and local texture (3 px, 7 px windows) → **K-means (k = 8)** fitted on all sites of the same source
and zoom → clusters whose median texture is > 1 SD above the mean are labelled **trees / homestead**,
the rest **field** → morphological clean-up.

**A4 · Boundaries inside the field mask.** The boundary map is rescaled within fields → h-minima
seeds → marker-controlled **watershed** → regions merged weakest-border-first only if the border is
weak (< 0.30) **and** their CIELAB colours are similar (ΔE < 8) → each parcel outline is refined by
a parametric **active contour** (Kass et al. 1988; α = 0.0005, β = 0.001, image energy pulls the
contour onto boundary ridges) → polygons.
""")
code(r"""
# ── A1 · pre-processing ─────────────────────────────────────────────────────
def gray_world(rgb):
    f = rgb.astype(np.float32); m = f.reshape(-1, 3).mean(0)
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

# ── A2 · augmented boundary map ────────────────────────────────────────────
def _n99(x):
    return np.clip(x / (np.percentile(x, 99) + 1e-6), 0, 1)

def boundary_probability(pre, scale=1.0):
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

# ── A3 · superpixel features for the unsupervised mask ─────────────────────
odd = lambda x: int(round(x)) // 2 * 2 + 1
def local_std(x, k):
    return np.sqrt(np.maximum(ndi.uniform_filter(x**2, k) - ndi.uniform_filter(x, k)**2, 0))

def superpixel_features(pre, scale):
    lab = rgb2lab(pre); L = lab[..., 0]
    R, G, B = pre[..., 0], pre[..., 1], pre[..., 2]
    sp = slic(pre, n_segments=int(2500 * (pre.shape[0] / 1024) ** 2 / scale**2), compactness=12,
              start_label=1, channel_axis=-1)
    idx = np.unique(sp); f = pd.DataFrame({"sp": idx})
    for k, v in {"L": L, "a": lab[..., 1], "b": lab[..., 2], "exg": (2 * G - R - B) / (R + G + B + 1e-6),
                 "tex3": local_std(L, odd(3 * scale)), "tex7": local_std(L, odd(7 * scale))}.items():
        f[k] = ndi.median(v, sp, idx)
    return sp, f

# ── A4 · watershed → merge → snakes ────────────────────────────────────────
def merge_regions(labels, Bs, lab, t_edge=0.30, t_de=8.0):
    rag = graph.rag_boundary(labels, Bs); n = labels.max() + 1; idx = np.arange(n)
    S = np.stack([ndi.sum(lab[..., k], labels, idx) for k in range(3)], 1); C = ndi.sum(np.ones(labels.shape), labels, idx)
    parent = np.arange(n)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for w, u, v in sorted((d["weight"], u, v) for u, v, d in rag.edges(data=True) if 0 not in (u, v)):
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

def snake_parcels(B, field, pre, scale):
    Bf = np.clip(B / (np.percentile(B[field], 99) + 1e-6), 0, 1)
    Bs = gaussian(Bf, 1.5 * scale)
    markers, _ = ndi.label(h_minima(Bs, 0.04) & field)
    merged = merge_regions(watershed(Bs, markers, mask=field), Bs, rgb2lab(pre))
    Bsn = gaussian(Bf, 2.0 * scale); Bsn = Bsn / (Bsn.max() + 1e-6)
    out = np.zeros_like(merged); pad, step = int(12 * scale), int(2 * scale)
    for r in sorted(regionprops(merged), key=lambda r: r.area, reverse=True):
        if r.area < 300 * scale**2:
            continue
        r0, c0, r1, c1 = r.bbox
        r0, c0, r1, c1 = max(r0 - pad, 0), max(c0 - pad, 0), min(r1 + pad, B.shape[0]), min(c1 + pad, B.shape[1])
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

def labels_to_polygons(labels, transform, min_m2=150, simplify_m=1.0):
    geoms = [(shape(g), int(v)) for g, v in features.shapes(labels.astype(np.int32), mask=labels > 0,
                                                             transform=transform, connectivity=4)]
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    g = gpd.GeoDataFrame({"pid": [v for _, v in geoms]}, geometry=[x for x, _ in geoms], crs="EPSG:3857").to_crs(UTM)
    g = g.dissolve("pid").explode(index_parts=False).reset_index()
    g["geometry"] = g.geometry.buffer(0).simplify(simplify_m, preserve_topology=True)
    return g[g.area >= min_m2].to_crs(3857)
""")
code(r"""
snake_cache = os.path.join(DIRS["vectors"], "snake.gpkg")
FIELD_MASK = {}          # (source, site, zoom) → boolean field mask on that image's grid
mask_dir = os.path.join(DIRS["rasters"], "fieldmask"); os.makedirs(mask_dir, exist_ok=True)
CLUSTER_TABLES = {}

if os.path.exists(snake_cache):
    snake = gpd.read_file(snake_cache)
    for key in RUN_IMAGES:
        with rasterio.open(os.path.join(mask_dir, "{}_{}_z{}.tif".format(*key))) as src:
            FIELD_MASK[key] = src.read(1).astype(bool)
else:
    parts, t_all = [], time.time()
    for (s, z), keys in pd.Series(RUN_IMAGES).groupby(lambda i: (RUN_IMAGES[i][0], RUN_IMAGES[i][2])):
        keys = list(keys); sc = scale_at(z)
        pre, bmap, sps, feats = {}, {}, {}, []
        for key in keys:
            pre[key] = preprocess(IMG[key])
            bmap[key] = boundary_probability(pre[key], sc)
            sps[key], f = superpixel_features(pre[key], sc); f["key"] = [key] * len(f); feats.append(f)
        F = pd.concat(feats, ignore_index=True); cols = ["L", "a", "b", "exg", "tex3", "tex7"]
        F["cl"] = KMeans(8, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(F[cols]))
        cz = (F.groupby("cl").tex7.median() - F.tex7.mean()) / F.tex7.std()
        rough = cz.index[cz > 1.0].tolist()
        CLUSTER_TABLES[(s, z)] = F.groupby("cl")[cols].median().assign(tex7_z=cz, assigned=lambda d: np.where(d.index.isin(rough), "trees/homestead", "field"))
        for key in keys:
            sub = F[F.key.apply(lambda k: k == key)]
            lut = np.zeros(sps[key].max() + 1, bool); lut[sub.sp] = ~sub.cl.isin(rough)
            m = opening(lut[sps[key]], disk(int(round(3 * sc))))
            m = remove_small_holes(remove_small_objects(m, max_size=int(800 * sc**2)), max_size=int(600 * sc**2))
            FIELD_MASK[key] = m
            write_raster(os.path.join(mask_dir, "{}_{}_z{}.tif".format(*key)), m.astype(np.uint8), TR[key])
            parcels = snake_parcels(bmap[key], m, pre[key], sc)
            g = labels_to_polygons(parcels, TR[key]); g["source"], g["site"], g["zoom"] = key
            parts.append(g)
            print(f"  {s:<6} {key[1]:<11} z{z}: field {m.mean()*100:5.1f} %  → {len(g):4d} parcels")
    snake = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:3857")
    snake.to_file(snake_cache, driver="GPKG")
    print(f"Snake pipeline finished in {(time.time()-t_all)/60:.1f} min")
for k, t in CLUSTER_TABLES.items():
    print(f"K-means clusters — {k[0]} z{k[1]}"); display(t.round(2))
snake.groupby(["source", "zoom", "site"]).size().unstack("site")
""")
code(r"""
# Detail of the snake pipeline for one site (Esri z18): pre-processing, boundary map, field mask, parcels
key = ("Esri", "Rajbari", 18)
pre_k = preprocess(IMG[key]); B_k = boundary_probability(pre_k, 1.0)
fig, axes = plt.subplots(1, 4, figsize=(24, 6.3))
axes[0].imshow(pre_k); axes[0].set_title("A1 pre-processed")
axes[1].imshow(B_k, cmap="magma", vmax=0.8); axes[1].set_title("A2 augmented boundary probability")
fm = FIELD_MASK[key]; ov = IMG[key] / 255.0
axes[2].imshow(np.where(fm[..., None], ov, ov * 0.35 + np.array([0.75, 0.1, 0.1]) * 0.65)); axes[2].set_title("A3 field mask (red = non-field)")
b = AOI["Rajbari"].bounds; axes[3].imshow(IMG[key], extent=(b[0], b[2], b[1], b[3]))
snake[(snake.source == "Esri") & (snake.site == "Rajbari") & (snake.zoom == 18)].boundary.plot(ax=axes[3], color="#3cb44b", lw=0.7)
axes[3].set_title("A4 snake-refined parcels")
for a in axes: a.axis("off")
plt.suptitle("Figure 3 · Snake pipeline stages (Rajbari, Esri z18)", y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig03_snake_stages.png"), bbox_inches="tight"); plt.show()
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 4 · Method B: SAM 2 (Segment Anything 2.1)
SAM 2 (Ravi et al. 2024) is a promptable segmentation foundation model trained on SA-1B and SA-V. In
**automatic mask generation** it places a grid of point prompts, predicts a mask per point, and keeps
masks that pass quality filters. It knows nothing about fields: it segments any coherent object.

Configuration (tuned earlier on these sites): `sam2.1-hiera-large`, CLAHE-enhanced input, 64 × 64 point
grid plus one crop layer, `pred_iou_thresh = 0.65`, `stability_score_thresh = 0.80`, mask-to-mask
refinement (`use_m2m`). Masks covering > 25 % of the image are dropped. The rest are painted
largest → smallest into a label raster (so small fields are not hidden), then vectorised (≥ 150 m²).

**B2 · SAM 2 + field mask (hybrid).** Because SAM 2 also outlines trees, homesteads and ponds, its
polygons are filtered with the **unsupervised field mask from Method A for the same image**: a polygon
is kept if ≥ 50 % of its area is field. This combines SAM 2's precise outlines with Method A's
land-cover screening.
""")
code(r"""
sam_cache = os.path.join(DIRS["vectors"], "sam2.gpkg")
sam_dir = os.path.join(DIRS["vectors"], "sam2_per_image"); os.makedirs(sam_dir, exist_ok=True)

def clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

if os.path.exists(sam_cache):
    sam = gpd.read_file(sam_cache)
else:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    gen = None
    parts, t_all = [], time.time()
    for key in RUN_IMAGES:
        f_img = os.path.join(sam_dir, "{}_{}_z{}.gpkg".format(*key))
        if os.path.exists(f_img):                                   # per-image cache
            parts.append(gpd.read_file(f_img)); continue
        if gen is None:
            gen = SAM2AutomaticMaskGenerator.from_pretrained(
                "facebook/sam2.1-hiera-large", device=DEVICE, points_per_side=64, points_per_batch=128,
                pred_iou_thresh=0.65, stability_score_thresh=0.80, box_nms_thresh=0.70, crop_n_layers=1,
                crop_n_points_downscale_factor=2, min_mask_region_area=200, use_m2m=True)
        t0 = time.time(); rgb = IMG[key]; sc = scale_at(key[2])
        # masks are returned at full image size: shrink the batch for large images to bound GPU memory
        gen.points_per_batch = int(max(16, 128 * (1024 / rgb.shape[0]) ** 2)) if rgb.shape[0] > 1024 else 128
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            masks = gen.generate(clahe(rgb))
        masks = [m for m in masks if 400 * sc**2 <= m["area"] <= 0.25 * rgb.shape[0] * rgb.shape[1]]
        lab = np.zeros(rgb.shape[:2], np.int32)
        for i, m in enumerate(sorted(masks, key=lambda m: m["area"], reverse=True), start=1):
            lab[m["segmentation"]] = i
        del masks
        g = labels_to_polygons(lab, TR[key], simplify_m=0.5); g["source"], g["site"], g["zoom"] = key
        g.to_file(f_img, driver="GPKG"); parts.append(g); torch.cuda.empty_cache()
        print(f"  {key[0]:<6} {key[1]:<11} z{key[2]}: {len(g):4d} polygons  {time.time()-t0:5.1f}s")
    sam = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:3857")
    sam.to_file(sam_cache, driver="GPKG")
    if gen is not None:
        del gen
    torch.cuda.empty_cache()
    print(f"SAM 2 finished in {(time.time()-t_all)/60:.1f} min")

def filter_by_field_mask(g, min_frac=0.5):
    keep = []
    for key, sub in g.groupby(["source", "site", "zoom"]):
        key = (key[0], key[1], int(key[2]))
        if key not in FIELD_MASK or len(sub) == 0:
            continue
        lab = features.rasterize(zip(sub.geometry, np.arange(1, len(sub) + 1)), out_shape=FIELD_MASK[key].shape,
                                 transform=TR[key], fill=0, dtype="int32")
        frac = np.nan_to_num(ndi.mean(FIELD_MASK[key].astype(float), lab, np.arange(1, len(sub) + 1)))
        keep.append(sub[frac >= min_frac])
    return gpd.GeoDataFrame(pd.concat(keep), crs=g.crs)

sam_masked = filter_by_field_mask(sam)
pd.concat({"SAM 2": sam.groupby(["source", "zoom"]).size(),
           "SAM 2 + field mask": sam_masked.groupby(["source", "zoom"]).size()}, axis=1)
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 5 · Method C: FTW Delineate Anything (`ftw_tools`)
**Delineate Anything** (Lavreniuk et al. 2025) is an instance-segmentation model (YOLO-v11-seg) trained
on **FBIS-22M**: 672 k image patches from 0.25 m to 10 m resolution with 22.9 M field instances. It is
*resolution-agnostic*, so it can run on VHR screenshots. FTW tools ship it as
`ftw inference instance-segmentation`. Two versions are compared:

| Run | How it is executed |
|---|---|
| **C1 · Delineate Anything v1** (YOLO11x) | the released FTW function `ftw_tools.inference.inference.run_instance_segmentation`, unmodified |
| **C2 · Delineate Anything V2** (2026) | FTW's own `DelineateAnything` class (same normalisation, resize, YOLO call and polygonisation). The installed `ftw-tools 2.0.0b5` does not yet list V2, so its checkpoint URL is registered from FTW's development branch. The patch loop mirrors `run_instance_segmentation`. |

**Adaptations for 8-bit VHR screenshots (both versions)**
1. **Input scaling.** FTW divides Sentinel-2 reflectance by 3000 to reach [0, 1]. The RGB screenshot is
   written as a uint16 GeoTIFF scaled by 3000 / 255, so FTW's normalisation maps it back to [0, 1].
2. **De-duplication.** FTW's built-in merge unions overlapping patch predictions on a 10 m grid
   (designed for Sentinel-2). At 0.28–1.1 m this snaps outlines to 10 m and chains nested detections
   (plot inside block) into block-sized blobs. It is disabled (overlap thresholds 0), and duplicates are
   removed with **score-ordered non-maximum suppression** (IoU ≥ 0.3 or ≥ 60 % containment in a
   higher-scoring polygon). Remaining overlaps go to the higher score.

Settings: FTW defaults (patch 256 px, resize × 2, padding 64, conf 0.05, IoU 0.3), `max_detections = 300`,
area 100 m² – 10 ha. FTW's default 500 m² minimum would drop most Bangladeshi plots.
""")
code(r"""
from ftw_tools.inference.inference import run_instance_segmentation
from ftw_tools.inference.models import DelineateAnything

DA_V2_URL = ("https://hf.co/MykolaL/DelineateAnything/resolve/369d0b4c44cf9bec2bd3a27bc81810cadd2c963e/"
             "DelineateAnythingv2.pt")                       # as registered in ftw-baselines (main)
DelineateAnything.checkpoints.setdefault("DelineateAnythingV2", DA_V2_URL)

def nms_partition(gdf, iou_t=0.3, cont_t=0.6):
    gdf = gdf.sort_values("score", ascending=False).reset_index(drop=True)
    kept, ks = [], []
    for g, sc in zip(gdf.geometry.make_valid(), gdf.score):
        if g.is_empty or g.area <= 0:
            continue
        hits = STRtree(kept).query(g, predicate="intersects") if kept else []
        dup = False
        for k in hits:
            i = g.intersection(kept[k]).area
            if i / (g.area + kept[k].area - i) >= iou_t or i / min(g.area, kept[k].area) >= cont_t:
                dup = True; break
        if dup:
            continue
        for k in hits:
            g = g.difference(kept[k])
        if not g.is_empty and g.area > 0:
            kept.append(g); ks.append(sc)
    out = gpd.GeoDataFrame({"score": ks}, geometry=kept, crs=gdf.crs).explode(index_parts=False)
    return out[out.geom_type == "Polygon"].reset_index(drop=True)

def finish(raw):
    if len(raw) == 0:
        return gpd.GeoDataFrame({"score": []}, geometry=[], crs="EPSG:3857")
    u = raw.to_crs(UTM)
    u["geometry"] = u.geometry.make_valid()
    u = u[(u.area > 100) & (u.area < 100_000)]
    g = nms_partition(u); g["geometry"] = g.geometry.simplify(0.5)
    return g[g.area > 100].to_crs(3857)

def ftw_input(key):
    path = os.path.join(DIRS["ftw_input"], "{}_{}_z{}".format(*key), "image.tif")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        write_raster(path, (IMG[key].transpose(2, 0, 1).astype(np.float32) * (3000 / 255)).astype(np.uint16), TR[key])
    return path

def run_da_v1(key):                       # released FTW function, unmodified
    inp = ftw_input(key); out = inp.replace("image.tif", "da_v1_raw.gpkg")
    try:
        run_instance_segmentation(input=inp, model="DelineateAnything", out=out, gpu=0 if DEVICE == "cuda" else None,
                                  num_workers=0, patch_size=256, resize_factor=2, batch_size=8, max_detections=300,
                                  iou_threshold=0.3, conf_threshold=0.05, overwrite=True, simplify=0, min_size=100,
                                  max_size=100_000, close_interiors=True, overlap_iou_threshold=0.0,
                                  overlap_contain_threshold=0.0)
    except ValueError as e:                   # FTW raises when no field is detected in the whole image
        if "No objects to concatenate" not in str(e):
            raise
        return finish(gpd.GeoDataFrame({"score": []}, geometry=[], crs="EPSG:3857"))
    raw = gpd.read_file(out); raw["score"] = raw["confidence"]
    return finish(raw)

def run_da_v2(model, key, patch=256, pad=64):   # FTW DelineateAnything class, same patch logic
    with rasterio.open(ftw_input(key)) as src:
        x = torch.from_numpy(src.read().astype(np.float32)); tr = src.transform; crs = src.crs
    H, W = x.shape[1:]; stride = patch - 2 * pad
    starts = lambda n: sorted(set(list(range(0, max(n - patch, 0) + 1, stride)) + [max(n - patch, 0)]))
    polys = []
    for r0 in starts(H):
        for c0 in starts(W):
            res = model(x[:, r0:r0 + patch, c0:c0 + patch][None])[0]
            ptr = tr * rasterio.Affine.translation(c0, r0)
            g = DelineateAnything.polygonize(res, ptr, crs)
            if g is not None:
                polys.append(g)
    if not polys:
        return finish(gpd.GeoDataFrame({"score": []}, geometry=[], crs=crs))
    raw = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True), crs=crs)
    raw["score"] = raw["confidence"]
    raw["geometry"] = raw.geometry.make_valid().apply(lambda g: Polygon(g.exterior) if g.geom_type == "Polygon" else g)
    return finish(raw)

da_cache = os.path.join(DIRS["vectors"], "delineate_anything.gpkg")
if os.path.exists(da_cache):
    da = gpd.read_file(da_cache)
else:
    parts, t_all = [], time.time()
    v2 = DelineateAnything(model="DelineateAnythingV2", patch_size=256, resize_factor=2, max_detections=300,
                           iou_threshold=0.3, conf_threshold=0.05, device=DEVICE)
    for key in RUN_IMAGES:
        t0 = time.time()
        g1 = run_da_v1(key); g1["version"] = "v1"
        g2 = run_da_v2(v2, key); g2["version"] = "V2"
        for g in (g1, g2):
            g["source"], g["site"], g["zoom"] = key
            parts.append(g[["version", "score", "source", "site", "zoom", "geometry"]])
        print(f"  {key[0]:<6} {key[1]:<11} z{key[2]}: v1 {len(g1):4d} fields · V2 {len(g2):4d} fields  {time.time()-t0:5.1f}s")
    da = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:3857")
    da.to_file(da_cache, driver="GPKG")
    del v2; torch.cuda.empty_cache()
    print(f"Delineate Anything finished in {(time.time()-t_all)/60:.1f} min")
da.groupby(["version", "source", "zoom", "site"]).size().unstack("site")
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 6 · Method D: FTW on Sentinel-2 (PRUE) and the FTW Global Map
The core FTW models are **semantic segmentation** U-Nets trained on the FTW benchmark (Kerner et al.
2025: 1.6 M fields in 24 countries) using **two Sentinel-2 dates** (planting and harvest window), bands
B04, B03, B02 and B08 at 10 m, with three output classes (field interior, field boundary, background).
They cannot use the RGB screenshots.

**D1 · FTW PRUE, run with the FTW CLI.** `ftw inference all` picks scenes from a global crop
calendar. For these sites it selects a **monsoon** window (June–August), where no Sentinel-2 scene is
below 20 % cloud, so it fails. The equivalent step-by-step FTW workflow is used instead:

1. **Scene selection** (Microsoft Planetary Computer STAC, `sentinel-2-l2a`): the clearest scene in the
   dry-season **Boro planting window** (1 Dec 2024 – 28 Feb 2025, *window A*) and in the **growing/harvest
   window** (1 Mar – 15 May 2025, *window B*), cloud < 10 %, lowest cloud then lowest no-data.
2. `ftw inference download --win_a … --win_b … --bbox …` stacks B04, B03, B02, B08 of both dates over a
   2.6 km box around the site (FTW needs ≥ 128 × 128 px; the extra context also helps the model).
3. `ftw inference run --model FTW_PRUE_EFNET_B7`: the **PRUE EfficientNet-B7** U-Net, the model behind
   the FTW Global Map.
4. `ftw inference polygonize` turns the field-interior class into polygons (FTW defaults).

**D2 · FTW Global Map v1.1** (Robinson et al. 2026). The published wall-to-wall predictions of the same
PRUE model on 2024 and 2025 planting/harvest composites, read from Source Cooperative with DuckDB (only
row groups intersecting the sites are fetched from the 2.2 GB Bangladesh GeoParquet).

> **Scale caveat:** one Sentinel-2 pixel is ≈ 18 × 18 z18 screenshot pixels. Plots 15–30 m wide
> are at or below what a 10 m model can resolve. FTW polygons are *remote-sensing field units* that may
> group several plots.
""")
code(r"""
FTW_MODEL, FTW_YEAR = "FTW_PRUE_EFNET_B7", 2025
import pystac_client, planetary_computer
FTW = os.path.join(os.path.dirname(sys.executable), "ftw")
WINDOWS = {"A (planting)": "2024-12-01/2025-02-28", "B (growing/harvest)": "2025-03-01/2025-05-15"}
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)

def ftw_cli(*args):
    r = subprocess.run([FTW, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr).strip().splitlines()[-1])

scenes, s2_parts = [], []
for name in SITES:
    out_dir = os.path.join(DIRS["ftw_s2"], name); os.makedirs(out_dir, exist_ok=True)
    poly = os.path.join(out_dir, "polygons.parquet")
    # download a 2.6 km box (≈ 260 × 260 S2 pixels; FTW needs ≥ 128 px) and clip predictions to the footprint
    w, s_, e, n_ = transform_bounds("EPSG:3857", "EPSG:4326", *AOI[name].buffer(1000 / math.cos(math.radians(SITES[name][0]))).bounds)
    chosen = {}
    for win, dates in WINDOWS.items():
        items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=[w, s_, e, n_], datetime=dates,
                                    query={"eo:cloud_cover": {"lt": 10}}).items())
        items.sort(key=lambda i: (round(i.properties["eo:cloud_cover"], 1), i.properties.get("s2:nodata_pixel_percentage", 0)))
        chosen[win] = items[0]
        scenes.append({"site": name, "window": win, "scene": items[0].id,
                       "date": items[0].datetime.date(), "cloud_%": round(items[0].properties["eo:cloud_cover"], 2)})
    if not os.path.exists(poly) and not os.path.exists(os.path.join(out_dir, "NO_FIELDS")):
        t0 = time.time()
        ftw_cli("inference", "download", "--win_a", chosen["A (planting)"].id, "--win_b", chosen["B (growing/harvest)"].id,
                f"--bbox={w:.5f},{s_:.5f},{e:.5f},{n_:.5f}", "--out", os.path.join(out_dir, "inference_data.tif"), "--overwrite")
        ftw_cli("inference", "run", os.path.join(out_dir, "inference_data.tif"), "--model", FTW_MODEL,
                "--out", os.path.join(out_dir, "inference_output.tif"), "--gpu", "0" if DEVICE == "cuda" else "-1",
                "--resize_factor", "2", "--overwrite")
        with rasterio.open(os.path.join(out_dir, "inference_output.tif")) as src:
            n_field_px = int((src.read(1) == 1).sum())
        if n_field_px > 0:            # FTW's polygonizer fails on a raster without any field pixel
            ftw_cli("inference", "polygonize", os.path.join(out_dir, "inference_output.tif"), "--out", poly, "--overwrite")
        else:
            open(os.path.join(out_dir, "NO_FIELDS"), "w").close()
        print(f"  {name:<11} FTW PRUE-B7 done in {time.time()-t0:.0f}s  (field-interior pixels: {n_field_px})")
    if os.path.exists(poly):
        g = gpd.read_parquet(poly).to_crs(3857)
        g["geometry"] = g.geometry.intersection(AOI[name]); g = g[~g.geometry.is_empty & g.geometry.notna()]
        g["site"] = name; s2_parts.append(g[["site", "geometry"]])
ftw_s2 = (gpd.GeoDataFrame(pd.concat(s2_parts, ignore_index=True), crs="EPSG:3857") if s2_parts
          else gpd.GeoDataFrame({"site": []}, geometry=[], crs="EPSG:3857"))
SCENES = pd.DataFrame(scenes); SCENES.to_csv(os.path.join(DIRS["tables"], "sentinel2_scenes.csv"), index=False)
display(SCENES.pivot(index="site", columns="window", values=["date", "cloud_%"]))
print("FTW PRUE polygons per site (clipped to footprint):", ftw_s2.groupby("site").size().to_dict())
""")
md(r"""
**What the 10 m model sees.** The PRUE output has three classes. The table gives their share inside each
site footprint, and Figure S2 maps them next to the Sentinel-2 *window B* true-colour composite and the
Esri z18 screenshot. Polygons are built only from the **field-interior** class. If almost every pixel is
*boundary* or *background*, there are few or no polygons. This is what happens when plots are narrower
than about 2–3 Sentinel-2 pixels.
""")
code(r"""
CLS = {0: "background", 1: "field interior", 2: "field boundary"}
frac_rows = []
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#2ca02c", "#d62728"])
for j, name in enumerate(SITES):
    out_dir = os.path.join(DIRS["ftw_s2"], name)
    with rasterio.open(os.path.join(out_dir, "inference_output.tif")) as src:
        pred = src.read(1); p_tr, p_crs = src.transform, src.crs
    aoi_p = gpd.GeoSeries([AOI[name]], crs=3857).to_crs(p_crs).iloc[0]
    inside = features.geometry_mask([aoi_p], pred.shape, p_tr, invert=True)
    vals, cnt = np.unique(pred[inside], return_counts=True)
    row = {"site": name, **{CLS[int(v)]: c / inside.sum() * 100 for v, c in zip(vals, cnt)}}
    frac_rows.append(row)
    with rasterio.open(os.path.join(out_dir, "inference_data.tif")) as src:
        d = src.read([5, 6, 7]).astype(float)                       # window B: B04, B03, B02
    rgb = np.clip(d.transpose(1, 2, 0) / np.percentile(d, 99), 0, 1)
    b = aoi_p.bounds; e_ = (p_tr.c, p_tr.c + p_tr.a * pred.shape[1], p_tr.f + p_tr.e * pred.shape[0], p_tr.f)
    axes[0, j].imshow(rgb, extent=e_); axes[1, j].imshow(pred, cmap=cmap, vmin=0, vmax=2, extent=e_, interpolation="nearest")
    for ax in axes[:2, j]:
        ax.plot(*aoi_p.exterior.xy, color="yellow", lw=1.5); ax.set_xlim(b[0] - 300, b[2] + 300); ax.set_ylim(b[1] - 300, b[3] + 300); ax.axis("off")
    axes[0, j].set_title(f"{name}: Sentinel-2 window B (10 m)"); axes[1, j].set_title("PRUE-B7 classes")
    axes[2, j].imshow(IMG[("Esri", name, 18)]); axes[2, j].set_title("Esri z18 (same footprint)"); axes[2, j].axis("off")
fig.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(["#d9d9d9", "#2ca02c", "#d62728"], CLS.values())] +
           [mpatches.Patch(edgecolor="yellow", facecolor="none", label="site footprint")], loc="lower center", ncol=4, frameon=False)
plt.suptitle("Figure S2 · FTW PRUE-B7 on Sentinel-2: 10 m input, predicted classes, and the VHR view of the same footprint", y=1.0)
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "figS2_prue_classes.png"), bbox_inches="tight"); plt.show()
PRUE_FRAC = pd.DataFrame(frac_rows).set_index("site").fillna(0).round(1)
PRUE_FRAC.to_csv(os.path.join(DIRS["tables"], "prue_class_fractions.csv"))
PRUE_FRAC.style.format("{:.1f} %").set_caption("PRUE-B7 class shares inside each footprint")
""")
code(r"""
import duckdb
FTW_URL = ("https://data.source.coop/ftw/global-data/predictions/vectors/alpha/"
           "results-by-admin-conf/admin:country_code=BD/Bangladesh.parquet")
gm_cache = os.path.join(DIRS["vectors"], "ftw_global_map.gpkg")
if os.path.exists(gm_cache):
    gmap = gpd.read_file(gm_cache)
else:
    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    parts = []
    for name in SITES:
        w, s_, e, n_ = transform_bounds("EPSG:3857", "EPSG:4326", *AOI[name].bounds)
        df = con.execute(f'''SELECT "metrics:area" AS area_m2, confidence,
                   year("determination:datetime" AT TIME ZONE 'UTC') AS year, ST_AsWKB(geometry) AS wkb
                   FROM read_parquet('{FTW_URL}')
                   WHERE bbox.xmax >= {w} AND bbox.xmin <= {e} AND bbox.ymax >= {s_} AND bbox.ymin <= {n_}''').fetchdf()
        g = gpd.GeoDataFrame(df.drop(columns="wkb"), geometry=gpd.GeoSeries.from_wkb(df.wkb.apply(bytes)), crs="EPSG:4326")
        g["site"] = name; parts.append(g)
    gmap = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    gmap.to_file(gm_cache, driver="GPKG")
gmap = gmap.to_crs(3857)
gmap["geometry"] = [g.intersection(AOI[s]) for g, s in zip(gmap.geometry, gmap.site)]
gmap = gmap[~gmap.geometry.is_empty]
gmap.groupby(["site", "year"]).agg(n=("area_m2", "size"), median_m2=("area_m2", "median"),
                                   confidence=("confidence", "mean")).round(1).unstack("year")
""")

# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 7 · Results
### 7.1 Assemble all runs
A **run** is one method applied to one image (source × site × zoom). FTW Sentinel-2 products are
image-independent and have a single run per site. All polygons are clipped to the site footprint and
rasterised onto a **common zoom-18 evaluation grid** (1024 × 1024 per site). Point sampling, coverage and
agreement are computed on that grid; edge alignment is computed on each run's own input image.
""")
code(r"""
rows = []
def add(g, method, source, zoom):
    g = g.copy(); g["method"], g["source"], g["zoom"] = method, source, zoom
    rows.append(g[["site", "method", "source", "zoom", "geometry"]])

for (s, n, z), g in snake.groupby(["source", "site", "zoom"]):      add(g, "Snake pipeline", s, int(z))
for (s, n, z), g in sam.groupby(["source", "site", "zoom"]):        add(g, "SAM 2", s, int(z))
for (s, n, z), g in sam_masked.groupby(["source", "site", "zoom"]): add(g, "SAM 2 + field mask", s, int(z))
for (v, s, n, z), g in da.groupby(["version", "source", "site", "zoom"]): add(g, f"FTW Delineate Anything {v}", s, int(z))
if len(ftw_s2):
    add(ftw_s2, "FTW PRUE-B7 S2 (ftw inference)", "Sentinel-2", 10)
for yr in (2025, 2024):
    add(gmap[gmap.year == yr], f"FTW Global Map {yr}", "Sentinel-2", 10)
ALL = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), crs="EPSG:3857")
ALL = ALL[ALL.geometry.notna() & ~ALL.geometry.is_empty].reset_index(drop=True)
ALL.to_crs(4326).to_file(os.path.join(DIRS["vectors"], "all_runs.gpkg"), driver="GPKG")

METHODS = ["Snake pipeline", "SAM 2", "SAM 2 + field mask", "FTW Delineate Anything v1", "FTW Delineate Anything V2"]
S2_METHODS = [m for m in ALL.method.unique() if "S2" in m or "Global" in m]
COLORS = {"Snake pipeline": "#3cb44b", "SAM 2": "#42d4f4", "SAM 2 + field mask": "#4363d8",
          "FTW Delineate Anything v1": "#f032e6", "FTW Delineate Anything V2": "#ffe119",
          "FTW Global Map 2025": "#e6194b", "FTW Global Map 2024": "#f58231", "FTW PRUE-B7 S2 (ftw inference)": "#911eb4"}

RUNS = (ALL[["method", "source", "zoom"]].drop_duplicates()
        .sort_values(["method", "source", "zoom"]).itertuples(index=False, name=None))
RUNS = list(RUNS)
EVAL_TR = {n: TR[("Esri", n, 18)] for n in SITES}
def rasterize_eval(g, n):
    if len(g) == 0:
        return np.zeros((REF_SIZE, REF_SIZE), np.int32)
    return features.rasterize(zip(g.geometry, np.arange(1, len(g) + 1)), out_shape=(REF_SIZE, REF_SIZE),
                              transform=EVAL_TR[n], fill=0, dtype="int32")
LAB = {}
for (m, s, z) in RUNS:
    for n in SITES:
        g = ALL[(ALL.method == m) & (ALL.source == s) & (ALL.zoom == z) & (ALL.site == n)]
        if len(g) or (s, n, z) in RUN_IMAGES or s == "Sentinel-2":
            LAB[(m, s, z, n)] = rasterize_eval(g, n)
print(f"{len(RUNS)} method × source × zoom combinations, {len(LAB)} site-level runs, {len(ALL)} polygons")
""")

md(r"""
### 7.2 Visual comparison
Figure 4 shows each site at zoom 18 (both sources) for every method. Figure 5 shows a 200 m × 200 m
detail at the site centre, so outline quality can be judged. Figure 6 shows how one method's result
changes with zoom.
""")
code(r"""
def ext(n):
    b = AOI[n].bounds; return (b[0], b[2], b[1], b[3])
s2_main = "FTW PRUE-B7 S2 (ftw inference)"
for n in SITES:
    cols = METHODS + [s2_main]
    fig, axes = plt.subplots(2, len(cols), figsize=(5.2 * len(cols), 10.8))
    for i, s in enumerate(SOURCES):
        for j, m in enumerate(cols):
            ax = axes[i, j]; ax.imshow(IMG[(s, n, 18)], extent=ext(n))
            sel = ALL[(ALL.site == n) & (ALL.method == m) & ((ALL.source == s) & (ALL.zoom == 18) if m in METHODS else True)]
            sel.boundary.plot(ax=ax, color=COLORS[m], linewidth=0.7)
            ax.set_title(f"{s} z18 · {m} ({len(sel)})", fontsize=10); ax.axis("off")
    plt.suptitle(f"Figure 4 · {n}: all methods at zoom 18 (top Esri, bottom Google)", y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], f"fig04_{n}.png"), bbox_inches="tight"); plt.show()
""")
code(r"""
cols = METHODS + [s2_main, "FTW Global Map 2025"]
fig, axes = plt.subplots(len(SITES), len(cols), figsize=(4.6 * len(cols), 4.6 * len(SITES)))
for i, n in enumerate(SITES):
    cx, cy = AOI[n].centroid.x, AOI[n].centroid.y; half = 100 / math.cos(math.radians(SITES[n][0]))
    for j, m in enumerate(cols):
        ax = axes[i, j]; ax.imshow(IMG[("Esri", n, 18)], extent=ext(n))
        sel = ALL[(ALL.site == n) & (ALL.method == m) & ((ALL.source == "Esri") & (ALL.zoom == 18) if m in METHODS else True)]
        sel.boundary.plot(ax=ax, color=COLORS[m], linewidth=1.3)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{n} · {m}", fontsize=10)
plt.suptitle("Figure 5 · 200 m × 200 m detail at each site centre (Esri z18)", y=1.0)
plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], "fig05_detail.png"), bbox_inches="tight"); plt.show()
""")
code(r"""
for m in ["SAM 2 + field mask", "FTW Delineate Anything V2", "Snake pipeline"]:  # v1 is near-empty, see Table 1
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    n = "Patuakhali"; cx, cy = AOI[n].centroid.x, AOI[n].centroid.y; half = 140 / math.cos(math.radians(SITES[n][0]))
    for i, s in enumerate(SOURCES):
        for j, z in enumerate(ZOOMS):
            ax = axes[i, j]; ax.imshow(IMG[(s, n, z)], extent=ext(n))
            sel = ALL[(ALL.site == n) & (ALL.method == m) & (ALL.source == s) & (ALL.zoom == z)]
            sel.boundary.plot(ax=ax, color=COLORS[m], linewidth=1.1)
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{s} z{z} ({len(sel)} polygons)" + ("" if (s, n, z) in RUN_IMAGES else " — not usable"))
    plt.suptitle(f"Figure 6 · Effect of zoom on {m} (Patuakhali, 280 m × 280 m detail)", y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(DIRS["figures"], f"fig06_zoom_{m.split()[0]}.png"), bbox_inches="tight"); plt.show()
""")

md(r"""
### 7.3 Field-detection accuracy and polygon statistics
For every method × source × zoom: field recall, false-field rate, overall accuracy (OA) and κ, pooled
over the sites where the run exists. Plus number of polygons, median polygon size and footprint coverage.
**Edge alignment** is the mean image gradient (Sobel on the bilateral-filtered L\*) on polygon outlines
divided by that inside polygons, measured on the run's own image (Esri z18 for Sentinel-2 products).
Higher means outlines sit on visible edges. *The snake pipeline explicitly moves outlines onto gradient
ridges, so this metric favours it by construction.*
""")
code(r"""
def grad_map(rgb):
    L = cv2.bilateralFilter(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0], 9, 25, 5).astype(np.float32)
    return np.hypot(ndi.sobel(L, 0), ndi.sobel(L, 1))
GRAD = {k: grad_map(v) for k, v in IMG.items() if k in RUN_IMAGES or k[2] == 18}

def edge_alignment(g, key, sc):
    if len(g) == 0:
        return np.nan
    lab = features.rasterize(zip(g.geometry, np.arange(1, len(g) + 1)), out_shape=IMG[key].shape[:2],
                             transform=TR[key], fill=0, dtype="int32")
    edge = find_boundaries(lab, mode="inner") & (lab > 0)
    inner = ndi.binary_erosion(lab > 0, iterations=max(1, int(3 * sc)))
    return GRAD[key][edge].mean() / GRAD[key][inner].mean() if edge.sum() and inner.sum() else np.nan

recs = []
for (m, s, z) in RUNS:
    y, p, ea, nf, cov, med = [], [], [], 0, [], []
    for n in SITES:
        if (m, s, z, n) not in LAB:
            continue
        lab = LAB[(m, s, z, n)]; r = REF[REF.site == n]
        inv = ~EVAL_TR[n]
        for pt, f in zip(r.geometry, r.is_field):
            c, rr = inv * (pt.x, pt.y); y.append(f); p.append(int(lab[int(rr), int(c)] > 0))
        g = ALL[(ALL.method == m) & (ALL.source == s) & (ALL.zoom == z) & (ALL.site == n)]
        nf += len(g); cov.append((lab > 0).mean() * 100); med += list(g.to_crs(UTM).area)
        key = (s, n, z) if s != "Sentinel-2" else ("Esri", n, 18)
        ea.append(edge_alignment(g, key, scale_at(key[2])))
    y, p = np.array(y), np.array(p)
    recs.append({"method": m, "source": s, "zoom": z, "sites": len(cov), "n_points": len(y),
                 "field_recall": p[y == 1].mean(), "false_field_rate": p[y == 0].mean(),
                 "OA": accuracy_score(y, p), "kappa": cohen_kappa_score(y, p),
                 "n_polygons": nf, "median_m2": np.median(med) if med else np.nan,
                 "coverage_%": np.mean(cov), "edge_alignment": np.nanmean(ea)})
RES = pd.DataFrame(recs)
RES.to_csv(os.path.join(DIRS["tables"], "results_by_run.csv"), index=False)
fmt = {"field_recall": "{:.2f}", "false_field_rate": "{:.2f}", "OA": "{:.2f}", "kappa": "{:.2f}",
       "median_m2": "{:.0f}", "coverage_%": "{:.1f}", "edge_alignment": "{:.2f}"}
(RES.sort_values(["method", "source", "zoom"]).set_index(["method", "source", "zoom"])
    .style.format(fmt).background_gradient(subset=["kappa"], cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    .set_caption("Table 1 · Accuracy and polygon statistics per method × source × zoom"))
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(21, 5.6))
for ax, metric, ttl in zip(axes, ["kappa", "field_recall", "false_field_rate"],
                           ["Cohen's κ (field vs non-field)", "Field recall", "False-field rate"]):
    for m in METHODS:
        for s, ls in zip(SOURCES, ["-", "--"]):
            d = RES[(RES.method == m) & (RES.source == s)].sort_values("zoom")
            ax.plot(d.zoom, d[metric], ls, marker="o", color=COLORS[m], label=f"{m} · {s}")
    for m in S2_METHODS:
        v = RES[RES.method == m][metric]
        if len(v):
            ax.axhline(v.iloc[0], color=COLORS.get(m, "grey"), lw=1.2, ls=":", label=m)
    ax.set_xticks(ZOOMS); ax.set_xlabel("zoom level"); ax.set_title(ttl); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.62), ncol=2)
plt.suptitle("Figure 7 · Effect of zoom level and imagery source on field detection (dotted = Sentinel-2 products)", y=1.02)
plt.savefig(os.path.join(DIRS["figures"], "fig07_zoom_effect.png"), bbox_inches="tight"); plt.show()
""")

md(r"""
### 7.4 Consistency across zoom levels and imagery sources
A robust method should give similar fields for the same ground area at different zooms, and
similar fields from Esri and Google where the land has not changed. Consistency is measured on the
common zoom-18 grid as **field-mask IoU** and **boundary F-score** (outline pixels matched within 5 m).
""")
code(r"""
def iou(a, b):
    a, b = a > 0, b > 0; u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan

def boundary_f(a, b, tol_px):
    ea, eb = find_boundaries(a, mode="inner") & (a > 0), find_boundaries(b, mode="inner") & (b > 0)
    if ea.sum() == 0 or eb.sum() == 0:
        return np.nan
    pa = (ndi.distance_transform_edt(~eb)[ea] <= tol_px).mean(); rb = (ndi.distance_transform_edt(~ea)[eb] <= tol_px).mean()
    return 2 * pa * rb / (pa + rb) if pa + rb else 0.0

TOL = {n: 5.0 / (EVAL_TR[n].a * math.cos(math.radians(SITES[n][0]))) for n in SITES}
def pair_score(k1, k2):
    v = [(iou(LAB[k1 + (n,)], LAB[k2 + (n,)]), boundary_f(LAB[k1 + (n,)], LAB[k2 + (n,)], TOL[n]))
         for n in SITES if k1 + (n,) in LAB and k2 + (n,) in LAB]
    return (np.nanmean([a for a, _ in v]), np.nanmean([b for _, b in v]), len(v)) if v else (np.nan, np.nan, 0)

cons = []
for m in METHODS:
    for s in SOURCES:
        for z1, z2 in [(17, 18), (18, 19), (17, 19)]:
            i, b, k = pair_score((m, s, z1), (m, s, z2))
            cons.append({"method": m, "comparison": f"{s}: z{z1} ↔ z{z2}", "IoU": i, "boundary_F": b, "sites": k})
    for z in ZOOMS:
        i, b, k = pair_score((m, "Esri", z), (m, "Google", z))
        cons.append({"method": m, "comparison": f"Esri ↔ Google at z{z}", "IoU": i, "boundary_F": b, "sites": k})
g25, g24 = ("FTW Global Map 2025", "Sentinel-2", 10), ("FTW Global Map 2024", "Sentinel-2", 10)
i, b, k = pair_score(g25, g24); cons.append({"method": "FTW Global Map", "comparison": "2025 ↔ 2024", "IoU": i, "boundary_F": b, "sites": k})
CONS = pd.DataFrame(cons); CONS.to_csv(os.path.join(DIRS["tables"], "consistency.csv"), index=False)
CONS.pivot_table(index="comparison", columns="method", values="IoU").round(2).style.format("{:.2f}", na_rep="–") \
    .background_gradient(cmap="viridis", vmin=0, vmax=1).set_caption("Table 2a · Field-mask IoU between zooms / sources")
""")
code(r"""
CONS.pivot_table(index="comparison", columns="method", values="boundary_F").round(2).style.format("{:.2f}", na_rep="–") \
    .background_gradient(cmap="viridis", vmin=0, vmax=1).set_caption("Table 2b · Boundary F-score (5 m tolerance) between zooms / sources")
""")
code(r"""
# Agreement between methods at zoom 18 (Esri) and the Sentinel-2 products
sel = [(m, "Esri", 18) for m in METHODS] + [(m, "Google", 18) for m in METHODS] + \
      [r for r in RUNS if r[1] == "Sentinel-2"]
lbl = [f"{m} · {s}" + (f" z{z}" if s != "Sentinel-2" else "") for m, s, z in sel]
M = np.array([[pair_score(a, b)[1] for b in sel] for a in sel])
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
for i in range(len(sel)):
    for j in range(len(sel)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] < 0.6 else "black")
ax.set_xticks(range(len(sel))); ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(sel))); ax.set_yticklabels(lbl, fontsize=9); plt.colorbar(im, fraction=0.04)
ax.set_title("Figure 8 · Boundary agreement between methods (F-score, 5 m tolerance, mean of 5 sites)")
plt.savefig(os.path.join(DIRS["figures"], "fig08_agreement.png"), bbox_inches="tight"); plt.show()
""")

md(r"""
### 7.5 Ranking
Each method's **best configuration** (source × zoom with the highest κ), and its mean and range over all
configurations, which shows how sensitive it is to the imagery. Only configurations covering **all five
sites** are ranked. Esri z19 exists only at Barisal (24 points) and is too small a sample to rank.
""")
code(r"""
R5 = RES[RES.sites == len(SITES)]
best = R5.loc[R5.groupby("method").kappa.idxmax(), ["method", "source", "zoom", "kappa", "field_recall",
                                                      "false_field_rate", "edge_alignment", "median_m2", "coverage_%"]]
best = best.rename(columns={"source": "best source", "zoom": "best zoom", "kappa": "best κ"})
best["mean κ (all configs)"] = best.method.map(R5.groupby("method").kappa.mean())
best["κ range"] = best.method.map(R5.groupby("method").kappa.apply(lambda v: f"{v.min():.2f} – {v.max():.2f}"))
best = best.sort_values("best κ", ascending=False).set_index("method")
best.to_csv(os.path.join(DIRS["tables"], "ranking.csv"))
best.style.format({"best κ": "{:.2f}", "field_recall": "{:.2f}", "false_field_rate": "{:.2f}", "edge_alignment": "{:.2f}",
                   "median_m2": "{:.0f}", "coverage_%": "{:.1f}", "mean κ (all configs)": "{:.2f}"}) \
    .set_caption("Table 3 · Best configuration per method")
""")


# ════════════════════════════════════════════════════════════════════════════
md(r"""
---
# Part 8 · Interpretation and conclusions
*Numbers refer to the reference run in this notebook: 121 independent points (96 field, 25 non-field);
configurations covering all five sites unless stated otherwise.*

### 8.1 Imagery: which zoom shows field boundaries best?
* **Esri publishes no zoom-19 imagery at four of five sites** (grey placeholder tiles), and genuine
  z19 only at Barisal. **Google has genuine z19 imagery at all five sites**, with 12–17× more
  fine-scale detail than an enlarged z18, of the same scene.
* Esri and Google show **different acquisition dates** (crop stage, flooding, dry-season bare soil). This
  changed boundary visibility as much as zoom did. For example, Faridpur's boundaries are faint in both
  sources, while Bhola and Patuakhali are much crisper in Google.
* The dynamic rule (highest genuine zoom with boundary density ≥ 0.02) chose **z19 for 4 of 5 Google
  sites**, but z18 for 4 of 5 Esri sites.
* **More pixels did not mean better field detection** (Figure 7). The three methods with useful accuracy
  (snake pipeline, SAM 2 + field mask, Delineate Anything V2) all reached their highest κ at **z17**.
  The two weakest (SAM 2 alone, Delineate Anything v1) peak at Google z19, but at κ ≤ 0.23. At z19, within-field texture (crop rows, furrows, soil patches) becomes visible and
  produces spurious edges and over-segmentation. For plots 15–60 m wide, ≈ 0.5–1 m pixels are enough.

### 8.2 Method by method
| Method | Best (all 5 sites) | Behaviour |
|---|---|---|
| **Snake pipeline** (A) | κ ≈ 0.64 (Google z17), 0.63 (Esri z17) | Highest κ, and field recall 0.80–0.92 at z17–z18. Its unsupervised field mask removes most trees and homesteads. Outlines sit on visible edges (edge alignment ≈ 2.8–2.9, favoured by design). Over-segments at z18–19 (median plot 260–300 m²). |
| **SAM 2 + field mask** (B2) | κ ≈ 0.54 (Google z17); 0.52–0.53 at Google z18/z19 | The most **stable** hybrid across zooms. Keeps SAM 2's plot-faithful outlines and removes most non-field objects. Works noticeably better on Google than on Esri (κ 0.25–0.42). |
| **SAM 2** alone (B) | κ ≈ 0.23 (Google z19) | Highest field recall (0.58–0.93) and the most consistent geometry across zooms (IoU 0.73–0.87) and sources (0.65–0.79). But it is class-agnostic: trees, homesteads and ponds are outlined too (false-field rate 0.68–0.96), so κ ≈ 0. Good geometry, no semantics. |
| **FTW Delineate Anything V2** (C2) | κ ≈ 0.36 (Google z17) | Conservative: almost never labels non-field land (false-field ≤ 0.08) but finds only 37–63 % of field points. Crisp rectangular plots (median ≈ 150–195 m²). Least consistent across zooms and sources (IoU 0.21–0.37): sensitive to the image. |
| **FTW Delineate Anything v1** (C1) | κ ≤ 0.19 | Released FTW pipeline, unmodified. It detects few, very large objects (median 1,000–10,000 m²) and is unstable across zooms and sources. **Not suitable** for these VHR smallholder scenes. V2 is a large improvement. |
| **FTW PRUE-B7 on Sentinel-2** (D1) | κ ≈ 0.09 | At 10 m most of the landscape is predicted as **field boundary** (62–81 % at Bhola, Patuakhali, Barisal) or background (Faridpur, a dry-season bare scene), so few field-interior polygons remain. The exception is **Rajbari** (74 % interior), the only site with larger plots, where PRUE reproduces the field mosaic convincingly (Figure S2). |
| **FTW Global Map 2025 / 2024** (D2) | κ ≈ 0.17 / 0.10 | Conservative (false-field ≤ 0.04) but low recall (0.21–0.37). The 2024 and 2025 maps agree poorly (IoU 0.23). Its confidence layer is nonetheless high (89–98) here, so it does **not** flag the problem. |

### 8.3 Imagery source
For the VHR methods, **Google imagery gave equal or higher κ than Esri** at the same zoom in most
configurations (e.g. SAM 2 + field mask 0.54 vs 0.25 at z17, Delineate Anything V2 0.36 vs 0.19).
Google was sharper and more recent here, and it has real z19 imagery. Esri remains the licence-friendly
choice. Its z17–z18 imagery with the snake pipeline gives comparable results (κ 0.47–0.63).

### 8.4 Conclusions and recommendations
1. **Sentinel-2 field-boundary products, including FTW, are too coarse for Bangladeshi smallholder
   plots.** They either merge plots into "boundary" or miss them. They remain useful as a cropland
   prior and where plots are larger (Rajbari-type landscapes).
2. **On VHR screenshots, separating field from non-field land matters as much as drawing the outline.**
   The two best approaches both combine a geometric delineator with an unsupervised field / non-field
   mask:
   * the **snake pipeline** for the best field-detection accuracy (κ ≈ 0.63–0.64), and
   * **SAM 2 + field mask** for the most stable outlines across zooms and dates (κ ≈ 0.52–0.54 on Google).
3. **Use zoom 17–18** (≈ 0.5–1 m). Zoom 19 adds texture rather than better boundaries, and Esri rarely has it.
4. **Delineate Anything V2** is a good "precision" option (very few false fields), but it needs a
   recall-boosting step, e.g. as prompts or seeds for SAM 2 or the snake pipeline.
5. For operational mapping, the next steps are (i) hand-digitised reference parcels for a boundary-level
   metric (Jaccard / polygon IoU), (ii) fine-tuning Delineate Anything or SAM 2 on a few hundred
   Bangladeshi plots, and (iii) multi-date imagery to resolve boundaries invisible on a single date.

### 8.5 Limitations
* **Field detection, not boundary position.** Without surveyed parcels, the accuracy metric checks whether
  field and non-field points fall inside polygons, not how precisely outlines match true boundaries.
  Edge alignment and the consistency metrics are proxies.
* **Small non-field sample.** 25 non-field points: one point changes the false-field rate by 0.04,
  so κ differences below ~0.1 are within noise.
* **Tuning bias.** The snake pipeline and the field mask were developed on these five sites, so their
  scores are optimistic relative to methods used off the shelf (Delineate Anything, SAM 2, FTW).
* **Different dates.** The Esri, Google and Sentinel-2 scenes have different, partly unknown
  acquisition dates, and the reference points were interpreted on Esri z18.
* **Pre-release software.** `ftw-tools 2.0.0b5` is a pre-release. Delineate Anything V2 was run through
  FTW's classes with the V2 checkpoint registered manually, and FTW's 10 m polygon merging was replaced
  by score-ordered NMS for VHR imagery.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3.12 (FieldEdge)", "language": "python", "name": "fieldedge"},
    "language_info": {"name": "python"}})
nbf.write(nb, "field_boundary_comparison.ipynb")
print("written", len(cells), "cells")
