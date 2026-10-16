"""Field-edge detection core: capture basemap imagery for a bounding box, resample to the models'
training resolution (0.25 m), run a trained model, and return field polygons + boundary lines.

Models (trained on ../Labeled_Data):
  • "Delineate Anything V2 (fine-tuned)"  — fast; F1 0.64 / boundary F1 0.65 on the held-out Test sites
  • "SAM 2.1 (pre-trained, calibrated)"   — most accurate; F1 0.68 / boundary F1 0.77; ≈ 1 min per 0.06 km²
  • "FC-DenseNet"                         — boundary segmentation; depth and threshold chosen on Valid
"""
import io, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import cv2
import geopandas as gpd
import numpy as np
import shapely
import torch
from PIL import Image
from rasterio import features
from rasterio.transform import Affine, from_bounds
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from shapely.geometry import LineString, Polygon, box, shape
from shapely.strtree import STRtree
from skimage.color import rgb2lab
from skimage.draw import polygon as draw_polygon
from skimage.measure import find_contours, regionprops
from skimage.segmentation import find_boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "..", "Labeled_Data")
DA_FINETUNED = os.path.join(STUDY, "train_outputs", "da", "runs", "finetune", "weights", "best.pt")
DA_FINETUNED_LEGACY = os.path.join(
    HERE, "..", "Updated_analysis_labeled_data", "train_outputs", "da", "runs", "finetune", "weights", "best.pt"
)
DA_URL = os.environ.get(
    "FIELD_EDGE_WEIGHTS_URL",
    "https://github.com/zia207/field-edge-dashboard/releases/download/weights-v1/best.pt",
)
SAM2_ID = "facebook/sam2.1-hiera-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GSD = 0.25                      # training resolution of the fine models (m / px)
# SAM 2.1 on a Sentinel-2 screenshot. 2.5 m keeps a large field inside one window.
S2_LARGE_GSD = 2.5
MIN_FIELD_M2 = 20
MIN_PX = int(MIN_FIELD_M2 / GSD**2)

SENTINEL2 = "Sentinel-2"
# False colour is near-infrared, red, green (B08, B04, B03) from the same L2A scene.
S2_FALSE = "False colour"
S2_TRUE = "True colour"
# Same scene true colour would pick, then the four 10 m bands: red, green, blue, near-infrared.
S2_RGBNIR = "RGB + NIR (SAM 2.1)"
S2_FALSE_BANDS = ("B08", "B04", "B03")
S2_RGBNIR_BANDS = ("B04", "B03", "B02", "B08")
# Annual cloudless mosaic for the map. Detection uses a dated L2A scene, not these tiles.
S2_MOSAIC = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg"
S2_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
S2_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
SOURCES = {
    "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "Esri World Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    SENTINEL2: S2_MOSAIC,
}
# Highest zoom the source actually publishes. Sentinel-2 mosaic tiles stop at 14 (~10 m).
NATIVE_ZOOM = {"Google Satellite": 19, "Esri World Imagery": 19, SENTINEL2: 14}
# Map-scale presets: zoom level and its nominal ground pixel size at the equator
SCALES = {
    "20 m scale (zoom 19, ≈ 0.28 m/px) — recommended": 19,
    "50 m scale (zoom 18, ≈ 0.55 m/px)": 18,
}
# Settings selected on the Valid sites in the training study
DA_SETTINGS = dict(patch=768, conf=0.25)
SAM2_SETTINGS = dict(pred_iou_thresh=0.85, stability_score_thresh=0.80)

TILE = 256
ORIGIN = math.pi * 6378137.0


# ── imagery ──────────────────────────────────────────────────────────────
def _global_px(lon, lat, z):
    n = TILE * 2 ** z
    s = math.sin(math.radians(lat))
    return (lon + 180.0) / 360.0 * n, (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n

def _px_to_3857(px, py, z):
    res = 2 * ORIGIN / (TILE * 2 ** z)
    return px * res - ORIGIN, ORIGIN - py * res

def _fetch(url, retries=4):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (field-edge research app)"})
    for k in range(retries):
        try:
            with urlopen(req, timeout=20) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            time.sleep(0.5 * (k + 1))
    return None

def bbox_area_km2(west, south, east, north):
    lat = (south + north) / 2
    return ((east - west) * 111.32 * math.cos(math.radians(lat))) * ((north - south) * 110.57)


# One detection pass covers up to DIRECT_KM2. A larger rectangle, up to SELECT_MAX_KM2, is split
# into sample tiles of at most DIRECT_KM2. The split is a lat/lon grid, so the selection is a rectangle.
DIRECT_KM2 = 4.0
SELECT_MAX_KM2 = 10.0
TILE_OVERLAP_M = 80.0


def is_latlon_rectangle(geom, tol=1e-7):
    """True when `geom` is an axis-aligned rectangle in longitude/latitude."""
    if geom is None or geom.is_empty or geom.geom_type != "Polygon" or len(geom.interiors):
        return False
    coords = list(geom.exterior.coords)
    if len(coords) != 5:
        return False
    xs = [c[0] for c in coords[:-1]]
    ys = [c[1] for c in coords[:-1]]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    if maxx - minx <= tol or maxy - miny <= tol:
        return False
    return all(min(abs(x - minx), abs(x - maxx)) <= tol and min(abs(y - miny), abs(y - maxy)) <= tol
               for x, y in coords[:-1])


def sample_tile_grid(west, south, east, north, max_km2=DIRECT_KM2):
    """Split an axis-aligned rectangle into cells of at most `max_km2`.

    One cell is returned when the rectangle is already within the limit. Cells meet at their edges
    and cover the rectangle.
    """
    def cells(ncols, nrows):
        out = []
        for r in range(nrows):
            s = south + (north - south) * r / nrows
            n = south + (north - south) * (r + 1) / nrows
            for c in range(ncols):
                w = west + (east - west) * c / ncols
                e = west + (east - west) * (c + 1) / ncols
                out.append((w, s, e, n))
        return out

    if bbox_area_km2(west, south, east, north) <= max_km2 + 1e-6:
        return [(west, south, east, north)]
    lat = (south + north) / 2
    width_km = (east - west) * 111.32 * math.cos(math.radians(lat))
    height_km = (north - south) * 110.57
    ncols = nrows = 1
    tiles = cells(1, 1)
    while max(bbox_area_km2(*t) for t in tiles) > max_km2 + 1e-6:
        if (width_km / ncols) >= (height_km / nrows):
            ncols += 1
        else:
            nrows += 1
        tiles = cells(ncols, nrows)
        if ncols * nrows > 64:
            break
    return tiles


def expand_bbox(bbox, meters, limit):
    """Grow `bbox` by `meters` on each side, without leaving `limit`."""
    west, south, east, north = bbox
    lat = (south + north) / 2
    dlon = meters / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    dlat = meters / 110570.0
    lw, ls, le, ln = limit
    return (max(lw, west - dlon), max(ls, south - dlat), min(le, east + dlon), min(ln, north + dlat))


def _core_box(tile, outer):
    """Cell polygon. Shared east and north edges belong to the next cell so a point is in one cell."""
    west, south, east, north = tile
    ow, os_, oe, on = outer
    eps = 1e-12
    if east < oe - eps:
        east -= eps
    if north < on - eps:
        north -= eps
    return box(west, south, east, north)


def keep_core_fields(fields, core_bbox, outer_bbox):
    """Fields whose representative point lies in this sample tile, clipped to the selected rectangle."""
    empty = fields.iloc[0:0].copy() if fields is not None else gpd.GeoDataFrame(
        {"field_id": [], "area_m2": [], "perimeter_m": [], "score": []}, geometry=[], crs="EPSG:4326")
    if fields is None or not len(fields):
        return empty
    core = _core_box(core_bbox, outer_bbox)
    outer = box(*outer_bbox)
    geoms, scores = [], []
    for geom, score in zip(fields.geometry, fields.score if "score" in fields.columns else [np.nan] * len(fields)):
        if geom is None or geom.is_empty:
            continue
        if not core.covers(geom.representative_point()):
            continue
        clipped = shapely.make_valid(geom).intersection(outer)
        if clipped.is_empty:
            continue
        parts = [clipped] if clipped.geom_type == "Polygon" else list(getattr(clipped, "geoms", []))
        for part in parts:
            if getattr(part, "geom_type", None) != "Polygon" or part.is_empty:
                continue
            geoms.append(part)
            scores.append(float(score) if score == score else np.nan)
    if not geoms:
        return empty
    g = gpd.GeoDataFrame({"score": scores}, geometry=geoms, crs="EPSG:4326")
    utm = utm_crs(outer_bbox)
    gu = g.to_crs(utm)
    gu = gu[gu.area >= MIN_FIELD_M2].copy()
    if not len(gu):
        return empty
    gu["area_m2"] = gu.area.round(1)
    gu["perimeter_m"] = gu.length.round(1)
    gu["field_id"] = np.arange(1, len(gu) + 1)
    return gu[["field_id", "area_m2", "perimeter_m", "score", "geometry"]].to_crs(4326)


def concat_fields(frames):
    """Stack per-tile field layers and renumber `field_id`."""
    parts = [g for g in frames if g is not None and len(g)]
    empty = gpd.GeoDataFrame(
        {"field_id": [], "area_m2": [], "perimeter_m": [], "score": []}, geometry=[], crs="EPSG:4326")
    if not parts:
        return empty
    g = gpd.GeoDataFrame(pd_concat_fields(parts), geometry="geometry", crs="EPSG:4326")
    g["field_id"] = np.arange(1, len(g) + 1)
    return g


def pd_concat_fields(parts):
    import pandas as pd
    return pd.concat(parts, ignore_index=True)


def concat_gdfs(frames):
    """Stack GeoDataFrames that share a CRS. None when every frame is empty."""
    import pandas as pd
    parts = [g for g in frames if g is not None and len(g)]
    if not parts:
        return None
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs)


def lines_from_fields(fields):
    """Boundary of each field polygon, in EPSG:4326."""
    empty = gpd.GeoDataFrame({"field_id": []}, geometry=[], crs="EPSG:4326")
    if fields is None or not len(fields):
        return empty
    utm = utm_crs(tuple(fields.total_bounds))
    gu = fields.to_crs(utm)
    return gpd.GeoDataFrame(gu[["field_id"]].copy(), geometry=gu.geometry.boundary.values, crs=utm).to_crs(4326)


def polygonize_fields(fields, simplify_m=15.0, min_m2=500.0, max_m2=0.0, close_interiors=False):
    """Turn field polygons into the vector outlines used by Fields of the World.

    Matches `ftw inference polygonize`: simplify in metres (0 leaves the vertices),
    drop fields smaller than `min_m2` (0 keeps them), drop fields larger than `max_m2`
    when that value is above 0, and optionally remove holes.
    """
    empty = gpd.GeoDataFrame(
        {"field_id": [], "area_m2": [], "perimeter_m": [], "score": []}, geometry=[], crs="EPSG:4326")
    if fields is None or not len(fields):
        return empty, lines_from_fields(empty)
    utm = utm_crs(tuple(fields.total_bounds))
    metric = fields.to_crs(utm)
    kept, scores = [], []
    score_col = metric["score"] if "score" in metric.columns else [float("nan")] * len(metric)
    for geom, score in zip(metric.geometry, score_col):
        geom = shapely.make_valid(geom)
        if geom is None or geom.is_empty:
            continue
        if float(simplify_m) > 0:
            geom = geom.simplify(float(simplify_m), preserve_topology=True)
            geom = shapely.make_valid(geom)
        for part in iter_polygons(geom):
            if close_interiors and len(part.interiors):
                part = Polygon(part.exterior.coords)
            if part.is_empty:
                continue
            area = float(part.area)
            if float(min_m2) > 0 and area < float(min_m2):
                continue
            if float(max_m2) > 0 and area > float(max_m2):
                continue
            kept.append(part)
            scores.append(float(score) if score == score else float("nan"))
    if not kept:
        return empty, lines_from_fields(empty)
    gu = gpd.GeoDataFrame({"score": scores}, geometry=kept, crs=utm)
    gu["area_m2"] = gu.area.round(1)
    gu["perimeter_m"] = gu.length.round(1)
    gu = gu.reset_index(drop=True)
    gu["field_id"] = np.arange(1, len(gu) + 1)
    out = gu[["field_id", "area_m2", "perimeter_m", "score", "geometry"]].to_crs(4326)
    return out, lines_from_fields(out)

def capture(west, south, east, north, zoom, source):
    """Stitch basemap tiles covering the bbox. Returns (RGB array, Affine in EPSG:3857, missing-tile count)."""
    x0, y0 = _global_px(west, north, zoom); x1, y1 = _global_px(east, south, zoom)
    x0, y0, x1, y1 = int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))
    tx0, ty0, tx1, ty1 = x0 // TILE, y0 // TILE, (x1 - 1) // TILE, (y1 - 1) // TILE
    jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        tiles = list(ex.map(lambda t: _fetch(SOURCES[source].format(x=t[0], y=t[1], z=zoom)), jobs))
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
    for (tx, ty), im in zip(jobs, tiles):
        if im is not None:
            mosaic.paste(im, ((tx - tx0) * TILE, (ty - ty0) * TILE))
    ox, oy = x0 - tx0 * TILE, y0 - ty0 * TILE
    arr = np.asarray(mosaic.crop((ox, oy, ox + (x1 - x0), oy + (y1 - y0))))
    w, n = _px_to_3857(x0, y0, zoom); e, s = _px_to_3857(x1, y1, zoom)
    return arr, from_bounds(w, s, e, n, arr.shape[1], arr.shape[0]), sum(t is None for t in tiles)


def geocode_location(query, limit=5):
    """Look up a place name with the OpenStreetMap Nominatim geocoder.

    Returns a list of {name, lat, lon, zoom}. The caller must show
    "Geocoding provided © OpenStreetMap" and link to openstreetmap.org.
    """
    query = " ".join((query or "").split())
    if not query:
        return []
    from urllib.parse import urlencode
    url = "https://nominatim.openstreetmap.org/search?" + urlencode({
        "q": query,
        "format": "jsonv2",
        "limit": int(limit),
    })
    req = Request(url, headers={
        "User-Agent": "FieldEdgeDetector/1.0 (agricultural field-boundary research)",
        "Accept": "application/json",
        "Accept-Language": "en",
    })
    try:
        with urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode())
    except Exception as exc:
        raise LookupError(f"Location search failed ({exc}).") from exc
    hits = []
    for item in payload if isinstance(payload, list) else []:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
            south, north, west, east = (float(v) for v in item["boundingbox"])
        except (KeyError, TypeError, ValueError):
            continue
        hits.append({
            "name": item.get("display_name") or query,
            "lat": lat,
            "lon": lon,
            "zoom": _zoom_for_span(south, north, west, east),
        })
    return hits


def _zoom_for_span(south, north, west, east):
    """Web-mercator zoom that fits this latitude/longitude box in the map."""
    lat = (south + north) / 2.0
    height_m = abs(north - south) * 110570.0
    width_m = abs(east - west) * 111320.0 * max(math.cos(math.radians(lat)), 0.01)
    span = max(height_m, width_m, 80.0)
    zoom = math.log2(40075016.686 * max(math.cos(math.radians(lat)), 0.01) / span)
    return int(min(17, max(4, round(zoom))))


def _s2_json(url, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"User-Agent": "field-edge research app"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers)
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _s2_cover(feat, rect):
    geom = feat.get("geometry")
    if geom:
        foot = shape(geom)
    elif feat.get("bbox"):
        foot = box(*feat["bbox"])
    else:
        return 0.0
    if foot.is_empty or rect.is_empty or rect.area <= 0:
        return 0.0
    return float(foot.intersection(rect).area / rect.area)


def capture_sentinel2(west, south, east, north, start, end, max_cloud=30.0, colour=S2_FALSE):
    """Clearest Sentinel-2 L2A scene covering the box.

    `colour` is `S2_FALSE` (near-infrared, red, green), `S2_TRUE` (the visual asset),
    or `S2_RGBNIR` (the true-colour scene, read as red, green, blue, and near-infrared).
    Returns (RGB uint8, Affine in EPSG:3857, missing-tile count, scene).
    The mosaic is 10 m on the ground, in the same Web Mercator frame as the basemap captures.
    `scene` has id, date, cloud (percent, for the whole satellite tile), and colour.
    """
    if colour not in (S2_FALSE, S2_TRUE, S2_RGBNIR):
        raise ValueError(f"Unknown Sentinel-2 colour {colour!r}.")
    if start > end:
        start, end = end, start
    rect = box(west, south, east, north)
    try:
        found = _s2_json(S2_STAC, {
            "collections": ["sentinel-2-l2a"],
            "bbox": [west, south, east, north],
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": float(max_cloud)}},
            "limit": 100,
        })
    except Exception as exc:
        raise LookupError(f"Sentinel-2 search failed ({exc}).") from exc
    feats = [f for f in found.get("features") or [] if _s2_hrefs(f, colour)]
    feats = [f for f in feats if _s2_cover(f, rect) >= 0.5]
    if not feats:
        raise LookupError(
            f"No Sentinel-2 scene with cloud cover under {max_cloud:.0f}% between {start} and {end} "
            "covers this rectangle. Widen the dates or raise the cloud limit."
        )
    feats.sort(key=lambda f: f.get("properties", {}).get("datetime") or "", reverse=True)
    feats.sort(key=lambda f: (-round(_s2_cover(f, rect), 2),
                            float(f.get("properties", {}).get("eo:cloud_cover", 100))))
    errors = []
    for feat in feats[:4]:
        props = feat.get("properties") or {}
        try:
            signed = [_sign_href(href) for href in _s2_hrefs(feat, colour)]
            if colour == S2_FALSE:
                rgb, transform = _read_sentinel2_false(signed, west, south, east, north)
            elif colour == S2_RGBNIR:
                rgb, transform = _read_sentinel2_rgbnir(signed, west, south, east, north)
            else:
                rgb, transform = _read_sentinel2(signed[0], west, south, east, north)
        except Exception as exc:
            errors.append(str(exc))
            continue
        date = (props.get("datetime") or "")[:10]
        return rgb, transform, 0, {
            "id": feat.get("id", ""),
            "date": date,
            "cloud": float(props.get("eo:cloud_cover", float("nan"))),
            "colour": colour,
        }
    detail = errors[-1] if errors else "no readable scene"
    raise LookupError(f"Sentinel-2 scenes were found, but the image could not be read ({detail}).")


def _s2_hrefs(feat, colour):
    """Asset URLs for this colour, or None when the scene does not have them.

    RGB + NIR uses a scene that also has the true-colour visual asset, then reads
    red, green, blue, and near-infrared from that scene.
    """
    assets = feat.get("assets") or {}
    if colour == S2_FALSE:
        keys = S2_FALSE_BANDS
    elif colour == S2_RGBNIR:
        if not (assets.get("visual") or {}).get("href"):
            return None
        keys = S2_RGBNIR_BANDS
    else:
        keys = ("visual",)
    hrefs = []
    for key in keys:
        href = (assets.get(key) or {}).get("href")
        if not href:
            return None
        hrefs.append(href)
    return hrefs


def _sign_href(href):
    from urllib.parse import urlencode
    return _s2_json(S2_SIGN + "?" + urlencode({"href": href}))["href"]


def _mercator_10m(west, south, east, north):
    """EPSG:3857 grid whose pixels are 10 m on the ground at this latitude."""
    from rasterio.warp import transform_bounds

    lat = (south + north) / 2
    res = 10.0 / max(math.cos(math.radians(lat)), 1e-6)
    w, s, e, n = transform_bounds("EPSG:4326", "EPSG:3857", west, south, east, north, densify_pts=21)
    width = max(1, int(round((e - w) / res)))
    height = max(1, int(round((n - s) / res)))
    return from_bounds(w, s, e, n, width, height), width, height


def _read_sentinel2(href, west, south, east, north):
    """Window the true-colour COG and warp it to EPSG:3857 at 10 m on the ground."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window
    from rasterio.windows import from_bounds as window_from_bounds

    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    with rasterio.open(href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
        window = window_from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width < 1 or window.height < 1:
            raise LookupError("the scene does not cover this rectangle")
        data = src.read(window=window)
        src_transform = src.window_transform(window)
        src_crs = src.crs
    dst_transform, width, height = _mercator_10m(west, south, east, north)
    out = np.zeros((data.shape[0], height, width), np.uint8)
    reproject(data, out, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=Resampling.bilinear)
    return np.moveaxis(out, 0, -1), dst_transform


def _read_sentinel2_false(hrefs, west, south, east, north):
    """Window B08, B04, and B03, warp to 10 m, and composite near-infrared, red, green."""
    refl, transform = _warp_reflectance_bands(hrefs, west, south, east, north)
    return reflectance_to_uint8(refl), transform


def _read_sentinel2_rgbnir(hrefs, west, south, east, north):
    """Window red, green, blue, and near-infrared and build the three-channel SAM picture."""
    refl, transform = _warp_reflectance_bands(hrefs, west, south, east, north)
    return rgbnir_for_sam(refl), transform


def _warp_reflectance_bands(hrefs, west, south, east, north):
    """Window each band COG and warp it to the shared 10 m EPSG:3857 grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window
    from rasterio.windows import from_bounds as window_from_bounds

    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    dst_transform, width, height = _mercator_10m(west, south, east, north)
    refl = np.zeros((height, width, len(hrefs)), np.float32)
    for channel, href in enumerate(hrefs):
        with rasterio.open(href) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
            window = window_from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
            if window.width < 1 or window.height < 1:
                raise LookupError("the scene does not cover this rectangle")
            data = src.read(1, window=window)
            src_transform = src.window_transform(window)
            src_crs = src.crs
            scale = _reflectance_scale(src, data)
        band = np.zeros((height, width), np.float32)
        reproject(data.astype(np.float32) * scale, band,
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs="EPSG:3857",
                  resampling=Resampling.bilinear)
        refl[:, :, channel] = band
    return refl, dst_transform


def rgbnir_for_sam(refl):
    """Fold red, green, blue, and near-infrared into the three channels SAM 2.1 reads.

    `refl` is height × width × 4 in band order B04, B03, B02, B08.
    The picture is near-infrared, red, and the average of green and blue.
    """
    refl = np.asarray(refl, np.float32)
    if refl.ndim != 3 or refl.shape[2] != 4:
        raise ValueError("Expected red, green, blue, and near-infrared reflectance.")
    red, green, blue, nir = (refl[:, :, i] for i in range(4))
    picture = np.stack([nir, red, 0.5 * (green + blue)], axis=-1)
    return reflectance_to_uint8(picture)


def _reflectance_scale(src, data):
    """Multiply integer Sentinel-2 digital numbers into reflectance (0–1)."""
    scale = 1.0
    if src.scales and src.scales[0] not in (None, 1, 1.0):
        scale = float(src.scales[0])
    elif np.issubdtype(np.dtype(data.dtype), np.integer):
        scale = 0.0001
    return scale


def reflectance_to_uint8(refl):
    """Stretch a near-infrared, red, green reflectance image into an 8-bit picture.

    Each band uses its own 2–98 percentile on pixels that have data. The stretch
    span is at least 0.02 reflectance so a flat field is not turned into noise.
    A mild gamma keeps crops bright without clipping the field edges.
    """
    refl = np.asarray(refl, np.float32)
    out = np.zeros(refl.shape, np.uint8)
    if refl.ndim != 3 or refl.shape[2] != 3:
        raise ValueError("Expected a height × width × 3 reflectance image.")
    valid = np.any(np.isfinite(refl) & (refl > 0), axis=2)
    if int(valid.sum()) < 16:
        return out
    for channel in range(3):
        values = refl[:, :, channel][valid]
        values = values[np.isfinite(values)]
        if values.size < 16:
            continue
        lo, hi = np.percentile(values, (2, 98))
        span = max(float(hi - lo), 0.02)
        stretched = np.clip((refl[:, :, channel] - float(lo)) / span, 0, 1)
        stretched = np.power(stretched, 1.0 / 1.5)
        picture = (stretched * 255.0).astype(np.uint8)
        picture[~valid] = 0
        out[:, :, channel] = picture
    return out

def placeholder_fraction(rgb, block=32):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = (g.shape[0] // block) * block, (g.shape[1] // block) * block
    if h == 0 or w == 0:
        return 0.0
    return float((g[:h, :w].reshape(h // block, block, w // block, block).std(axis=(1, 3)) < 1.5).mean())

def view_rgb(rgb, brightness):
    """Display copy of an RGB image. `brightness` 1 returns the same array.

    The returned image is for the viewer only. Detection and GeoTIFF export keep the original.
    """
    if rgb is None or abs(float(brightness) - 1.0) < 1e-6:
        return rgb
    scaled = np.asarray(rgb, np.float32) * float(brightness)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def sharpen_rgb(rgb, amount, sigma=1.0):
    """Unsharp mask on the image's own pixel grid. `amount` 0 returns the same array.

    For Sentinel-2 this runs on the 10 m scene, before the resample to 0.25 m, so the mask
    strengthens edges that are already about one pixel wide and does not invent finer detail.
    """
    if rgb is None or rgb.size == 0 or amount <= 0:
        return rgb
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return cv2.addWeighted(rgb, 1.0 + float(amount), blur, -float(amount), 0)


def to_training_resolution(rgb, transform, lat, gsd=GSD):
    """Resample so one pixel ≈ `gsd` metres on the ground (Web-Mercator pixels shrink by cos(lat))."""
    ground = transform.a * math.cos(math.radians(lat))
    f = ground / gsd
    H, W = int(round(rgb.shape[0] * f)), int(round(rgb.shape[1] * f))
    out = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    tr = transform * Affine.scale(rgb.shape[1] / W, rgb.shape[0] / H)
    return out, tr, ground


# ── models ───────────────────────────────────────────────────────────────
def da_weights_path():
    """Local fine-tuned weight if present, otherwise the copy published with the online dashboard."""
    for path in (DA_FINETUNED, DA_FINETUNED_LEGACY, os.path.join(HERE, "weights", "best.pt")):
        if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
            return path
    dest = os.path.join(HERE, "weights", "best.pt")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = Request(DA_URL, headers={"User-Agent": "field-edge-dashboard"})
    try:
        with urlopen(req, timeout=180) as r, open(dest + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as exc:
        part = dest + ".part"
        if os.path.isfile(part):
            os.remove(part)
        raise FileNotFoundError(
            "Delineate Anything weights were not found locally, and the download failed "
            f"({exc}). Expected {DA_FINETUNED}."
        ) from exc
    os.replace(dest + ".part", dest)
    return dest


def load_delineate_anything():
    from ultralytics import YOLO
    return YOLO(da_weights_path()).to(DEVICE)

def load_sam2():
    from sam2.build_sam import build_sam2_hf
    return build_sam2_hf(SAM2_ID, device=DEVICE)

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

def _windows(n, size, overlap):
    if n <= size:
        return [0]
    st = size - overlap
    return sorted(set(list(range(0, n - size + 1, st)) + [n - size]))

def detect_delineate_anything(model, img, patch=768, conf=0.25, progress=None):
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(DEVICE)
    H, W = img.shape[:2]; pad = patch // 4
    # small images: shrink the patch so it fits (multiple of 32)
    patch = min(patch, (min(H, W) // 32) * 32) if min(H, W) < patch else patch
    pad = patch // 4
    rows, cols = _windows(H, patch, 2 * pad), _windows(W, patch, 2 * pad)
    polys, sc, k, total = [], [], 0, len(rows) * len(cols)
    for r0 in rows:
        for c0 in cols:
            p = x[:, r0:r0 + patch, c0:c0 + patch][None]
            res = model.predict(p, conf=conf, iou=0.3, max_det=300, verbose=False, imgsz=patch)[0]
            if res.masks is not None:
                for seg, s in zip(res.masks.xyn, res.boxes.conf.cpu().numpy()):
                    if len(seg) >= 3:
                        polys.append(Polygon([(c0 + u * patch, r0 + v * patch) for u, v in seg])); sc.append(float(s))
            k += 1
            if progress:
                progress(k / total, f"Delineate Anything: window {k}/{total}")
    return nms_partition(polys, sc)

def detect_sam2(model, img, window=1024, overlap=192, progress=None, max_area_frac=0.25, **thr):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    gen = SAM2AutomaticMaskGenerator(model, points_per_side=64, points_per_batch=128, box_nms_thresh=0.70,
                                     crop_n_layers=1, crop_n_points_downscale_factor=2,
                                     min_mask_region_area=MIN_PX, use_m2m=True, **{**SAM2_SETTINGS, **thr})
    H, W = img.shape[:2]
    rows, cols = _windows(H, window, overlap), _windows(W, window, overlap)
    polys, sc, k, total = [], [], 0, len(rows) * len(cols)
    for r0 in rows:
        for c0 in cols:
            sub = np.ascontiguousarray(img[r0:r0 + window, c0:c0 + window])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                ms = gen.generate(sub)
            limit = max_area_frac * sub.shape[0] * sub.shape[1]
            for m in ms:
                if not (MIN_PX <= m["area"] <= limit):
                    continue
                cs, _ = cv2.findContours(m["segmentation"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cs:
                    continue
                c = max(cs, key=cv2.contourArea).reshape(-1, 2)
                if len(c) >= 3:
                    polys.append(Polygon(c + [c0, r0])); sc.append(float(m["predicted_iou"]))
            del ms; torch.cuda.empty_cache()
            k += 1
            if progress:
                progress(k / total, f"SAM 2: window {k}/{total}")
    return nms_partition(polys, sc)


# ── FC-DenseNet (models/fc_densenet.py, trained in the labelled-data notebook) ─
FCDN_DIR = os.path.join(HERE, "..", "Labeled_Data", "train_outputs", "fcdn")
_REPO = os.path.abspath(os.path.join(HERE, ".."))


def fcdn_selected():
    """Depth and threshold written after training. None until that file exists."""
    path = os.path.join(FCDN_DIR, "selected.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_fc_densenet():
    """Load the validation-selected FC-DenseNet onto the GPU."""
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from models.fc_densenet import FCDenseNet

    chosen = fcdn_selected()
    path = None
    if chosen and os.path.isfile(os.path.join(FCDN_DIR, f"{chosen['preset']}.pt")):
        path = os.path.join(FCDN_DIR, f"{chosen['preset']}.pt")
    elif os.path.isdir(FCDN_DIR):
        ranked = []
        for name in os.listdir(FCDN_DIR):
            if name.endswith(".pt"):
                ranked.append(os.path.join(FCDN_DIR, name))
        path = ranked[0] if ranked else None
    if path is None:
        raise FileNotFoundError(
            "No trained FC-DenseNet checkpoint in Labeled_Data/train_outputs/fcdn. "
            "Run Part 6b of field_boundary_training.ipynb first."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    preset = ckpt.get("preset") or ckpt.get("name") or "FC-DenseNet103"
    model = FCDenseNet(preset_model=preset, num_classes=int(ckpt.get("num_classes", 2)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    model.fcdn_threshold = float(ckpt.get("threshold", 0.5))
    model.fcdn_preset = preset
    model.fcdn_valid_f1 = ckpt.get("valid_f1")
    return model


def _fcdn_forward(model, img):
    """Boundary probability for one tile. img is RGB uint8."""
    x = torch.from_numpy(np.ascontiguousarray(img.astype(np.float32) / 255.0)).permute(2, 0, 1)[None].to(DEVICE)
    height, width = x.shape[-2:]
    pad_h, pad_w = (32 - height % 32) % 32, (32 - width % 32) % 32
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    logits = model(x)
    if logits.shape[-2:] != x.shape[-2:]:
        logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return torch.softmax(logits, dim=1)[0, 1, :height, :width]


def fcdn_probability(model, img, progress=None):
    """Boundary probability for a 0.25 m mosaic. Large images are averaged over 512 px windows."""
    height, width = img.shape[:2]
    if height <= 1280 and width <= 1280:
        if progress:
            progress(1.0, "FC-DenseNet")
        with torch.inference_mode():
            return _fcdn_forward(model, img).float().cpu().numpy()
    tile, overlap = 512, 128
    tile_h, tile_w = min(tile, height), min(tile, width)
    rows, cols = _windows(height, tile_h, overlap), _windows(width, tile_w, overlap)
    total = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    step, n_win = 0, len(rows) * len(cols)
    with torch.inference_mode():
        for r0 in rows:
            for c0 in cols:
                prob = _fcdn_forward(model, img[r0:r0 + tile_h, c0:c0 + tile_w]).float().cpu().numpy()
                total[r0:r0 + prob.shape[0], c0:c0 + prob.shape[1]] += prob
                count[r0:r0 + prob.shape[0], c0:c0 + prob.shape[1]] += 1
                step += 1
                if progress:
                    progress(step / n_win, f"FC-DenseNet: window {step}/{n_win}")
    return total / np.maximum(count, 1)


def instances_from_boundary(prob, thr):
    """Enclosed regions of a boundary probability map, same rule as the training reference."""
    barrier = ndi.binary_dilation(prob >= thr, iterations=1)
    barrier[0, :] = True
    barrier[-1, :] = True
    barrier[:, 0] = True
    barrier[:, -1] = True
    inst, _ = ndi.label(~barrier)
    size = np.bincount(inst.ravel())
    keep = size >= MIN_PX
    keep[0] = False
    inst = np.where(keep[inst], inst, 0)
    _, inverse = np.unique(inst, return_inverse=True)
    return inverse.reshape(inst.shape).astype(np.int32)


def detect_fc_densenet(model, img, thr=None, progress=None):
    """Return field polygons in pixel coordinates and a confidence for each."""
    if thr is None:
        thr = float(getattr(model, "fcdn_threshold", 0.5))
    prob = fcdn_probability(model, img, progress=progress)
    labels = instances_from_boundary(prob, thr)
    polys, scores = [], []
    if int(labels.max()) == 0:
        return polys, scores
    for geom, val in features.shapes(labels, mask=labels > 0, transform=Affine.identity(), connectivity=4):
        polygon = shape(geom)
        if polygon.is_empty or polygon.area < MIN_PX:
            continue
        polys.append(polygon)
        scores.append(float(np.clip(1.0 - prob[labels == int(val)].mean(), 0, 1)))
    return polys, scores


# ── Steger 1998 lines and closed snakes (training notebook, section 7.2) ─
# Dark bund centres, then each predicted outline may move at most 0.75 m along its
# normal onto a centre. Interior lines are returned and are not used to split fields.
SIGMAS = (1.0, 1.6, 2.2)
SNAKE_SEARCH_PX = 3          # 3 × 0.25 m = 0.75 m


def gray_world(rgb):
    f = rgb.astype(np.float32)
    m = f.reshape(-1, 3).mean(0) + 1e-6
    return np.clip(f * (m.mean() / m), 0, 255)


def percentile_stretch(f, lo=1, hi=99):
    out = np.empty_like(f)
    for c in range(3):
        p1, p2 = np.percentile(f[..., c], (lo, hi))
        out[..., c] = np.clip((f[..., c] - p1) / (p2 - p1 + 1e-6), 0, 1)
    return out


def sigmoid_luminance(rgb01, gain=5.0):
    yuv = cv2.cvtColor(rgb01.astype(np.float32), cv2.COLOR_RGB2YUV)
    Y = yuv[..., 0]
    hist, edges = np.histogram(Y, 256, (0, 1))
    hist = ndi.gaussian_filter1d(hist.astype(float), 4)
    pk, _ = find_peaks(hist, distance=20, prominence=hist.max() * 0.05)
    x0 = edges[int(pk[np.argsort(hist[pk])[-2:]].mean())] if len(pk) >= 2 else float(np.median(Y))
    ys = 1 / (1 + np.exp(-gain * (Y - x0)))
    ys = (ys - ys.min()) / (np.ptp(ys) + 1e-6)
    yuv[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply((ys * 255).astype(np.uint8)) / 255.0
    return np.clip(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB), 0, 1)


def preprocess(rgb):
    """Same luminance preparation the training study used before Steger extraction."""
    u8 = (percentile_stretch(gray_world(rgb)) * 255).astype(np.uint8)
    for _ in range(2):
        u8 = cv2.bilateralFilter(u8, 9, 25, 5)
    return sigmoid_luminance(u8 / 255.0)


def steger_response(lum, sigma):
    """Scale-normalised Gaussian derivatives. t is the sub-pixel offset along the normal."""
    s2 = sigma ** 2
    Ix = ndi.gaussian_filter(lum, sigma, order=(0, 1)) * sigma
    Iy = ndi.gaussian_filter(lum, sigma, order=(1, 0)) * sigma
    Ixx = ndi.gaussian_filter(lum, sigma, order=(0, 2)) * s2
    Ixy = ndi.gaussian_filter(lum, sigma, order=(1, 1)) * s2
    Iyy = ndi.gaussian_filter(lum, sigma, order=(2, 0)) * s2
    diff = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy ** 2)
    lam_hi = 0.5 * (Ixx + Iyy + diff)
    lam_lo = 0.5 * (Ixx + Iyy - diff)
    lam = np.where(np.abs(lam_hi) >= np.abs(lam_lo), lam_hi, lam_lo)
    nx = np.where(np.abs(Ixy) < 1e-8, np.where(np.abs(Ixx) >= np.abs(Iyy), 1.0, 0.0), Ixy)
    ny = np.where(np.abs(Ixy) < 1e-8, np.where(np.abs(Ixx) >= np.abs(Iyy), 0.0, 1.0), lam - Ixx)
    nrm = np.hypot(nx, ny) + 1e-12
    nx, ny = nx / nrm, ny / nrm
    second = nx ** 2 * Ixx + 2 * nx * ny * Ixy + ny ** 2 * Iyy
    t = -(nx * Ix + ny * Iy) / (second + 1e-12)
    return lam.astype(np.float32), t.astype(np.float32), nx.astype(np.float32), ny.astype(np.float32)


def steger_lines(lum, high_q=88, low_q=68, min_px=28):
    """Link dark Steger centres into polylines. Returns (lines, response magnitude)."""
    best = None
    for sigma in SIGMAS:
        lam, t, nx, ny = steger_response(lum, sigma)
        centre = (np.abs(t) <= 0.5) & (lam > 0)
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
        return [], mag.astype(np.float32)
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


def lateral_de(cielab, curve, valid, radius=8):
    if len(curve) < 5:
        return 0.0
    step = max(1, len(curve) // 40)
    pts = curve[::step]
    dlt = np.diff(curve, axis=0)
    dlt = np.vstack([dlt, dlt[-1]])[::step]
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


def snap_outlines(labels, mag, valid, search=SNAKE_SEARCH_PX):
    """Move each outline onto a nearby Steger centre. Keep the field partition."""
    positive = mag[mag > 0]
    if positive.size < 50:
        return labels, 0
    thr = float(np.percentile(positive, 80))
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
        r0 = max(int(np.floor(pts[:, 0].min())) - 2, 0)
        c0 = max(int(np.floor(pts[:, 1].min())) - 2, 0)
        r1 = min(int(np.ceil(pts[:, 0].max())) + 3, H)
        c1 = min(int(np.ceil(pts[:, 1].max())) + 3, W)
        local = np.zeros((r1 - r0, c1 - c0), bool)
        rr, cc = draw_polygon(pts[:, 0] - r0, pts[:, 1] - c0, local.shape)
        local[rr, cc] = True
        new = np.zeros(labels.shape, bool)
        new[r0:r1, c0:c1] = local
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


def candidate_lines(labels, lines, cielab, valid):
    """Interior dark lines that look like missing boundaries. Not used to split parcels."""
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


def refine_outlines(img, labels, progress=None):
    """Snap predicted outlines onto Steger centres and list interior candidate lines."""
    if int(labels.max()) == 0:
        return labels, 0, []
    if progress:
        progress(0.05, "Preparing luminance for Steger line extraction…")
    pre = preprocess(img)
    cielab = rgb2lab(np.clip(pre, 0, 1))
    lum = cielab[..., 0] / 100.0
    valid = img.sum(2) > 0
    if progress:
        progress(0.25, "Extracting dark line centres (Steger 1998)…")
    lines, mag = steger_lines(lum)
    if progress:
        progress(0.7, "Snapping outlines onto bund centres…")
    refined, n_moved = snap_outlines(labels, mag, valid)
    if progress:
        progress(0.9, "Listing interior Steger lines…")
    cands = candidate_lines(labels, lines, cielab, valid)
    return refined, n_moved, cands


def curves_to_gdf(curves, transform):
    """Pixel polylines → WGS84 lines. Length is measured at the 0.25 m working grid."""
    if not len(curves):
        return gpd.GeoDataFrame({"length_m": []}, geometry=[], crs="EPSG:4326")
    rows = []
    for curve in curves:
        xy = [transform * (float(c), float(r)) for r, c in curve]
        length_m = float(np.hypot(*np.diff(curve, axis=0).T).sum()) * GSD
        rows.append({"length_m": round(length_m, 1), "geometry": LineString(xy)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857").to_crs(4326)


# ── vector output ────────────────────────────────────────────────────────
def iter_polygons(g):
    """Yield every Polygon inside a (Multi)Polygon or GeometryCollection."""
    if g.geom_type == "Polygon":
        yield g
    elif hasattr(g, "geoms"):
        for part in g.geoms:
            yield from iter_polygons(part)

def polygons_to_labels(polys, shape_hw):
    """Rasterise pixel-space polygons to an instance label image. Label i+1 is polygon i."""
    from shapely.geometry import MultiPolygon
    shapes_ = [(MultiPolygon(list(iter_polygons(p))), i + 1) for i, p in enumerate(polys) if any(True for _ in iter_polygons(p))]
    if not shapes_:
        return np.zeros(shape_hw, np.int32)
    return features.rasterize(shapes_, out_shape=shape_hw, transform=Affine.identity(), fill=0, dtype="int32")


def to_vectors(polys, scores, transform, shape_hw, clip_box_4326=None, labels=None):
    """Pixel-space polygons → field polygons and boundary lines in EPSG:4326.

    Pass `labels` to vectorise a refined instance raster (Steger snake) instead of the raw masks.
    Label ids still index `scores`.
    """
    if labels is None and not polys:
        empty = gpd.GeoDataFrame({"field_id": [], "area_m2": [], "perimeter_m": [], "score": []}, geometry=[], crs="EPSG:4326")
        return empty, empty.copy()
    lab = polygons_to_labels(polys, shape_hw) if labels is None else np.asarray(labels, np.int32)
    if int(lab.max()) == 0:
        empty = gpd.GeoDataFrame({"field_id": [], "area_m2": [], "perimeter_m": [], "score": []}, geometry=[], crs="EPSG:4326")
        return empty, empty.copy()
    geoms = [(shape(g), int(v)) for g, v in features.shapes(lab, mask=lab > 0, transform=transform, connectivity=4)]
    g = gpd.GeoDataFrame({"pid": [v for _, v in geoms]}, geometry=[x for x, _ in geoms], crs="EPSG:3857")
    g = g.dissolve("pid").reset_index().explode(index_parts=False).reset_index(drop=True)
    lat = g.to_crs(4326).union_all().centroid.y
    utm = 32600 + int((g.to_crs(4326).union_all().centroid.x + 180) // 6) + 1 if lat >= 0 else 32700
    gu = g.to_crs(utm)
    gu["geometry"] = gu.geometry.buffer(0).simplify(0.5, preserve_topology=True)
    gu = gu[gu.area >= MIN_FIELD_M2].copy()
    if clip_box_4326 is not None:
        gu["geometry"] = gu.geometry.intersection(gpd.GeoSeries([clip_box_4326], crs=4326).to_crs(utm).iloc[0])
        gu = gu[~gu.geometry.is_empty & (gu.area >= MIN_FIELD_M2)]
    gu["score"] = [float(scores[int(p) - 1]) if scores is not None and 0 < int(p) <= len(scores) else np.nan for p in gu.pid]
    gu["area_m2"] = gu.area.round(1); gu["perimeter_m"] = gu.length.round(1)
    gu = gu.reset_index(drop=True); gu["field_id"] = np.arange(1, len(gu) + 1)
    fields = gu[["field_id", "area_m2", "perimeter_m", "score", "geometry"]].to_crs(4326)
    lines = gpd.GeoDataFrame(fields[["field_id"]].copy(), geometry=gu.geometry.boundary.values, crs=utm).to_crs(4326)
    return fields, lines


def _perp_dist(a, b, c):
    """Distance in metres from point b to the straight line through a and c."""
    dx, dy = c[0] - a[0], c[1] - a[1]
    den = math.hypot(dx, dy)
    if den < 1e-9:
        return math.hypot(b[0] - a[0], b[1] - a[1])
    return abs(dx * (a[1] - b[1]) - (a[0] - b[0]) * dy) / den


def _dedupe_ring(pts, gap=0.05):
    out = []
    for x, y in pts:
        if not out or math.hypot(x - out[-1][0], y - out[-1][1]) > gap:
            out.append((float(x), float(y)))
    if len(out) >= 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= gap:
        out.pop()
    return out


def _drop_bends(pts, tol):
    """Drop vertices that sit within `tol` metres of the straight side through their neighbours.

    A vertex on an edge shorter than `tol` is dropped too, so a stair-step collapses into one line.
    A sharp corner between two long sides stays.
    """
    pts = _dedupe_ring(pts)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        n = len(pts)
        best_i, best_d = None, tol
        for i in range(n):
            a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            shorter = min(math.hypot(b[0] - a[0], b[1] - a[1]), math.hypot(c[0] - b[0], c[1] - b[1]))
            dist = _perp_dist(a, b, c)
            score = dist if shorter >= tol else min(dist, shorter)
            if score <= best_d:
                best_d, best_i = score, i
        if best_i is None or best_d > tol:
            break
        del pts[best_i]
        pts = _dedupe_ring(pts)
        changed = True
    return pts


def _straight_polygon(poly, tol):
    """One polygon in a metric CRS, with each side replaced by a straight segment."""
    poly = shapely.make_valid(poly)
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon":
        return None
    simple = poly.simplify(tol, preserve_topology=True)
    if simple.is_empty:
        return None
    if simple.geom_type == "MultiPolygon":
        simple = max(simple.geoms, key=lambda g: g.area)
    if simple.geom_type != "Polygon":
        return None
    ext = _drop_bends(list(simple.exterior.coords), tol)
    if len(ext) < 3:
        return simple if simple.area >= MIN_FIELD_M2 else None
    holes = []
    for ring in simple.interiors:
        hole = _drop_bends(list(ring.coords), tol)
        if len(hole) >= 3 and Polygon(hole).area >= MIN_FIELD_M2:
            holes.append(hole)
    out = Polygon(ext, holes)
    if not out.is_valid:
        out = shapely.make_valid(out)
        if out.geom_type == "MultiPolygon":
            out = max((g for g in out.geoms if g.geom_type == "Polygon"), key=lambda g: g.area, default=None)
        if out is None or out.geom_type != "Polygon":
            return simple
    return out


def _snap_corners(polys, snap_m):
    """Move corners that lie within `snap_m` of each other onto one shared point."""
    if snap_m <= 0 or len(polys) < 2:
        return polys
    rings = []
    pts = []
    for poly in polys:
        if poly is None or poly.is_empty or poly.geom_type != "Polygon":
            rings.append(None)
            continue
        parts = [list(poly.exterior.coords)[:-1]]
        parts += [list(r.coords)[:-1] for r in poly.interiors]
        rings.append(parts)
        for part in parts:
            pts.extend(part)
    if len(pts) < 2:
        return polys
    cell = snap_m
    buckets = {}
    for i, (x, y) in enumerate(pts):
        buckets.setdefault((math.floor(x / cell), math.floor(y / cell)), []).append(i)
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (x, y) in enumerate(pts):
        ix, iy = math.floor(x / cell), math.floor(y / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((ix + dx, iy + dy), ()):
                    if j <= i:
                        continue
                    if math.hypot(pts[j][0] - x, pts[j][1] - y) <= snap_m:
                        union(i, j)
    groups = {}
    for i in range(len(pts)):
        groups.setdefault(find(i), []).append(i)
    moved = {}
    for members in groups.values():
        mx = sum(pts[m][0] for m in members) / len(members)
        my = sum(pts[m][1] for m in members) / len(members)
        for m in members:
            moved[m] = (mx, my)
    cursor = 0
    out = []
    for poly, parts in zip(polys, rings):
        if parts is None:
            out.append(poly)
            continue
        new_parts = []
        for part in parts:
            new_parts.append([moved[cursor + k] for k in range(len(part))])
            cursor += len(part)
        ext = _dedupe_ring(new_parts[0])
        if len(ext) < 3:
            out.append(poly)
            continue
        holes = []
        for part in new_parts[1:]:
            hole = _dedupe_ring(part)
            if len(hole) >= 3:
                holes.append(hole)
        rebuilt = Polygon(ext, holes)
        if not rebuilt.is_valid:
            rebuilt = shapely.make_valid(rebuilt)
        if rebuilt.geom_type == "MultiPolygon":
            rebuilt = max(rebuilt.geoms, key=lambda g: g.area)
        if rebuilt.geom_type != "Polygon" or rebuilt.is_empty:
            out.append(poly)
        else:
            out.append(rebuilt)
    return out


def straighten_fields(fields, tolerance_m=3.0):
    """Clean field outlines: drop bends within `tolerance_m` and make each side a straight line.

    Shared corners of neighbouring fields snap together when they fall within half the tolerance
    (at most 1.5 m). Area and perimeter are recomputed. A field whose area would change by more
    than 35% keeps its original outline.
    """
    empty_lines = gpd.GeoDataFrame({"field_id": []}, geometry=[], crs="EPSG:4326")
    if fields is None or not len(fields):
        empty = fields.iloc[0:0].copy() if fields is not None else gpd.GeoDataFrame(
            {"field_id": [], "area_m2": [], "perimeter_m": [], "score": []}, geometry=[], crs="EPSG:4326")
        return empty, empty_lines
    tol = max(float(tolerance_m), 0.25)
    src = fields.to_crs(4326) if fields.crs is None or fields.crs.to_epsg() != 4326 else fields
    utm = utm_crs(src.union_all().bounds)
    gu = src.to_crs(utm)
    cleaned = []
    for geom in gu.geometry:
        orig = shapely.make_valid(geom)
        if orig.geom_type == "MultiPolygon":
            orig = max(orig.geoms, key=lambda g: g.area)
        new = _straight_polygon(orig, tol) if orig.geom_type == "Polygon" else None
        if new is None or new.is_empty or new.area < MIN_FIELD_M2:
            cleaned.append(orig if orig.geom_type == "Polygon" else geom)
            continue
        if orig.area > 0 and abs(new.area - orig.area) / orig.area > 0.35:
            milder = orig.simplify(tol, preserve_topology=True)
            cleaned.append(milder if milder.geom_type == "Polygon" and milder.area >= MIN_FIELD_M2 else orig)
        else:
            cleaned.append(new)
    snap_m = min(1.5, tol * 0.5)
    snapped = _snap_corners(cleaned, snap_m)
    final = []
    for orig, geom in zip(cleaned, snapped):
        if geom is None or geom.is_empty or geom.geom_type != "Polygon" or geom.area < MIN_FIELD_M2:
            final.append(orig)
        elif orig.area > 0 and abs(geom.area - orig.area) / orig.area > 0.35:
            final.append(orig)
        else:
            final.append(geom)
    gu = gu.copy()
    gu["geometry"] = final
    gu["area_m2"] = gu.area.round(1)
    gu["perimeter_m"] = gu.length.round(1)
    keep = ["field_id", "area_m2", "perimeter_m", "score", "geometry"]
    for col in keep:
        if col not in gu.columns and col != "geometry":
            gu[col] = np.nan
    out_fields = gu[keep].to_crs(4326)
    out_lines = gpd.GeoDataFrame(out_fields[["field_id"]].copy(), geometry=gu.geometry.boundary.values, crs=utm).to_crs(4326)
    return out_fields, out_lines


def outline_preview(img, labels):
    """Draw the refined field outlines on the 0.25 m image."""
    preview = np.array(img, copy=True)
    edge = ndi.binary_dilation(find_boundaries(labels, mode="inner") & (labels > 0), iterations=1)
    preview[edge] = (255, 40, 40)
    return preview


# ── training-data export ─────────────────────────────────────────────────
# Same layout as Labeled_Data/Data/<split>/<site>/{Tiles,Polyline}: an RGB GeoTIFF in a
# metric CRS and a polyline shapefile (column `Id`) of the field boundaries, both named after the tile.
BUTM_2010 = "EPSG:9680"         # WGS 84 / TM 90 NE: the CRS of the labelled training tiles


def utm_crs(bbox):
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"


def _m_per_deg(lat):
    """Meters per degree of longitude and of latitude at `lat`."""
    lon_m = 111320.0 * max(math.cos(math.radians(lat)), 1e-6)
    return lon_m, 110570.0


def export_geotiff(rgb, transform, bbox, crs, gsd=GSD):
    """Warp the captured EPSG:3857 mosaic onto a `gsd` m grid in `crs`, cropped to the box. Returns GeoTIFF bytes.

    Pixels outside the capture are 0, which the training loader treats as no-data.
    A geographic CRS (WGS 84) uses degree pixel sizes that still cover `gsd` metres on the ground.
    """
    from pyproj import CRS
    from rasterio.io import MemoryFile
    from rasterio.warp import Resampling, reproject, transform_bounds
    w, s, e, n = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    if CRS.from_user_input(crs).is_geographic:
        lon_m, lat_m = _m_per_deg((bbox[1] + bbox[3]) / 2)
        res_x, res_y = gsd / lon_m, gsd / lat_m
    else:
        res_x = res_y = gsd
    W, H = int(math.ceil((e - w) / res_x)), int(math.ceil((n - s) / res_y))
    dst_tr = Affine(res_x, 0, w, 0, -res_y, n)
    out = np.zeros((3, H, W), np.uint8)
    reproject(np.ascontiguousarray(rgb.transpose(2, 0, 1)), out, src_transform=transform, src_crs="EPSG:3857",
              dst_transform=dst_tr, dst_crs=crs, resampling=Resampling.cubic if transform.a > gsd else Resampling.average,
              src_nodata=0, dst_nodata=0)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs=crs, transform=dst_tr,
                      photometric="RGB", compress="deflate", tiled=True, blockxsize=256, blockysize=256) as dst:
            dst.write(out)
        return mem.read()


def training_lines(lines, bbox, crs):
    """Field-boundary polylines for retraining: shared edges merged once, the box edge removed, one LineString per row."""
    empty = gpd.GeoDataFrame({"Id": []}, geometry=[], crs=crs)
    if lines is None or not len(lines):
        return empty
    g = lines.to_crs(crs)
    # 0.5 m = the outline simplification tolerance; edges this close to the tile border are masked in training anyway.
    # In a geographic CRS that tolerance is expressed in degrees at the box latitude.
    from pyproj import CRS
    tol = 0.5
    if CRS.from_user_input(crs).is_geographic:
        lon_m, lat_m = _m_per_deg((bbox[1] + bbox[3]) / 2)
        tol = 0.5 / min(lon_m, lat_m)
    frame = gpd.GeoSeries([Polygon.from_bounds(*bbox).exterior], crs=4326).to_crs(crs).iloc[0].buffer(tol)
    merged = shapely.line_merge(shapely.union_all(g.geometry.values).difference(frame))
    parts = [p for p in getattr(merged, "geoms", [merged]) if isinstance(p, LineString) and p.length >= tol]
    if not parts:
        return empty
    return gpd.GeoDataFrame({"Id": np.arange(1, len(parts) + 1, dtype=np.int32)}, geometry=parts, crs=crs)


def geotiff_display(tif_bytes, max_side=2048):
    """RGB view of a GeoTIFF and its EPSG:4326 bounds, for the vertex editor."""
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    with MemoryFile(tif_bytes) as mem, mem.open() as src:
        arr = src.read()
        rgb = np.moveaxis(arr[:3], 0, -1) if arr.shape[0] >= 3 else np.repeat(np.moveaxis(arr[:1], 0, -1), 3, axis=2)
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    height, width = rgb.shape[:2]
    scale = max_side / max(height, width, 1)
    if scale < 1:
        rgb = cv2.resize(rgb, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb), (float(west), float(south), float(east), float(north))


def normalize_drawings(features):
    """LineStrings from the map editor. None until the editor reports a drawing."""
    if features is None:
        return None
    out, seen = [], set()
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        kind, coords = geom.get("type"), geom.get("coordinates")
        if kind == "LineString":
            parts = [coords]
        elif kind == "MultiLineString":
            parts = list(coords or [])
        elif kind == "Polygon" and coords:
            parts = [coords[0]]
        else:
            continue
        for line in parts:
            if not line or len(line) < 2:
                continue
            clean = []
            for x, y in line:
                pt = (round(float(x), 7), round(float(y), 7))
                if not clean or pt != clean[-1]:
                    clean.append(pt)
            if len(clean) < 2 or tuple(clean) in seen:
                continue
            seen.add(tuple(clean))
            out.append({"type": "Feature", "properties": {},
                        "geometry": {"type": "LineString", "coordinates": [[x, y] for x, y in clean]}})
    return out


def features_from_lines(gdf):
    """Editor features (EPSG:4326 LineStrings) for a line layer."""
    if gdf is None or not len(gdf):
        return []
    gg = gdf if getattr(gdf, "crs", None) is not None and gdf.crs.to_epsg() == 4326 else gdf.to_crs(4326)
    feats = []
    for geom in gg.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))
        for part in parts:
            if getattr(part, "geom_type", None) != "LineString":
                continue
            feats.append({"type": "Feature", "properties": {},
                          "geometry": {"type": "LineString", "coordinates": [list(xy) for xy in part.coords]}})
    return normalize_drawings(feats) or []


def lines_from_drawings(features, crs):
    """Edited map lines as a training shapefile layer in `crs`."""
    empty = gpd.GeoDataFrame({"Id": []}, geometry=[], crs=crs)
    norm = normalize_drawings(features) or []
    if not norm:
        return empty
    from pyproj import CRS
    geoms = [shape(feat["geometry"]) for feat in norm]
    gdf = gpd.GeoDataFrame({"Id": np.arange(1, len(geoms) + 1, dtype=np.int32)}, geometry=geoms, crs="EPSG:4326")
    if not CRS.from_user_input(crs).equals(CRS.from_epsg(4326)):
        gdf = gdf.to_crs(crs)
        gdf["Id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    return gdf


def shapefile_bytes(gdf, name):
    """Zip of the shapefile components (<name>.shp/.shx/.dbf/.prj/.cpg)."""
    import tempfile, zipfile
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        gdf.to_file(os.path.join(tmp, f"{name}.shp"), driver="ESRI Shapefile", engine="pyogrio")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in sorted(os.listdir(tmp)):
                z.write(os.path.join(tmp, fn), fn)
    return buf.getvalue()


def training_pair_zip(tif, shp_zip, name):
    """Tiles/<name>.tif + Polyline/<name>.* — drop into Data/<split>/<site>/."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"Tiles/{name}.tif", tif)
        with zipfile.ZipFile(io.BytesIO(shp_zip)) as s:
            for fn in s.namelist():
                z.writestr(f"Polyline/{fn}", s.read(fn))
    return buf.getvalue()


def sample_tiles_zip(tiles, lines, crs, name):
    """Training layout for every sample tile: Tiles/<name>_01.tif and Polyline/<name>_01.*."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, tile in enumerate(tiles, 1):
            tname = f"{name}_{i:02d}"
            tif = export_geotiff(tile["rgb"], tile["transform"], tile["bbox"], crs)
            tlines = training_lines(lines, tile["bbox"], crs)
            shp = shapefile_bytes(tlines, tname)
            z.writestr(f"Tiles/{tname}.tif", tif)
            with zipfile.ZipFile(io.BytesIO(shp)) as s:
                for fn in s.namelist():
                    z.writestr(f"Polyline/{fn}", s.read(fn))
    return buf.getvalue()
