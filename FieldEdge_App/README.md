# Field Edge Detector

The public map of the final Test polygons is at <https://zia207.github.io/field-edge-app/>.
The page source is in `web/`.

The online detector (draw a box, run Delineate Anything, Steger lines and the snake) is the Streamlit Community Cloud app in [CLOUD.md](CLOUD.md). The repository is <https://github.com/zia207/field-edge-dashboard>. You turn the public link on at [share.streamlit.io](https://share.streamlit.io) with main file `streamlit_app.py`.

The Streamlit app below draws a new rectangle and runs the trained models locally, including SAM 2.1.

Draw a rectangle on Google or Esri satellite imagery and detect agricultural field boundaries with the
models trained in `../Labeled_Data/`. The deployed Delineate Anything weights are `weights/best.pt` (Git LFS).

## Run

```bash
cd FieldEdge_App
../.venv/bin/streamlit run app.py
```
Then open http://localhost:8501.

## How it works

1. **Draw** a rectangle with the ▭ tool, or jump to a latitude/longitude first.
2. **Imagery:** tiles for the box are downloaded from the chosen source (Google Satellite by default,
   or Esri World Imagery) at the chosen scale:
   - **20 m scale = zoom 19 (≈ 0.28 m/px):** recommended; closest to the models' training resolution.
   - **50 m scale = zoom 18 (≈ 0.55 m/px):** use where zoom 19 is missing.

   If the source has no zoom-19 imagery (grey "not available" tiles), the app falls back to zoom 18
   automatically.
3. The image is **resampled to 0.25 m/px**, the resolution of the training data.
4. **Detect** with one of three models:

   | Model | Held-out test accuracy | Speed (GPU) | Max box |
   |---|---|---|---|
   | Delineate Anything V2, fine-tuned (default) | field F1 0.64 · boundary F1 0.65 @ 1 m | ≈ 1–7 s per 0.05 km² | 2 km² |
   | SAM 2.1, pre-trained, calibrated | field F1 0.68 · boundary F1 0.77 @ 1 m | ≈ 45 s per 0.05 km² | 0.25 km² |
   | FC-DenseNet | depth and threshold chosen on Valid | a few seconds per 0.05 km² | 1 km² |
5. **Refine the outlines** (on by default), as in section 7.2 of the training notebook. Dark bund
   centres are extracted with Steger (1998). Each outline is a closed snake that may move at most
   0.75 m along its normal onto a centre, and only when the field still overlaps itself (IoU ≥ 0.92)
   and the new outline sits on a stronger line. Interior Steger lines are written out and are not
   used to split parcels.
6. **Results:** the refined field polygons over the imagery, a preview image, a field table, and downloads
   (fields GeoJSON, boundary GeoJSON, Steger-line GeoJSON, GeoPackage, PNG preview).
7. **Training data:** download the selected area and its detected edges for retraining the models:
   - **Image (GeoTIFF):** RGB at 0.25 m, warped from the captured imagery and cropped to the box.
   - **Field edges (Shapefile):** boundary polylines with an `Id` column; shared edges appear once and
     the box edge is removed.
   - **Both, training layout (ZIP):** `Tiles/<name>.tif` + `Polyline/<name>.shp`, the layout of
     `../Labeled_Data/Data/<split>/<site>/`. Unzip it into a new site folder.

   Both are in WGS 84 (EPSG:4326) or the local UTM zone. Turn on **Edit vertices on the GeoTIFF**
   for an ArcGIS-style toolbar: move vertices, add and delete vertices, move a whole line, continue
   from an endpoint, split, merge, reshape, draw a new line, or delete a line. Snapping, undo, and
   redo sit on the same bar. **Finish** writes the edited lines into the shapefile.

`detector.py` holds the processing core (capture, resampling, inference, vectorisation) and can be used
without the UI.

## Notes

- The models were trained on 10 sites in Bangladesh (Google imagery). Accuracy elsewhere may be lower.
- Google's Terms of Service restrict bulk download and reuse of its imagery: use it for research only.
  Esri World Imagery is the safer choice for shared or operational work.
- SAM 2 needs a CUDA GPU in practice. Delineate Anything also runs on CPU, but slowly.
