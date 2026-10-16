# Fieldedge-Detectors: field-boundary detection (FTW tools · SAM 2 · Active Contours)

Compares field-boundary delineation methods on Esri and Google screenshots captured at zoom 17, 18 and 19
over the same footprint at five sites in Bangladesh (Bhola, Faridpur, Patuakhali, Rajbari, Barisal).

| File | What it is |
|---|---|
| `FieldEdge_App/` | **Field Edge Detector app** (Streamlit): draw a box on Google/Esri imagery and detect field boundaries with the trained models. See its README |
| `Labeled_Data/` | **Labelled training data** (local only, not in git): `{Train,Valid,Test}/sw_xx/{Tiles,Polyline}`, 58 Google-imagery tiles with hand-digitised boundary polylines |
| `Updated_analysis_labeled_data/` | **Training study**: trained models and the notebook that trains them on `Labeled_Data/`. See its README |
| `build_notebook.py` | Generates the method-comparison notebook (`field_boundary_comparison.ipynb`) |
| `render.sh` | Renders the comparison notebook to HTML and a code-free PDF |
| `random_reference_points.csv` | Independent, visually interpreted reference points |
| `requirements-lock.txt` | Exact package versions of `.venv` |

Rendered notebooks, reports and the `outputs/` cache are not kept in the repository; re-run the steps below to regenerate them.

## Environment

```bash
cd Fieldedge-Detectors
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python --prerelease=if-necessary \
    "ftw-tools[delineate-anything]==2.0.0b5" "sam2>=1.1" "hydra-core>=1.3.2,<1.4" segment-anything \
    duckdb huggingface_hub iopath opencv-python-headless scikit-learn ipykernel nbconvert jupyter \
    streamlit folium streamlit-folium
# or reproduce exactly:
uv pip install -p .venv/bin/python -r requirements-lock.txt
.venv/bin/python -m ipykernel install --sys-prefix --name fieldedge --display-name "Python 3.12 (FieldEdge)"
```

This one environment runs the local app (`FieldEdge_App/`), the training study and the comparison notebook.
`FieldEdge_App/requirements.txt` is the separate CPU-only set for Streamlit Community Cloud.

- Python 3.12 is required by `ftw-tools` 2.x. `2.0.0b5` is the release with the PRUE models used for the
  FTW Global Map. Only `ftw-tools` is a pre-release; all other packages are stable.
- `hydra-core` must stay below 1.4 for SAM 2.
- GPU: PyTorch 2.10 + CUDA 12.8 wheels (any recent NVIDIA driver).

## Run

```bash
.venv/bin/python build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=fieldedge --ExecutePreprocessor.timeout=-1 field_boundary_comparison.ipynb
./render.sh
```

Steps are cached in `outputs/` (created on first run). Delete a cache file (e.g. `outputs/vectors/sam2.gpkg`) to recompute that method.

Google Maps tiles are subject to Google's Terms of Service, which restrict bulk download and reuse. Use them for research comparison only.

## Training study

See [`Updated_analysis_labeled_data/README.md`](Updated_analysis_labeled_data/README.md).
