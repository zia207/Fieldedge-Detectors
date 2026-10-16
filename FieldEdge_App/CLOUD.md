# Field Edge Detector — online dashboard

Public Streamlit app. A visitor draws a rectangle on Esri or Google imagery at the 20 m or 50 m scale. The app runs the fine-tuned Delineate Anything V2 model, snaps each outline onto a Steger (1998) line centre with a closed snake, and offers the fields as GeoJSON.

The box is limited to **0.05 km²** because the free host has a CPU and no GPU. SAM 2.1 stays on the local app.

Weights (119 MB) are not in git. On first start the app downloads them from the `weights-v1` release of this repository.

## Turn the public link on

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub as **zia207**.
2. Choose **Create app**.
3. Repository `zia207/field-edge-dashboard`, branch `main`, main file path `streamlit_app.py`.
4. Deploy. The first build installs PyTorch CPU and downloads `best.pt`. That can take several minutes.
5. The app URL will look like `https://field-edge-dashboard.streamlit.app`.

## Run the same cloud entry locally

```bash
cd FieldEdge_App
FIELD_EDGE_CLOUD=1 ../.venv/bin/streamlit run streamlit_app.py
```
