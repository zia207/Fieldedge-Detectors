# Field Edge Detector — online dashboard

Public Streamlit app. A visitor draws a rectangle on Esri or Google imagery at the 20 m or 50 m scale. The app runs the fine-tuned Delineate Anything V2 model, snaps each outline onto a Steger (1998) line centre with a closed snake, and offers the fields as GeoJSON.

The box is limited to **0.05 km²** because the free host has a CPU and no GPU. SAM 2.1 stays on the local app.

The fine-tuned weights (`weights/best.pt`, 125 MB) are in the repository through Git LFS, which Streamlit
Community Cloud fetches when it builds the app. Each build or reboot downloads them once; GitHub's free plan
includes 1 GB of LFS bandwidth per month.

## Deploy

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub as **zia207**.
   The repository is private: when asked, let Streamlit access your private repositories.
2. Choose **Create app** → **Deploy a public app from GitHub**.
3. Repository `zia207/Fieldedge-Detectors`, branch `main`, main file path `FieldEdge_App/streamlit_app.py`.
4. Optionally set a custom subdomain under **App URL**, for example `field-edge-detector`.
5. Open **Advanced settings** and choose **Python 3.12** (the CPU torch wheel in `requirements.txt` is built for it).
6. **Deploy.** The first build installs PyTorch CPU and pulls `best.pt`; it takes several minutes.

`requirements.txt` in this folder is the cloud environment (Streamlit Cloud uses the file next to the main file).
Keep `pyarrow==17.0.0` while NumPy is pinned below 2: newer pyarrow needs NumPy 2, and the map component needs pyarrow.

The deployed app is public by default even though the repository is private. To restrict it, open the app's
**Settings → Sharing** and invite viewers by email.

## Run the same cloud entry locally

```bash
cd FieldEdge_App
FIELD_EDGE_CLOUD=1 ../.venv/bin/streamlit run streamlit_app.py
```
