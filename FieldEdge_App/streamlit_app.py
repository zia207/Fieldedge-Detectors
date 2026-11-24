"""Entry point for Streamlit Community Cloud.

The public app runs Delineate Anything V2 on CPU, with Steger line extraction and the snake.
SAM 2.1 stays on the local app (app.py), which has a GPU.
"""
import os
import runpy

os.environ["FIELD_EDGE_CLOUD"] = "1"

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), run_name="__main__")
