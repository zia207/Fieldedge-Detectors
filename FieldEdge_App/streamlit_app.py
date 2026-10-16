"""Entry point for Streamlit Community Cloud.

The public app runs Delineate Anything V2 on CPU, with Steger line extraction and the snake.
SAM 2.1 stays on the local app (app.py), which has a GPU.
"""
import os
import sys

os.environ["FIELD_EDGE_CLOUD"] = "1"

def _ensure_headless_cv2():
    """ultralytics also installs opencv-python, which needs libGL.

    The free host has no libGL. Reinstall the headless wheel last so cv2 loads.
    """
    try:
        import cv2  # noqa: F401
        return
    except OSError:
        pass
    import subprocess
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
            "opencv-python-headless==4.10.0.84",
        ]
    )
    for name in list(sys.modules):
        if name == "cv2" or name.startswith("cv2."):
            del sys.modules[name]

_ensure_headless_cv2()

import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), run_name="__main__")
