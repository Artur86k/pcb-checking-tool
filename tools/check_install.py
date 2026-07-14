"""Verify the installation: imports, tkinter, and the CNN model file."""
import importlib
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORE = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("ezdxf", "ezdxf"),
    ("rawpy", "rawpy"),
    ("PIL", "pillow"),
    ("pandas", "pandas"),
    ("openpyxl", "openpyxl"),
    ("tkinter", "(ships with python.org installers; on Linux: "
                "apt install python3-tk)"),
]
OPTIONAL = [("torch", "torch - needed for the CNN presence backend")]

print(f"Python {sys.version.split()[0]}")
failed = []
for module, hint in CORE:
    try:
        importlib.import_module(module)
        print(f"  [ok]      {module}")
    except ImportError:
        print(f"  [MISSING] {module}  ->  pip install {hint}")
        failed.append(module)

cnn_ready = True
for module, hint in OPTIONAL:
    try:
        importlib.import_module(module)
        print(f"  [ok]      {module}")
    except ImportError:
        print(f"  [missing] {module}  ->  {hint}")
        cnn_ready = False

model = os.path.join(BASE, "golden", "presence_cnn.pt")
if os.path.isfile(model):
    print(f"  [ok]      CNN model: {model}")
else:
    print(f"  [missing] CNN model: {model}")
    print("            The presence check will use the unreliable color")
    print("            heuristic. Copy presence_cnn.pt from the machine")
    print("            where it was trained (into the golden/ folder),")
    print("            or train it: see docs/training.md")
    cnn_ready = False

print()
if failed:
    print("INSTALLATION INCOMPLETE - missing core packages:", ", ".join(failed))
    sys.exit(1)
print("Core installation OK - run the app with:  python -m overlay_tool")
if not cnn_ready:
    print("Note: CNN presence backend not ready (see [missing] lines above).")
