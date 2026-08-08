#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/probe_mediapipe.py
Run this FIRST to discover the correct MediaPipe import path on your system.

Usage:
    python tools/probe_mediapipe.py
"""
import sys
import importlib

print(f"Python  : {sys.version}")
print()

# 1. Basic mediapipe version
try:
    import mediapipe as mp
    print(f"mediapipe version : {mp.__version__}")
    print(f"mediapipe file    : {mp.__file__}")
except ImportError as e:
    print(f"[FAIL] Cannot import mediapipe at all: {e}")
    sys.exit(1)

print()

# 2. Probe every likely import path
candidates = [
    ("mediapipe.solutions.holistic",        "mediapipe.solutions.holistic"),
    ("mediapipe.solutions.drawing_utils",   "mediapipe.solutions.drawing_utils"),
    ("mediapipe.solutions.hands",           "mediapipe.solutions.hands"),
    ("mediapipe.python.solutions.holistic", "mediapipe.python.solutions.holistic"),
    ("mediapipe.python.solutions.drawing_utils", "mediapipe.python.solutions.drawing_utils"),
    ("mediapipe.python.solutions.hands",    "mediapipe.python.solutions.hands"),
]

print("Probing import paths:")
working = {}
for label, mod_path in candidates:
    try:
        mod = importlib.import_module(mod_path)
        print(f"  [OK ]  {mod_path}")
        key = mod_path.split(".")[-1]   # holistic / drawing_utils / hands
        working[key] = mod_path
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  [FAIL] {mod_path}  →  {e}")

print()

# 3. Try attribute access (legacy mp.solutions.holistic)
print("Probing attribute access:")
for attr in ["solutions", "solutions.holistic", "solutions.drawing_utils"]:
    try:
        obj = mp
        for part in attr.split("."):
            obj = getattr(obj, part)
        print(f"  [OK ]  mp.{attr}")
    except AttributeError as e:
        print(f"  [FAIL] mp.{attr}  →  {e}")

print()

# 4. List top-level subpackages of mediapipe
import pkgutil, pathlib
mp_dir = pathlib.Path(mp.__file__).parent
print(f"Top-level mediapipe subpackages in {mp_dir}:")
for item in sorted(mp_dir.iterdir()):
    if item.is_dir() and not item.name.startswith("_"):
        print(f"  {item.name}/")
    elif item.suffix == ".py" and not item.name.startswith("_"):
        print(f"  {item.name}")

# 5. Look for 'solutions' anywhere under mediapipe
print()
sol_dirs = list(mp_dir.rglob("solutions"))
print(f"'solutions' directories found under mediapipe ({len(sol_dirs)}):")
for d in sol_dirs:
    rel = d.relative_to(mp_dir)
    print(f"  {rel}/")
    for f in sorted(d.glob("*.py")):
        print(f"    {f.name}")

print()

# 6. Final verdict + recommended import snippet
if working.get("holistic") and working.get("drawing_utils"):
    h_path  = working["holistic"]
    d_path  = working["drawing_utils"]
    hd_path = working.get("hands", h_path.replace("holistic", "hands"))
    print("=" * 55)
    print("RECOMMENDED import snippet for your mediapipe version:")
    print("=" * 55)
    print(f"  import {h_path} as mp_holistic")
    print(f"  import {d_path} as mp_drawing")
    print(f"  import {hd_path} as mp_hands")
else:
    print("=" * 55)
    print("WARNING: Could not find a working holistic import path.")
    print("Consider: pip install 'mediapipe==0.10.9'")
    print("=" * 55)

