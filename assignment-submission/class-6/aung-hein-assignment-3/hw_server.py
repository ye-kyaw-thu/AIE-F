#!/usr/bin/env python3

import os
import re
import json
import argparse
from PIL import Image, ImageDraw
from PIL.ImagePalette import random
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import random

DATASET_DIR = "dataset"
IMAGE_DIRS   = {"single": "single", "stroke": "stroke", "time": "time"}
IMG_SIZE     = 128

app = Flask(__name__, static_folder=".")
CORS(app)

TEXT_LINES = []

def get_user_path(user_name):
    return os.path.join(DATASET_DIR, user_name)


def sort_key(name):
    return [int(n) for n in re.findall(r'\d+', name)]


def user_progress(user_name):
    path = get_user_path(user_name)
    if not os.path.isdir(path):
        return 0
    written = set()
    for f in os.listdir(path):
        m = re.match(r"(\d+)-\d+\.txt", f)
        if m:
            written.add(int(m.group(1)))
    return len(written)


def parse_strokes(filepath):
    strokes, current = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("STROKE"):
                if current:
                    strokes.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current.append((float(parts[0]), float(parts[1])))
    if current:
        strokes.append(current)
    return strokes


def normalize_strokes(strokes, img_size, padding=10):
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span = max(max_x - min_x, max_y - min_y)
    scale = (img_size - 2 * padding) / span if span > 0 else 1

    normed = [[(( x - min_x) * scale, (y - min_y) * scale) for x, y in s] for s in strokes]

    all_x2 = [p[0] for s in normed for p in s]
    all_y2 = [p[1] for s in normed for p in s]
    ox = (img_size - (max(all_x2) - min(all_x2))) / 2 - min(all_x2)
    oy = (img_size - (max(all_y2) - min(all_y2))) / 2 - min(all_y2)
    return [[(x + ox, y + oy) for x, y in s] for s in normed]


def generate_image(strokes, img_size, color_mode):
    img = Image.new("L", (img_size, img_size), 255)  # grayscale
    draw = ImageDraw.Draw(img)

    # Color handling
    if color_mode != "single":
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)

    for i, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue

        if color_mode == "single":
            color = 0  # black

        elif color_mode == "stroke":
            color = tuple(random.randint(0, 255) for _ in range(3))

        elif color_mode == "time":
            color = (
                int(255 * i / max(1, len(strokes))),
                0,
                255 - int(255 * i / max(1, len(strokes)))
            )

        for j in range(1, len(stroke)):
            draw.line([stroke[j - 1], stroke[j]], fill=color, width=2)

    return img


def ensure_image(user_name, txt_fname, color_mode):
    base     = os.path.splitext(txt_fname)[0]
    out_dir  = os.path.join(IMAGE_DIRS[color_mode], user_name)
    out_path = os.path.join(out_dir, base + ".png")

    if not os.path.exists(out_path):
        txt_path = os.path.join(get_user_path(user_name), txt_fname)
        strokes  = parse_strokes(txt_path)
        if strokes:
            strokes = normalize_strokes(strokes, IMG_SIZE)
            img     = generate_image(strokes, IMG_SIZE, color_mode)
            os.makedirs(out_dir, exist_ok=True)
            img.save(out_path)

    return out_path



@app.route("/api/lines")
def api_lines():
    return jsonify({"lines": TEXT_LINES})


@app.route("/api/users")
def api_users():
    os.makedirs(DATASET_DIR, exist_ok=True)
    users = []
    for u in sorted(os.listdir(DATASET_DIR)):
        p = get_user_path(u)
        if os.path.isdir(p):
            info_path = os.path.join(p, "user_info.json")
            info = {}
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
            users.append({"name": u, "progress": user_progress(u), "info": info})
    return jsonify({"users": users, "total": len(TEXT_LINES)})


@app.route("/api/users", methods=["POST"])
def create_user():
    data = request.json
    name = data.get("name", "").strip().replace(" ", "_")
    if not name:
        return jsonify({"error": "Name required"}), 400
    path = get_user_path(name)
    os.makedirs(path, exist_ok=True)
    info = {"name": name, "age": data.get("age", ""),
            "sex": data.get("sex", ""), "education": data.get("education", "")}
    with open(os.path.join(path, "user_info.json"), "w") as f:
        json.dump(info, f)
    return jsonify({"ok": True, "name": name, "progress": 0})


@app.route("/api/save", methods=["POST"])
def save_sample():
    data      = request.json
    user_name = data.get("user")
    line_index= data.get("index")
    strokes   = data.get("strokes")

    if not user_name or line_index is None or not strokes:
        return jsonify({"error": "Missing fields"}), 400

    path = get_user_path(user_name)
    if not os.path.isdir(path):
        return jsonify({"error": "Unknown user"}), 404

    base = str(line_index + 1)
    i = 1
    while True:
        fname = f"{base}-{i}.txt"
        if not os.path.exists(os.path.join(path, fname)):
            break
        i += 1

    with open(os.path.join(path, fname), "w") as f:
        for si, stroke in enumerate(strokes):
            f.write(f"STROKE {si+1}\n")
            for pt in stroke:
                f.write(f"{pt['x']} {pt['y']} {pt['t']:.6f}\n")
            f.write("\n")

    return jsonify({"ok": True, "file": fname, "progress": user_progress(user_name)})


@app.route("/api/progress/<user_name>")
def api_progress(user_name):
    return jsonify({"progress": user_progress(user_name), "total": len(TEXT_LINES)})


@app.route("/api/gallery/<user_name>/<color_mode>", methods=["POST"])
def api_gallery(user_name, color_mode):
    if color_mode not in IMAGE_DIRS:
        return jsonify({"error": "Invalid mode"}), 400

    user_path = get_user_path(user_name)
    if not os.path.isdir(user_path):
        return jsonify({"error": "Unknown user"}), 404

    data      = request.get_json(silent=True) or {}
    known_set = set(data.get("known", []))

    txt_files = sorted(
        [f for f in os.listdir(user_path)
         if f.endswith(".txt") and f != "user_info.json"],
        key=sort_key
    )

    items = []
    for txt_fname in txt_files:
        ensure_image(user_name, txt_fname, color_mode)

        if txt_fname in known_set:
            continue

        base  = os.path.splitext(txt_fname)[0]
        m     = re.match(r"^(\d+)-\d+$", base)
        label = ""
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(TEXT_LINES):
                label = f"{idx+1}: {TEXT_LINES[idx]}"

        items.append({
            "txt":   txt_fname,
            "img":   f"/api/gallery/image/{color_mode}/{user_name}/{base}.png",
            "label": label,
        })

    return jsonify({"items": items})


@app.route("/api/gallery/image/<color_mode>/<user_name>/<filename>")
def api_gallery_image(color_mode, user_name, filename):
    """Serve a generated gallery image."""
    if color_mode not in IMAGE_DIRS:
        abort(404)
    img_dir = os.path.join(IMAGE_DIRS[color_mode], user_name)
    return send_from_directory(img_dir, filename)


@app.route("/api/delete/<user_name>/<fname>", methods=["DELETE"])
def api_delete(user_name, fname):
    """Delete stroke .txt and all 3 mode images for it."""
    if not re.match(r'^\d+-\d+\.txt$', fname):
        return jsonify({"error": "Invalid filename"}), 400

    txt_path = os.path.join(get_user_path(user_name), fname)
    if not os.path.exists(txt_path):
        return jsonify({"error": "File not found"}), 404

    os.remove(txt_path)

    base = os.path.splitext(fname)[0]
    for mode_dir in IMAGE_DIRS.values():
        img_path = os.path.join(mode_dir, user_name, base + ".png")
        if os.path.exists(img_path):
            os.remove(img_path)

    return jsonify({"ok": True, "progress": user_progress(user_name)})

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

def main():
    global TEXT_LINES

    parser = argparse.ArgumentParser(description="Myanmar HW Collector – Mobile Web Server")
    parser.add_argument("--file", required=True, help="Text file with one syllable per line")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        TEXT_LINES = [l.strip() for l in f if l.strip()]

    print(f"\n  Loaded {len(TEXT_LINES)} lines from {args.file}")
    print(f"  Server running — open http://<YOUR_MAC_IP>:{args.port} on your phone\n")
    print("    Find your IP with:  ipconfig getifaddr en0\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()