## Myanmar Syllable Handwritten (sylHandwriting) Dataset to Image Conversion
## Vibe Coding with ChatGPT by Ye Kyaw Thu, LU Lab., Myanmar
## Last Updated: 23 Mar 2026
## How to run: python ./convert2image.py --help
## E.g.: python ./convert2image.py --dataset dataset --output image
## E.g.: python ./convert2image.py --dataset dataset --color_mode stroke --output stroke  
## E.g.: python ./convert2image.py --dataset dataset --color_mode time --output time   

import os
import argparse
import re
from PIL import Image, ImageDraw
import random


# -------------------------
# Parse Stroke File
# -------------------------
def parse_strokes(filepath):
    strokes = []
    current_stroke = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("STROKE"):
                if current_stroke:
                    strokes.append(current_stroke)
                    current_stroke = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    current_stroke.append((x, y))

    if current_stroke:
        strokes.append(current_stroke)

    return strokes


# -------------------------
# Normalize + Scale
# -------------------------
def normalize_strokes(strokes, img_size, padding=10):
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    width = max_x - min_x
    height = max_y - min_y

    scale = (img_size - 2 * padding) / max(width, height) if max(width, height) > 0 else 1

    normalized = []

    for stroke in strokes:
        new_stroke = []
        for (x, y) in stroke:
            nx = (x - min_x) * scale
            ny = (y - min_y) * scale
            new_stroke.append((nx, ny))
        normalized.append(new_stroke)

    # Centering
    all_x = [p[0] for s in normalized for p in s]
    all_y = [p[1] for s in normalized for p in s]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    offset_x = (img_size - (max_x - min_x)) / 2 - min_x
    offset_y = (img_size - (max_y - min_y)) / 2 - min_y

    final = []
    for stroke in normalized:
        new_stroke = []
        for (x, y) in stroke:
            new_stroke.append((x + offset_x, y + offset_y))
        final.append(new_stroke)

    return final


# -------------------------
# Draw Image
# -------------------------
def draw_image(strokes, img_size, color_mode):
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


# -------------------------
# Process Dataset
# -------------------------
def process_dataset(dataset_dir, output_dir, img_size, fmt, color_mode):
    for user in os.listdir(dataset_dir):
        user_path = os.path.join(dataset_dir, user)
        if not os.path.isdir(user_path):
            continue

        out_user_path = os.path.join(output_dir, user)
        os.makedirs(out_user_path, exist_ok=True)

        for fname in os.listdir(user_path):
            if not fname.endswith(".txt") or fname == "user_info.json":
                continue

            input_path = os.path.join(user_path, fname)

            strokes = parse_strokes(input_path)
            if not strokes:
                continue

            strokes = normalize_strokes(strokes, img_size)

            img = draw_image(strokes, img_size, color_mode)

            base = os.path.splitext(fname)[0]
            output_path = os.path.join(out_user_path, f"{base}.{fmt}")

            img.save(output_path)


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert LipiTK stroke dataset to images"
    )

    parser.add_argument("--dataset", default="dataset", help="Dataset folder")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--size", type=int, default=128, help="Image size (default: 128)")
    parser.add_argument("--format", default="png", choices=["png", "jpg"], help="Image format")
    parser.add_argument(
        "--color_mode",
        default="single",
        choices=["single", "stroke", "time"],
        help="Color mode: single, stroke, time"
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print("Dataset folder not found")
        return

    os.makedirs(args.output, exist_ok=True)

    process_dataset(
        args.dataset,
        args.output,
        args.size,
        args.format,
        args.color_mode
    )

    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
