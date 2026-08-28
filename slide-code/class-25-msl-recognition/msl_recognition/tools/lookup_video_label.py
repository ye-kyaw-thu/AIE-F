#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/lookup_video_label.py
───────────────────────────
Look up the correct annotation label for any video filename.

Because videos are matched to annotations by ALPHABETICAL SORT ORDER,
the number in a filename like "idx20-31.mp4" does NOT correspond to
annotation line 31 — it depends on where the file falls in the sorted list.

Usage:
    python tools/lookup_video_label.py --video idx20-31.mp4
    python tools/lookup_video_label.py --video idx20-100.mp4
    python tools/lookup_video_label.py --list   # print all video→label mappings
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils import parse_annotation_file, build_label_vocabulary, match_videos_to_annotations

CONFIG_DEFAULT = 'config/config.yaml'

def main():
    import yaml
    parser = argparse.ArgumentParser(description='Look up annotation for a video file')
    parser.add_argument('--config',  default=CONFIG_DEFAULT)
    parser.add_argument('--video',   default=None, help='Video filename (e.g. idx20-31.mp4)')
    parser.add_argument('--list',    action='store_true', help='Print all video→label mappings')
    parser.add_argument('--csv',     default=None, help='Save all mappings to CSV')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg['data']

    records              = parse_annotation_file(dcfg['annotation_file'])
    label2idx, idx2label = build_label_vocabulary(records)
    records              = match_videos_to_annotations(
        dcfg['video_dir'], records, dcfg['keypoint_dir']
    )

    # Build filename → record mapping
    filename_to_rec = {}
    for i, rec in enumerate(records):
        vp = rec.get('video_path')
        if vp:
            filename_to_rec[Path(vp).name] = (i, rec)

    if args.video:
        name = Path(args.video).name
        if name in filename_to_rec:
            pos, rec = filename_to_rec[name]
            print(f"\n{'─'*65}")
            print(f"  Video            : {name}")
            print(f"  Sort position    : #{pos}  (alphabetical order)")
            print(f"  Annotation index : {rec['idx']}")
            print(f"  Normal text      : {rec['label']}")
            print(f"  MSL gloss        : {rec.get('msl_gloss', 'N/A')}")
            print(f"  Class index      : {label2idx[rec['label']]}")
            print(f"{'─'*65}\n")
        else:
            print(f"[ERROR] Video not found: {name}")
            print(f"Available videos: {len(filename_to_rec)}")

    if args.list or args.csv:
        rows = [(pos, Path(rec.get('video_path','')).name,
                 rec['label'], rec.get('msl_gloss',''), label2idx[rec['label']])
                for pos, (pos, rec) in enumerate(
                    sorted(filename_to_rec.values(), key=lambda x: x[0]))]

        if args.list:
            print(f"\n{'Pos':>5}  {'Video filename':<25}  {'Label':<45}  {'Class':>6}")
            print('─'*90)
            for pos, fname, label, gloss, cls_idx in rows:
                print(f"{pos:>5}  {fname:<25}  {label[:43]:<45}  {cls_idx:>6}")

        if args.csv:
            import csv
            with open(args.csv, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['sort_position', 'video_filename', 'normal_text',
                            'msl_gloss', 'class_index'])
                w.writerows(rows)
            print(f"Saved all mappings → {args.csv}")

if __name__ == '__main__':
    main()

