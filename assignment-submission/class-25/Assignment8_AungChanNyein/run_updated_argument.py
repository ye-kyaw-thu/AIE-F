#!/usr/bin/env python3
"""
Builds an EXACT keypoint filename index (not a fragile substring
rglob) so 'idx20-2' never wrongly matches 'idx20-20' or 'idx20-200'.
"""
import argparse, json, os, sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils import parse_annotation_file, build_label_vocabulary
from augment import augment_dataset_all_classes


def build_exact_keypoint_index(keypoint_dir):
    keypoint_dir = Path(keypoint_dir)
    index = {}
    for p in keypoint_dir.rglob('*.npy'):
        if p.stem.endswith('_vis'):
            continue
        index[p.stem] = p
    return index


def match_records_to_keypoints(records, keypoint_dir):
    index = build_exact_keypoint_index(keypoint_dir)
    print(f"Indexed {len(index)} keypoint files (exact stems)")

    matched, unmatched = 0, 0
    for rec in records:
        idx = rec['idx']
        candidates = [f"idx{idx}", f"idx20-{idx}"]
        found = None
        for stem in candidates:
            if stem in index:
                found = index[stem]
                break
        if found is None:
            for stem, path in index.items():
                if stem.split('-')[-1] == str(idx):
                    found = path
                    break
        rec['keypoint_path'] = str(found) if found else None
        if found:
            matched += 1
        else:
            unmatched += 1

    print(f"Matched: {matched}  Unmatched: {unmatched}")
    if unmatched > 0:
        sample_unmatched = [r['idx'] for r in records if r.get('keypoint_path') is None][:10]
        print(f"Sample unmatched idx values: {sample_unmatched}")
        print(f"Sample available keypoint stems: {list(index.keys())[:10]}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--aug_factor', type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    acfg = cfg['augmentation']
    if args.aug_factor:
        acfg['aug_factor'] = args.aug_factor

    records = parse_annotation_file(dcfg['annotation_file'])
    label2idx, idx2label = build_label_vocabulary(records)

    records = match_records_to_keypoints(records, dcfg['keypoint_dir'])

    n_with_kp = sum(1 for r in records if r.get('keypoint_path'))
    print(f"\nRecords with valid keypoint_path: {n_with_kp}/{len(records)}")
    if n_with_kp < len(records) * 0.9:
        print("WARNING: less than 90% matched â€” check filename pattern before proceeding")

    manifest = augment_dataset_all_classes(
        keypoint_dir = dcfg['keypoint_dir'],
        output_dir   = dcfg['augmented_dir'],
        records      = records,
        label2idx    = label2idx,
        aug_factor   = acfg.get('aug_factor', 20),
        aug_cfg      = acfg,
        seed         = dcfg.get('seed', 42),
    )

    with open(Path(dcfg['label_map_file']), 'w', encoding='utf-8') as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    print(f"Saved label_map.json with {len(label2idx)} classes")


if __name__ == '__main__':
    main()