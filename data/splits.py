"""
Official MMVR train/val/test split loading from data_split.npz.
"""

import glob
import hashlib
import json
import os
import numpy as np
from tqdm import tqdm


def _split_cache_path(data_root, split_file, protocol, cache_dir=None):
    """
    Build a cache path that is specific to the dataset root, split file, and
    protocol. The path hash prevents Windows/WSL or P1/P2 roots from sharing
    an incompatible cache by accident.
    """
    cache_dir = cache_dir or os.path.join('.', 'cache', 'splits')
    cache_key = json.dumps({
        'data_root': os.path.abspath(data_root),
        'split_file': os.path.abspath(split_file),
        'protocol': protocol,
    }, sort_keys=True)
    digest = hashlib.sha1(cache_key.encode('utf-8')).hexdigest()[:12]
    return os.path.join(cache_dir, f'{protocol}_{digest}_samples.json')


def _load_cached_samples(cache_path, data_root, split_file, protocol):
    """Load cached split samples when the cache metadata still matches."""
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r') as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not read split cache '{cache_path}': {e}")
        return None

    expected_meta = {
        'data_root': os.path.abspath(data_root),
        'split_file': os.path.abspath(split_file),
        'protocol': protocol,
    }
    if payload.get('metadata') != expected_meta:
        print(f"[WARNING] Ignoring stale split cache: {cache_path}")
        return None

    train_samples = payload.get('train_samples', [])
    val_samples   = payload.get('val_samples', [])
    test_samples  = payload.get('test_samples', [])

    print(f"Loaded cached split index ({protocol}) → {cache_path}")
    print(f"  Train : {len(train_samples):,} person-frame samples")
    print(f"  Val   : {len(val_samples):,} person-frame samples")
    print(f"  Test  : {len(test_samples):,} person-frame samples")
    return train_samples, val_samples, test_samples


def _save_cached_samples(cache_path, data_root, split_file, protocol,
                         train_samples, val_samples, test_samples):
    """Persist split sample lists so later runs can skip directory scanning."""
    payload = {
        'metadata': {
            'data_root': os.path.abspath(data_root),
            'split_file': os.path.abspath(split_file),
            'protocol': protocol,
        },
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
    }

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(payload, f)
        print(f"Saved split index cache → {cache_path}")
    except Exception as e:
        print(f"[WARNING] Could not save split cache '{cache_path}': {e}")


def load_split_segments(split_file, protocol='P1S1'):
    """
    Load the official MMVR train/val/test segment lists from data_split.npz.

    Each entry in the split lists is a 'session/segment' string,
    e.g. 'd1s1/000', meaning all frames in P1/d1s1/000/ belong to that split.

    Args:
        split_file : path to data_split.npz
        protocol   : one of 'P1S1', 'P1S2', 'P2S1', 'P2S2'

    Returns:
        train_segs, val_segs, test_segs : sets of 'session/segment' strings
    """
    data       = np.load(split_file, allow_pickle=True)
    split_dict = data['data_split_dict'].item()

    if protocol not in split_dict:
        raise ValueError(f"Protocol '{protocol}' not found. "
                         f"Choose from: {list(split_dict.keys())}")

    splits = split_dict[protocol]
    train_segs = set(splits['train'])
    val_segs   = set(splits['val'])
    test_segs  = set(splits['test'])

    print(f"Protocol {protocol} — official segment counts:")
    print(f"  Train : {len(train_segs)} segments")
    print(f"  Val   : {len(val_segs)} segments")
    print(f"  Test  : {len(test_segs)} segments")
    return train_segs, val_segs, test_segs


def load_mmvr_samples_split(data_root, split_file, protocol='P1S1',
                            use_cache=True, force_rebuild=False,
                            cache_dir=None):
    """
    Scan P1 and assign each person-frame sample to train/val/test
    according to the official data_split.npz segment lists.

    A sample's split is determined by which set its 'session/segment'
    key belongs to (e.g. 'd1s2/007' → whichever split contains that string).

    Returns:
        train_samples, val_samples, test_samples : lists of sample dicts
            Each dict has keys:
            { radar_path, pose_path, bbox_path, person_idx, session, segment, frame_id }
    """
    if not os.path.exists(data_root):
        print(f"[WARNING] '{data_root}' not found.")
        return [], [], []
    if not os.path.exists(split_file):
        print(f"[WARNING] '{split_file}' not found — place data_split.npz "
              f"in the same folder as this project.")
        return [], [], []

    cache_path = _split_cache_path(data_root, split_file, protocol, cache_dir)
    if use_cache and not force_rebuild:
        cached = _load_cached_samples(cache_path, data_root, split_file, protocol)
        if cached is not None:
            return cached

    train_segs, val_segs, test_segs = load_split_segments(split_file, protocol)

    train_samples, val_samples, test_samples = [], [], []
    n_skipped = 0

    sessions = sorted([d for d in os.listdir(data_root)
                       if os.path.isdir(os.path.join(data_root, d))])

    for session in tqdm(sessions, desc='Scanning sessions'):
        sess_path = os.path.join(data_root, session)
        segments  = sorted([s for s in os.listdir(sess_path)
                             if os.path.isdir(os.path.join(sess_path, s))])

        for segment in segments:
            seg_key  = f"{session}/{segment}"   # e.g. 'd1s2/007'
            seg_path = os.path.join(sess_path, segment)

            # Determine which split this segment belongs to
            if seg_key in train_segs:
                target = train_samples
            elif seg_key in val_segs:
                target = val_samples
            elif seg_key in test_segs:
                target = test_samples
            else:
                # Segment not listed in the chosen protocol — skip it
                n_skipped += 1
                continue

            # Collect all frames in this segment
            pose_files = sorted(glob.glob(os.path.join(seg_path, '*_pose.npz')))
            for pose_path in pose_files:
                frame_id   = os.path.basename(pose_path).replace('_pose.npz', '')
                radar_path = pose_path.replace('_pose.npz', '_radar.npz')
                bbox_path  = pose_path.replace('_pose.npz', '_bbox.npz')

                if not os.path.exists(radar_path):
                    continue

                try:
                    kp = np.load(pose_path)['kp']   # (n, 17, 3)
                    n_persons = kp.shape[0]
                except Exception:
                    continue

                if n_persons == 0:
                    continue

                for person_idx in range(n_persons):
                    target.append({
                        'radar_path' : radar_path,
                        'pose_path'  : pose_path,
                        'bbox_path'  : bbox_path,
                        'person_idx' : person_idx,
                        'session'    : session,
                        'segment'    : segment,
                        'frame_id'   : frame_id,
                    })

    print(f"\nOfficial split loaded ({protocol}):")
    print(f"  Train : {len(train_samples):,} person-frame samples")
    print(f"  Val   : {len(val_samples):,} person-frame samples")
    print(f"  Test  : {len(test_samples):,} person-frame samples")
    if n_skipped:
        print(f"  Skipped {n_skipped} segments not in protocol '{protocol}'")

    if use_cache:
        _save_cached_samples(cache_path, data_root, split_file, protocol,
                             train_samples, val_samples, test_samples)

    return train_samples, val_samples, test_samples
