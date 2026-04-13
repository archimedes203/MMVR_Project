"""
Official MMVR train/val/test split loading from data_split.npz.
"""

import os
import glob
import numpy as np
from tqdm import tqdm


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


def load_mmvr_samples_split(data_root, split_file, protocol='P1S1'):
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

    return train_samples, val_samples, test_samples
