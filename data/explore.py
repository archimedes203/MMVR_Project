"""
Dataset structure exploration utility.
"""

import os


def explore_dataset(data_root):
    """
    Walk the P1 directory and print its real structure.
    Expected layout:
        P1/
          d1s1/000/00000_radar.npz  ...
          d1s2/000/00000_pose.npz   ...
    """
    if not os.path.exists(data_root):
        print(f"[WARNING] Data root '{data_root}' not found.")
        print("Please extract P1.zip into the same folder as this project.")
        return []

    print(f"Dataset root: {os.path.abspath(data_root)}")
    sessions = sorted([d for d in os.listdir(data_root)
                       if os.path.isdir(os.path.join(data_root, d))])
    print(f"Sessions found: {len(sessions)}  →  {sessions[:6]} ...")

    # Drill into first session to show segment/frame structure
    for sess in sessions[:2]:
        sess_path = os.path.join(data_root, sess)
        segments  = sorted(os.listdir(sess_path))[:2]
        for seg in segments:
            seg_path = os.path.join(sess_path, seg)
            if not os.path.isdir(seg_path):
                continue
            files = sorted(os.listdir(seg_path))[:5]
            print(f"  {sess}/{seg}/  →  {files}")
    return sessions
