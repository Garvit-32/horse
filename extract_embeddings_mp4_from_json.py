import os
import json
import torch
import cv2   # for MP4 handling

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
BASE_FOLDER   = '/home/ubuntu/workspace/output_videos_split'   # where *_frameinfo.json live
OUTPUT_FOLDER = '/home/ubuntu/workspace/filtered_output'       # new base for filtered tensors & videos

# Derived roots
JSON_ROOT       = BASE_FOLDER
DINO_EMB_ROOT   = BASE_FOLDER.replace('output_videos_split', 'dino_tensor')
DESSIE_EMB_ROOT = BASE_FOLDER.replace('output_videos_split', 'data_tensor')
MP4_ROOT        = BASE_FOLDER.replace('output_videos_split', 'dataset')

# ---------------------------------------------------
# Utility helpers
# ---------------------------------------------------

def process_frameinfo(json_path):
    """Load a frameinfo JSON with a permissive UTF‑8 decoder."""
    with open(json_path, 'rb') as f:
        raw = f.read()
    return json.loads(raw.decode('utf-8', errors='ignore'))


def list_frameinfo_jsons(json_root):
    """Return every *_frameinfo.json path under *json_root*."""
    out = []
    for dirpath, _, files in os.walk(json_root):
        for fn in files:
            if fn.endswith('_frameinfo.json'):
                out.append(os.path.join(dirpath, fn))
    return out


def derive_paths(json_path):
    """Given a JSON path, derive mp4, dino, and dessie tensor paths."""
    rel = os.path.relpath(json_path, JSON_ROOT)
    mp4_path   = os.path.join(MP4_ROOT, rel).replace('_frameinfo.json', '.mp4')
    dino_emb   = os.path.join(DINO_EMB_ROOT,  rel).replace('_frameinfo.json', '_results.pt')
    dessie_emb = os.path.join(DESSIE_EMB_ROOT, rel).replace('_frameinfo.json', '_results.pt')
    return mp4_path, dino_emb, dessie_emb


def subset_by_frame_indices(data_dict, idx_key, value_key, valid_idx):
    """Return dict containing only entries whose *idx_key* is in *valid_idx*."""
    frame_tensor = torch.as_tensor(data_dict[idx_key])
    keep_mask    = torch.isin(frame_tensor, torch.tensor(valid_idx))
    return {
        idx_key:   frame_tensor[keep_mask],
        value_key: data_dict[value_key][keep_mask]
    }


def build_output_path(src_path: str, root_src: str, suffix: str) -> str:
    """Mirror *src_path* tree under *OUTPUT_FOLDER*, changing filename suffix."""
    rel = os.path.relpath(src_path, root_src)          # keep sub‑folder structure
    rel = rel.replace('_results.pt', suffix)            # tensors
    rel = rel.replace('.mp4',       suffix)            # videos
    out_path = os.path.join(OUTPUT_FOLDER, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path

# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    json_list = sorted(list_frameinfo_jsons(JSON_ROOT))

    for json_path in json_list:
        print(f"Processing {json_path}")
        frameinfo = process_frameinfo(json_path)

        # left/right & valid → indices
        valid_idx = sorted(int(k) for k, v in frameinfo.items() if v[0] in {"left", "right"} and v[1])
        if not valid_idx:
            print("  no valid frames, skipping\n")
            continue

        mp4_file, dino_file, dessie_file = derive_paths(json_path)

        # ---------- DESSIE ----------
        if os.path.exists(dessie_file):
            dessie_data = torch.load(dessie_file, map_location='cpu')
            filtered = subset_by_frame_indices(dessie_data, 'frame_idx', 'pred_kp3d_crop', valid_idx)
            out_path = build_output_path(dessie_file, DESSIE_EMB_ROOT, '_best_view_dessie.pt')
            torch.save(filtered, out_path)
            print(f"  ▶ saved DESSIE subset → {out_path}")
        else:
            print("  (DESSIE tensor missing)")

        # ---------- DINO ----------
        if os.path.exists(dino_file):
            dino_data = torch.load(dino_file, map_location='cpu')
            filtered = subset_by_frame_indices(dino_data, 'frame_idx', 'x_norm_clstoken', valid_idx)
            out_path = build_output_path(dino_file, DINO_EMB_ROOT, '_best_view_dino.pt')
            torch.save(filtered, out_path)
            print(f"  ▶ saved DINO subset   → {out_path}")
        else:
            print("  (DINO tensor missing)")

        # ---------- MP4 (best‑view clip) ----------
        if os.path.exists(mp4_file):
            cap = cv2.VideoCapture(mp4_file)
            if not cap.isOpened():
                print("  (could not open MP4)")
            else:
                fps    = cap.get(cv2.CAP_PROP_FPS) or 30
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                out_mp4 = build_output_path(mp4_file, MP4_ROOT, '_best_view.mp4')
                writer  = cv2.VideoWriter(out_mp4, fourcc, fps, (width, height))

                for idx in valid_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        writer.write(frame)
                writer.release()
                cap.release()
                print(f"  ▶ saved MP4 subset   → {out_mp4}")
        else:
            print("  (MP4 missing)")

        break
