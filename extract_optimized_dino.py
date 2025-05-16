import argparse
import sys, os
import pandas as pd
import torch
import json
import cv2
import os
import numpy as np
import multiprocessing as mp
from torchvision import transforms
from tqdm import tqdm
import pickle
import gc
import traceback

# Normalization transform
img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])
norm_transform = transforms.Normalize(img_mean, img_std)

# Normalization transform (used for DINO)
dino_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized Video Processing Script")
    # --- Essential Paths ---
    parser.add_argument('--data_dir', type=str, default = '/home/ubuntu/workspace',  help="Root directory containing 'data_tensor', 'data', 'data_cropped'")
    parser.add_argument('--save_dir', type=str, default = 'videos',  help="Root directory to save processed videos (currently unused)")
    parser.add_argument('--prototype_path', type=str, default='/home/ubuntu/workspace/prototypes.pth', help="Path to pre-computed DINO prototypes (currently unused)")
    parser.add_argument('--ckpt_file', type=str, help="Path to DESSIE checkpoint (currently unused)")

    # --- Processing Parameters ---
    parser.add_argument('--num_workers', type=int, default=6, help="Number of worker processes for DataLoader")
    parser.add_argument('--dino_batch_size', type=int, default=256, help="Batch size for DINO inference")
    parser.add_argument('--imgsize', type=int, default=256, help="Target size for DESSIE input (used for KP scaling)")
    parser.add_argument('--output_fps', type=int, default=20, help="FPS for the output videos (currently unused)")
    parser.add_argument('--orientation_conf', type=float, default=0.5, help="Confidence threshold for kp-based orientation dx vs dz (currently unused)")
    parser.add_argument('--movement_threshold', type=float, default=1.5, help="Avg keypoint movement threshold (pixels) to determine if horse is moving (currently unused)")
    parser.add_argument('--smoothing_window', type=int, default=5, help="Window size for temporal smoothing of orientation (currently unused)")
    parser.add_argument('--bbox_expand_ratio', type=float, default=0.05, help="Ratio to expand bounding boxes")

    # --- Model & Feature Parameters ---
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vitb14", help="DINOv2 model name")

    # --- Debugging/Testing ---
    parser.add_argument('--limit_videos', type=int, default=None, help="Process only the first N videos for testing")
    parser.add_argument('--error_log_file', type=str, default='processing_errors.pkl', help="File to save list of errored video paths")

    # --- Inactive DESSIE arguments (kept for compatibility but not used) ---
    parser.add_argument('--train', type=str)
    parser.add_argument('--model_dir', type=str, default='/home/ubuntu/workspace/Dessie/internal_models')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--version', type=str)
    parser.add_argument('--ModelName', type=str, default='DESSIE')
    parser.add_argument('--DatasetName', type=str, default='DessiePIPE')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--W_shape_prior', default=0., type=float)
    parser.add_argument('--W_kp_img', default=0., type=float)
    parser.add_argument('--W_mask_img', default=0., type=float)
    parser.add_argument('--W_pose_img', default=0., type=float)
    parser.add_argument('--W_cos_shape', default=0., type=float)
    parser.add_argument('--W_cos_pose', default=0., type=float)
    parser.add_argument('--W_text_shape', default=0., type=float)
    parser.add_argument('--W_text_pose', default=0., type=float)
    parser.add_argument('--W_text_cam', default=0., type=float)
    parser.add_argument('--W_cosine_text_shape', default=0.0, type=float)
    parser.add_argument('--W_cosine_text_pose', default=0.0, type=float)
    parser.add_argument('--W_cosine_text_cam', default=0.0, type=float)
    parser.add_argument('--W_gt_shape', default=0., type=float)
    parser.add_argument('--W_gt_pose', default=0., type=float)
    parser.add_argument('--W_gt_trans', default=0., type=float)
    parser.add_argument('--W_l2_shape_1', default=0.0, type=float)
    parser.add_argument('--W_l2_pose_2', default=0.0, type=float)
    parser.add_argument('--W_l2_shape_3', default=0.0, type=float)
    parser.add_argument('--W_l2_pose_3', default=0.0, type=float)
    parser.add_argument('--W_l2_rootrot_1', default=0.0, type=float)
    parser.add_argument('--W_l2_rootrot_2', default=0.0, type=float)
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--PosePath', type=str)
    parser.add_argument("--TEXTUREPath", type=str)
    parser.add_argument('--uv_size', type=int, default=256)
    parser.add_argument('--data_batch_size', type=int, default=1)
    parser.add_argument('--useSynData', action="store_true", default=False)
    parser.add_argument('--useinterval', type=int, default=8)
    parser.add_argument("--getPairs", action="store_true", default=False)
    parser.add_argument("--TEXT", action="store_true", default=False)
    parser.add_argument("--DINO_frozen", action="store_true", default=False)
    parser.add_argument("--DINO_obtain_token", action="store_true", default=False)
    parser.add_argument("--GT", action="store_true", default=False)
    parser.add_argument("--pred_trans", action="store_true", default=False)
    parser.add_argument("--background", action="store_true", default=False)
    parser.add_argument("--background_path", default='')
    parser.add_argument("--REALDATASET", default='MagicPony')
    parser.add_argument("--REALPATH", default='')
    parser.add_argument("--web_images_num", type=int, default=0)
    parser.add_argument("--REALMagicPonyPATH", default='')

    return parser.parse_args()


# This function is kept minimal as most defaults are in parse_args
def set_default_args(args):
    return args

# Function to convert 3D keypoints to 2D (currently unused)
def convert_3d_to_2d(points, size=256):
    if points is None or points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)

    focal = 5000
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    z = np.maximum(z, 1e-6)
    x_proj = (focal * x) / z
    y_proj = (focal * y) / z

    screen_x = (size / 2) + x_proj
    screen_y = (size / 2) - y_proj # Flip Y

    screen_x = np.clip(screen_x, 0, size - 1)
    screen_y = np.clip(screen_y, 0, size - 1)

    proj_points = np.stack([screen_x, screen_y], axis=-1)
    return proj_points.astype(np.int32)

def list_mp4_files(root_dir):
    """
    Recursively lists .mp4 files that require processing (output not found, inputs found).
    """
    mp4_files = []
    print(f"Scanning directory: {root_dir}")
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                video_path = os.path.join(dirpath, filename)
                dino_tensor_path = video_path.replace('dataset', 'dino_tensor').replace('.mp4', '_results.pt')
                json_path = video_path.replace('dataset','data_cropped_new').replace('.mp4', '_seg.json')
                tensor_path = video_path.replace('dataset','data_tensor').replace('.mp4', '_results.pt')

                if not os.path.exists(dino_tensor_path) and os.path.exists(json_path) and os.path.exists(tensor_path):
                     mp4_files.append(video_path)
                     count += 1
                     if count % 1000 == 0:
                        print(f"Found {count} videos so far...")

    print(f"Finished scanning. Found {len(mp4_files)} videos requiring processing.")
    return mp4_files

def expand_bbox(bbox, frame_width, frame_height, expand_ratio=0.05):
    """Expands a bounding box by a given ratio, clamping to frame boundaries."""
    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    expand_w = int(bbox_width * expand_ratio)
    expand_h = int(bbox_height * expand_ratio)

    x1_new = max(0, x1 - expand_w)
    y1_new = max(0, y1 - expand_h)
    x2_new = min(frame_width, x2 + expand_w)
    y2_new = min(frame_height, y2 + expand_h)

    if x1_new >= x2_new or y1_new >= y2_new:
         return None

    return [x1_new, y1_new, x2_new, y2_new]

def get_dino_embedding_batch(image_batch_tensor, model):
    """Processes a batch of image tensors with DINO."""
    device = next(model.parameters()).device
    image_batch_tensor = image_batch_tensor.to(device)
    with torch.no_grad():
        output = model.forward_features(image_batch_tensor)
        if 'x_norm_clstoken' in output:
             embeddings = output['x_norm_clstoken']
        elif 'pooled_output' in output:
             embeddings = output['pooled_output']
        else:
             x_norm = output.get('x_norm', output.get('x'))
             if x_norm is not None and x_norm.ndim == 3:
                 embeddings = x_norm[:, 0]
             elif x_norm is not None and x_norm.ndim == 2:
                  embeddings = x_norm
             else:
                 raise ValueError(f"Could not find suitable embedding output in DINO model output keys: {output.keys()}")

    return embeddings # Returns embeddings on GPU

def init_worker(model_name, args_dict, error_list):
    """Initializes the DINO model and shared variables in each worker process."""
    global GLOBAL_MODEL, GLOBAL_ERROR_LIST, ARGS, DEVICE

    ARGS = argparse.Namespace(**args_dict)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Worker {os.getpid()} initializing on device {DEVICE}...")

    GLOBAL_ERROR_LIST = error_list

    try:
        GLOBAL_MODEL = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True).to(DEVICE)
        GLOBAL_MODEL.eval()
        print(f"Worker {os.getpid()} model '{model_name}' loaded successfully.")

    except Exception as e:
        error_msg = f"Error initializing worker {os.getpid()} with model '{model_name}': {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        GLOBAL_ERROR_LIST.append(f"Worker {os.getpid()} Init Error: {error_msg}")
        raise e

def get_super_bbox(bbox_list):
    """Combines multiple bounding boxes into a single encompassing box."""
    if not bbox_list:
        return None
    x1 = min(box[0] for box in bbox_list)
    y1 = min(box[1] for box in bbox_list)
    x2 = max(box[2] for box in bbox_list)
    y2 = max(box[3] for box in bbox_list)
    return [x1, y1, x2, y2]

def resize_keypoints(keypoints, original_size, new_size):
    """Resizes keypoint coordinates from an original image size to a new image size."""
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()

    if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 2 or keypoints.shape[1] != 2:
        return np.empty((0, 2), dtype=np.int32)

    original_height, original_width = original_size
    new_height, new_width = new_size

    if original_width <= 0 or original_height <= 0:
         return np.empty((0, 2), dtype=np.int32)

    scale_x = new_width / original_width if original_width > 0 else 1.0
    scale_y = new_height / original_height if original_height > 0 else 1.0

    keypoints_float = keypoints.astype(np.float64)
    resized_keypoints = keypoints_float * np.array([scale_x, scale_y], dtype=np.float64)

    resized_keypoints[:, 0] = np.clip(resized_keypoints[:, 0], 0, new_width - 1)
    resized_keypoints[:, 1] = np.clip(resized_keypoints[:, 1], 0, new_height - 1)

    return resized_keypoints.astype(np.int32)


# Optimized Function to process a single video
def process_video_optimized(video_path):
    """
    Processes a single video, reads frames sequentially, extracts crops, and
    batches DINO inference on the fly to manage memory.
    """
    global GLOBAL_MODEL, ARGS, GLOBAL_ERROR_LIST, DEVICE

    json_path = video_path.replace('dataset','data_cropped_new').replace('.mp4', '_seg.json')
    tensor_path = video_path.replace('dataset','data_tensor').replace('.mp4', '_results.pt')
    output_tensor_path = video_path.replace('dataset', 'dino_tensor').replace('.mp4', '_results.pt')

    # --- Check if output already exists ---
    if os.path.exists(output_tensor_path):
        return None

    # --- Load required metadata (JSON and Input Tensor Data) ---
    try:
        if not os.path.exists(tensor_path):
            msg = f"Input tensor file missing: {tensor_path}"
            print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
            GLOBAL_ERROR_LIST.append(video_path + " - " + msg)
            return video_path

        pred_data = torch.load(tensor_path, map_location='cpu')
        if 'frame_idx' not in pred_data or 'kp_2d' not in pred_data:
             msg = f"Input tensor file missing expected keys ('frame_idx' or 'kp_2d'): {tensor_path}"
             print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
             GLOBAL_ERROR_LIST.append(video_path + " - " + msg)
             del pred_data
             gc.collect()
             return video_path

        tensor_frame_idx_list = pred_data['frame_idx'].tolist()

        if not os.path.exists(json_path):
            msg = f"Segmentation JSON missing: {json_path}"
            print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
            GLOBAL_ERROR_LIST.append(video_path + " - " + msg)
            del pred_data
            gc.collect()
            return video_path

        with open(json_path, 'r') as f:
            frame_data_list = json.load(f)

        if not frame_data_list:
            dir_name = os.path.dirname(output_tensor_path)
            os.makedirs(dir_name, exist_ok=True)
            embed_dim = GLOBAL_MODEL.embed_dim if hasattr(GLOBAL_MODEL, 'embed_dim') else 1024
            # torch.save({'frame_idx': torch.empty(0, dtype=torch.long), 'x_norm_clstoken': torch.empty((0, embed_dim))}, output_tensor_path)
            del pred_data
            gc.collect()
            return None

        if isinstance(frame_data_list, dict):
             frame_data_dict = {int(entry_key): entry_data for entry_key, entry_data in frame_data_list.items()}
        # elif isinstance(frame_data_list, list):
        #      frame_data_dict = {}
        #      for entry in frame_data_list:
        #          if 'frame_idx' in entry:
        #              frame_data_dict[entry['frame_idx']] = entry
        #          else:
        #              print(f"Worker {os.getpid()}: Warning - JSON entry missing 'frame_idx' in {json_path}: {entry}", file=sys.stderr)
        else:
             msg = f"Could not parse valid frame entries from JSON (expected dict or list): {json_path}"
             print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
             GLOBAL_ERROR_LIST.append(video_path + " - " + msg)
             del pred_data
             gc.collect()
             return video_path


        if not frame_data_dict:
             msg = f"Could not parse valid frame entries from JSON: {json_path}"
             print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
             GLOBAL_ERROR_LIST.append(video_path + " - " + msg)
             del pred_data
             gc.collect()
             return video_path

        needed_frame_indices = sorted(list(frame_data_dict.keys()))
        needed_frame_set = set(needed_frame_indices)

        tensor_idx_map = {frame_idx: i for i, frame_idx in enumerate(tensor_frame_idx_list)}

    except Exception as e:
        msg = f"Error loading metadata for {video_path}: {e}\n{traceback.format_exc()}"
        print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
        GLOBAL_ERROR_LIST.append(video_path + " - " + msg)
        if 'pred_data' in locals(): del pred_data
        if 'frame_data_dict' in locals(): del frame_data_dict
        if 'needed_frame_set' in locals(): del needed_frame_set
        if 'tensor_idx_map' in locals(): del tensor_idx_map
        gc.collect()
        return video_path

    # --- Open Video and Process Frames Sequentially ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Error opening video {video_path}"
        print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
        GLOBAL_ERROR_LIST.append(video_path + " - " + msg)
        del pred_data, frame_data_dict, needed_frame_set, tensor_idx_map
        gc.collect()
        return video_path

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    current_frame_idx = -1

    current_dino_batch_tensors = []
    current_dino_batch_indices = []

    all_video_dino_embeddings = []
    all_video_frame_indices = []


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_idx += 1

        if current_frame_idx not in needed_frame_set:
            del frame
            continue

        # --- Process the frame ---
        try:
            frame_data = frame_data_dict.get(current_frame_idx)
            tensor_idx = tensor_idx_map.get(current_frame_idx)

            if frame_data is None or tensor_idx is None:
                # print(f"Worker {os.getpid()}: Warning - Data mismatch for frame {current_frame_idx} in {video_path}. Skipping frame.", file=sys.stderr)
                del frame
                continue

            horse_entries = [entry for entry in frame_data if entry.get('class_id') == 17]

            if not horse_entries:
                del frame
                continue

            horse_bboxs = [entry['bbox'] for entry in frame_data if 'bbox' in entry and entry['bbox'] is not None and len(entry['bbox']) == 4]
            horse_masks_polygons = [entry['segmentation'] for entry in frame_data if 'segmentation' in entry and entry['segmentation'] is not None]

            if not horse_bboxs or not horse_masks_polygons:
                 print(f"Worker {os.getpid()}: Warning - Missing valid bbox or segmentation data for horse in frame {current_frame_idx}. Skipping frame.", file=sys.stderr)
                 del frame
                 continue

            super_bbox = get_super_bbox(horse_bboxs)
            if super_bbox is None:
                 print(f"Worker {os.getpid()}: Warning - Could not get super bbox for frame {current_frame_idx}. Skipping frame.", file=sys.stderr)
                 del frame
                 continue

            expanded_bbox = expand_bbox(super_bbox, frame_width, frame_height, ARGS.bbox_expand_ratio)
            if expanded_bbox is None:
                print(f"Worker {os.getpid()}: Warning - Expanded bbox is invalid for frame {current_frame_idx}. Skipping frame.", file=sys.stderr)
                del frame
                continue

            x1, y1, x2, y2 = expanded_bbox

            combined_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            for mask_polygon in horse_masks_polygons:
                try:
                    points = np.asarray(mask_polygon, dtype=np.int32)
                    if points.ndim == 2 and points.shape[1] == 2:
                         cv2.fillPoly(combined_mask, [points], 255)
                    elif points.ndim == 3 and points.shape[1] == 1 and points.shape[2] == 2:
                        cv2.fillPoly(combined_mask, [points], 255)
                    else:
                        print(f"Worker {os.getpid()}: Warning - Unexpected mask polygon shape {points.shape} for frame {current_frame_idx}. Skipping this mask.", file=sys.stderr)
                        continue
                except Exception as e:
                    print(f"Worker {os.getpid()}: Error processing segmentation mask for frame {current_frame_idx}: {e}\n{traceback.format_exc()}", file=sys.stderr)

            # --- Add region below a specific keypoint if data available ---
            kp_2d_orig_data = np.empty((0, 2), dtype=np.float32) # Default to empty
            if 'kp_2d' in pred_data and pred_data['kp_2d'].ndim == 3 and tensor_idx < pred_data['kp_2d'].shape[0]:
                 kp_2d_orig_data = pred_data['kp_2d'][tensor_idx]
            else:
                 pass # Warning printed in metadata loading

            if kp_2d_orig_data.ndim == 2 and kp_2d_orig_data.shape[1] == 2 and kp_2d_orig_data.shape[0] > 0:
                kp_2d_resized = resize_keypoints(kp_2d_orig_data, (ARGS.imgsize, ARGS.imgsize), (frame_height, frame_width))
                reference_keypoint_index = 14
                if reference_keypoint_index < len(kp_2d_resized):
                    ref_y = int(np.round(kp_2d_resized[reference_keypoint_index, 1]))
                    ref_y = max(0, min(ref_y, frame_height - 1))
                    below_keypoint_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    below_keypoint_mask[ref_y:, :] = 255
                    final_keep_mask = cv2.bitwise_or(combined_mask, below_keypoint_mask)
                else:
                    final_keep_mask = combined_mask
            else:
                 final_keep_mask = combined_mask

            masked_frame = cv2.bitwise_and(frame, frame, mask=final_keep_mask)
            crop = masked_frame[y1:y2, x1:x2]

            del frame

            if crop.size == 0 or crop.shape[0] <= 0 or crop.shape[1] <= 0:
                print(f"Worker {os.getpid()}: Warning - Crop is empty or invalid dimensions ({crop.shape}) for frame {current_frame_idx}. Skipping frame.", file=sys.stderr)
                if 'crop' in locals(): del crop
                continue

            # Prepare for DINO
            try:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img_tensor = dino_transform(crop_rgb)
                del crop, crop_rgb
                gc.collect()

                current_dino_batch_tensors.append(img_tensor)
                current_dino_batch_indices.append(current_frame_idx)

            except Exception as e:
                print(f"Worker {os.getpid()}: Error converting/transforming crop for frame {current_frame_idx}: {e}\n{traceback.format_exc()}", file=sys.stderr)
                continue

            # --- Check if the DINO batch is full ---
            if len(current_dino_batch_tensors) >= ARGS.dino_batch_size:
                batch_tensors = torch.stack(current_dino_batch_tensors)
                batch_indices = current_dino_batch_indices

                try:
                    embeddings = get_dino_embedding_batch(batch_tensors, GLOBAL_MODEL)
                    all_video_dino_embeddings.append(embeddings.cpu())
                    all_video_frame_indices.extend(batch_indices)
                except Exception as e:
                     msg = f"Error during DINO inference batch (frames {batch_indices[0]}-{batch_indices[-1] if batch_indices else 'N/A'}) for {video_path}: {e}\n{traceback.format_exc()}"
                     print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
                     GLOBAL_ERROR_LIST.append(video_path + " - " + msg)

                del batch_tensors, current_dino_batch_tensors, current_dino_batch_indices, embeddings
                current_dino_batch_tensors = []
                current_dino_batch_indices = []
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            msg = f"Unexpected error processing frame {current_frame_idx} for {video_path}: {e}\n{traceback.format_exc()}"
            print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
            GLOBAL_ERROR_LIST.append(video_path + " - " + msg)


    # --- Finish processing any remaining tensors in the last batch ---
    if current_dino_batch_tensors:
        batch_tensors = torch.stack(current_dino_batch_tensors)
        batch_indices = current_dino_batch_indices

        try:
            embeddings = get_dino_embedding_batch(batch_tensors, GLOBAL_MODEL)
            all_video_dino_embeddings.append(embeddings.cpu())
            all_video_frame_indices.extend(batch_indices)
        except Exception as e:
            msg = f"Error during final DINO inference batch (frames {batch_indices[0]}-{batch_indices[-1] if batch_indices else 'N/A'}) for {video_path}: {e}\n{traceback.format_exc()}"
            print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
            GLOBAL_ERROR_LIST.append(video_path + " - " + msg)

        del batch_tensors, embeddings
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    # --- Release video capture and clean up frame data ---
    cap.release()
    del pred_data, frame_data_dict, needed_frame_set, tensor_idx_map
    del current_dino_batch_tensors, current_dino_batch_indices
    gc.collect()

    # --- Combine results and Save ---
    output_saved_successfully = False
    try:
        dir_name = os.path.dirname(output_tensor_path)
        os.makedirs(dir_name, exist_ok=True)

        combined_embeddings = torch.cat(all_video_dino_embeddings, dim=0)
        combined_indices = torch.tensor(all_video_frame_indices, dtype=torch.long)
        sorted_indices, sort_order = combined_indices.sort()
        sorted_embeddings = combined_embeddings[sort_order]

        final_results = {
            'frame_idx': sorted_indices,
            'x_norm_clstoken': sorted_embeddings
        }

        torch.save(final_results, output_tensor_path)
        output_saved_successfully = True
        print(f"Worker {os.getpid()}: Successfully saved DINO results")

    except Exception as e:
        msg = f"Error saving results for {video_path} to {output_tensor_path}: {e}\n{traceback.format_exc()}"
        print(f"Worker {os.getpid()}: {msg}", file=sys.stderr)
        GLOBAL_ERROR_LIST.append(video_path + " - " + msg)

    del all_video_dino_embeddings, all_video_frame_indices
    if 'combined_embeddings' in locals(): del combined_embeddings
    if 'combined_indices' in locals(): del combined_indices
    if 'final_results' in locals(): del final_results
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return video_path if not output_saved_successfully else None


def process_video_wrapper(video_path):
    return process_video_optimized(video_path)

# Main execution with multiprocessing
if __name__ == "__main__":
    args = parse_args()
    args = set_default_args(args)

    video_root = os.path.join(args.data_dir, 'dataset')

    print("Listing MP4 files and checking for required inputs/outputs...")
    main_video_list = sorted(list_mp4_files(video_root))
    print(f"Found {len(main_video_list)} videos matching criteria (needs processing, inputs exist).")

    error_log_path = args.error_log_file
    previous_errors = set()
    if os.path.exists(error_log_path):
        try:
            with open(error_log_path, 'rb') as f:
                loaded_errors = pickle.load(f)
                if isinstance(loaded_errors, list):
                     previous_errors = set(loaded_errors)
                else:
                    print(f"Warning: Error log {error_log_path} format unexpected.", file=sys.stderr)

            print(f"Loaded {len(previous_errors)} videos that errored in previous runs.")
        except Exception as e:
            print(f"Error loading previous error log {error_log_path}: {e}", file=sys.stderr)
            previous_errors = set()

    removed_videos_set = set()
    if os.path.exists('/home/ubuntu/workspace/running_videos.csv'):
        try:
            df = pd.read_csv('/home/ubuntu/workspace/running_videos.csv')
            if not df.empty:
                removed_videos_set = set(df.values.flatten().tolist())
            print(f"Loaded {len(removed_videos_set)} videos from running_videos.csv.")
        except Exception as e:
             print(f"Error loading running_videos.csv: {e}", file=sys.stderr)

    fully_excluded_videos = removed_videos_set.union(previous_errors)
    video_list_to_process = [path for path in main_video_list if path not in fully_excluded_videos]

    if args.limit_videos is not None:
        print(f"Limiting processing to the first {args.limit_videos} videos.")
        video_list_to_process = video_list_to_process[:args.limit_videos]

    print(f"Processing {len(video_list_to_process)} videos after filtering.")

    if not video_list_to_process:
        print("No videos to process. Exiting.")
        sys.exit(0)

    manager = mp.Manager()
    shared_error_list = manager.list()
    args_dict = vars(args)

    try:
        print(f"Starting multiprocessing pool with {args.num_workers} workers...")
        with mp.Pool(processes=args.num_workers,
                     initializer=init_worker,
                     initargs=(args.dino_model_name, args_dict, shared_error_list)) as pool:

            errored_videos_in_run = [result for result in tqdm(pool.imap(process_video_wrapper, video_list_to_process), total=len(video_list_to_process), desc="Processing Videos") if result is not None]

        print("Pool finished.")

    except Exception as e:
        print(f"An error occurred during multiprocessing pool execution: {e}\n{traceback.format_exc()}", file=sys.stderr)
        errored_videos_in_run = []

    final_errors_messages = list(shared_error_list)

    errored_video_paths_from_messages = []
    for msg in final_errors_messages:
         if msg.startswith('/home/ubuntu/workspace/dataset/'):
              errored_video_paths_from_messages.append(msg.split(' - ')[0])
         elif msg.startswith('Worker '):
              if ' processing ' in msg:
                  try:
                      path_part = msg.split(' processing ')[1].split('...')[0]
                      if path_part.endswith('.mp4'):
                           errored_video_paths_from_messages.append(path_part)
                  except:
                       pass

    all_errored_video_paths = set(errored_videos_in_run).union(set(errored_video_paths_from_messages))

    if final_errors_messages:
        print(f"\n--- Detailed Error Log ({len(final_errors_messages)} messages) ---", file=sys.stderr)
        for error_msg in final_errors_messages:
            print(error_msg, file=sys.stderr)
        print("---------------------------------------------", file=sys.stderr)

    if all_errored_video_paths:
        print(f"\nFound {len(all_errored_video_paths)} unique videos with errors. Logging paths to {error_log_path}")
        try:
            existing_errors_for_saving = []
            if os.path.exists(error_log_path):
                 with open(error_log_path, 'rb') as f:
                      existing_errors_for_saving = pickle.load(f)
                 if not isinstance(existing_errors_for_saving, list):
                     print(f"Warning: Existing error log {error_log_path} has unexpected format, overwriting.", file=sys.stderr)
                     existing_errors_for_saving = []

            combined_errors_for_saving = list(set(existing_errors_for_saving).union(all_errored_video_paths))

            with open(error_log_path, 'wb') as f:
                pickle.dump(combined_errors_for_saving, f)
            print(f"Updated error video paths saved to {error_log_path}")
        except Exception as e:
            print(f"Error saving error log to {error_log_path}: {e}\n{traceback.format_exc()}", file=sys.stderr)
    else:
        print("\nNo videos reported processing errors.")

    print("Video processing completed.")