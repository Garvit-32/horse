import argparse
import sys, os
# sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
# from src.trainer.vanillaupdatetrainer import InferenceModel
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
import gc                       # NEW – explicit garbage collection
cv2.setNumThreads(0)  

def _flush_dino_buffer(buf_imgs, buf_idxs, out_embs, out_idxs):
    global GLOBAL_MODEL  
    """
    buf_imgs / buf_idxs  : the current crop tensors and their frame-indices
    out_embs / out_idxs  : the *global* lists we append results to
    """
    if not buf_imgs:                       # nothing to do
        return

    batch = torch.stack(buf_imgs).to(next(GLOBAL_MODEL.parameters()).device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        emb = GLOBAL_MODEL.forward_features(batch)['x_norm_clstoken']

    out_embs.append(emb.cpu())             # keep only on CPU
    out_idxs.append(buf_idxs.copy())

    # ---- release GPU / RAM immediately ----------
    del batch, emb
    buf_imgs.clear(); buf_idxs.clear()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()


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
    parser.add_argument('--save_dir', type=str, default = 'videos',  help="Root directory to save processed videos")
    parser.add_argument('--prototype_path', type=str, default='/home/ubuntu/workspace/prototypes.pth', help="Path to pre-computed DINO prototypes")
    parser.add_argument('--ckpt_file', type=str, help="Path to DESSIE checkpoint (if used, currently inactive)") # Kept if needed later

    # --- Processing Parameters ---
    parser.add_argument('--num_workers', type=int, default=12, help="Number of worker processes for DataLoader")
    parser.add_argument('--dino_batch_size', type=int, default=256, help="Batch size for DINO inference")
    parser.add_argument('--imgsize', type=int, default=256, help="Target size for cropping/resizing (before DINO)") # Keep if used elsewhere
    parser.add_argument('--output_fps', type=int, default=20, help="FPS for the output videos")
    parser.add_argument('--orientation_conf', type=float, default=0.5, help="Confidence threshold for kp-based orientation dx vs dz")
    parser.add_argument('--movement_threshold', type=float, default=1.5, help="Avg keypoint movement threshold (pixels) to determine if horse is moving")
    parser.add_argument('--smoothing_window', type=int, default=5, help="Window size for temporal smoothing of orientation")
    parser.add_argument('--bbox_expand_ratio', type=float, default=0.05, help="Ratio to expand bounding boxes")

    # --- Model & Feature Parameters ---
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vitb14", help="DINOv2 model name")

    # --- Debugging/Testing ---
    parser.add_argument('--limit_videos', type=int, default=None, help="Process only the first N videos for testing")
    parser.add_argument('--error_log_file', type=str, default='processing_errors.pkl', help="File to save list of errored video paths")

    # --- Arguments kept from original but potentially unused with current DINO focus ---
    # These are kept for potential future use or if parts of DESSIE logic are re-enabled
    parser.add_argument('--train', type=str)
    parser.add_argument('--model_dir', type=str, default='/home/ubuntu/workspace/Dessie/internal_models') # Example path
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--version', type=str)
    parser.add_argument('--ModelName', type=str, default='DESSIE') # Inactive
    parser.add_argument('--DatasetName', type=str, default='DessiePIPE') # Inactive
    parser.add_argument('--batch_size', type=int, default=1, help="DataLoader batch size (usually 1 for video processing)") # Renamed for clarity
    parser.add_argument('--W_shape_prior', default=50., type=float) # Inactive weights
    parser.add_argument('--W_kp_img', default=0.001, type=float)
    parser.add_argument('--W_mask_img', default=0.0001, type=float)
    parser.add_argument('--W_pose_img', default=0.01, type=float)
    # ... other weights ...
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--PosePath', type=str)
    parser.add_argument("--TEXTUREPath", type=str)
    parser.add_argument('--uv_size', type=int, default=256)
    parser.add_argument('--data_batch_size', type=int, default=1) # Potentially confusing name, keep if needed
    parser.add_argument('--useSynData', action="store_true")
    parser.add_argument('--useinterval', type=int, default=8)
    parser.add_argument("--getPairs", action="store_true", default=False)
    parser.add_argument("--TEXT", action="store_true", default=False)
    parser.add_argument("--DINO_frozen", action="store_true", default=False)
    parser.add_argument("--DINO_obtain_token", action="store_true", default=False)
    parser.add_argument("--GT", action="store_true", default=False)
    parser.add_argument('--W_gt_shape', default=0, type=float)
    parser.add_argument('--W_gt_pose', default=0., type=float)
    parser.add_argument('--W_gt_trans', default=0., type=float)
    parser.add_argument("--pred_trans", action="store_true", default=False)
    parser.add_argument("--background", action="store_true", default=False)
    parser.add_argument("--background_path", default='')
    parser.add_argument("--REALDATASET", default='MagicPony')
    parser.add_argument("--REALPATH", default='')
    parser.add_argument("--web_images_num", type=int, default=0)
    parser.add_argument("--REALMagicPonyPATH", default='')

    return parser.parse_args()


def set_default_args(args):
    ########################same for all model#######################################
    args.DatasetName='DessiePIPE'
    args.ModelName = 'DESSIE'
    args.name='TOTALRANDOM'
    args.useSynData= True
    args.TEXT = True
    args.DINO_model_name='dino_vits8'
    args.imgsize=256
    args.GT = True
    args.pred_trans = True
    args.W_shape_prior=0.0
    args.W_kp_img=0.0
    args.W_pose_img=0.0
    args.W_mask_img=0.0
    args.W_cos_shape=0.
    args.W_cos_pose=0.
    args.W_text_shape=0.
    args.W_text_pose=0.
    args.W_text_cam=0.
    args.W_cosine_text_shape=0.0
    args.W_cosine_text_pose=0.0
    args.W_cosine_text_cam=0.0
    args.W_gt_shape=0.
    args.W_gt_pose=0.
    args.W_gt_trans=0.
    args.W_l2_shape_1=0.0
    args.W_l2_pose_2=0.0
    args.W_l2_shape_3=0.0
    args.W_l2_pose_3=0.0
    args.W_l2_rootrot_1=0.0
    args.W_l2_rootrot_2=0.0
    args.batch_size = 1
    args.data_batch_size=1
    args.getPairs = True
    args.model_dir = '/home/ubuntu/workspace/Dessie/internal_models'
    return args

# Function to convert 3D keypoints to 2D
def convert_3d_to_2d(points, size=256):
    '''
    Input:
        points: numpy array of shape (V, 3)
    Output:
        proj_points: numpy array of shape (V, 2) - projected 2D points
    '''

    focal = 5000

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Perspective Projection
    x_proj = (focal * x) / z
    # y_proj = (focal * y) / z

    # Convert to pixel coordinates
    screen_x = (size / 2) * (1 + x_proj / (size / 2))
    # screen_y = (size / 2) * (1 - y_proj / (size / 2))

    R = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]])

    T = np.zeros(3)

    # 1. World to Camera transformation
    points_camera = (points @ R.T) + T  # (V, 3)

    # 2. Perspective projection
    # x_camera = points_camera[:, 0]
    y_camera = points_camera[:, 1]
    z_camera = points_camera[:, 2]

    # x_screen = (focal * x_camera) / z_camera
    y_screen = (focal * y_camera) / z_camera

    # 4. Scale to image size (assuming NDC [-size/2, size/2] maps to image [0, size])
    # x_image = (x_screen / (size/2) ) * (size / 2) + (size / 2)
    y_image = (y_screen / (size/2) ) * (size / 2) + (size / 2)

    # In OpenGL, +Y is up and image +y is down, so flip y coordinate
    y_image = size - y_image

    proj_points = np.stack([screen_x, y_image], axis=-1)

    return proj_points.astype(np.int32)

def list_mp4_files(root_dir):
    """
    Recursively lists all .mp4 files in a directory, excluding files that match specific patterns.
    """
    mp4_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                dino_path = os.path.join(dirpath, filename).replace('dataset', 'dino_tensor').replace('.mp4', '_results.pt')
                if not os.path.exists(dino_path):
                    mp4_files.append(os.path.join(dirpath, filename))
    return mp4_files

def expand_bbox(bbox, frame_width, frame_height, expand_ratio=0.05):

    x1, y1, x2, y2 = bbox

    # Calculate width and height of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Calculate expansion amount (10% of width & height)
    expand_w = int(bbox_width * expand_ratio)
    expand_h = int(bbox_height * expand_ratio)

    # Expand the bounding box
    x1_new = max(0, x1 - expand_w)  # Ensure x1 is not negative
    y1_new = max(0, y1 - expand_h)  # Ensure y1 is not negative
    x2_new = min(frame_width, x2 + expand_w)  # Ensure x2 does not exceed frame width
    y2_new = min(frame_height, y2 + expand_h)  # Ensure y2 does not exceed frame height

    return [x1_new, y1_new, x2_new, y2_new]

def get_dino_embedding_batch(image_batch_tensor, model):
    """Processes a batch of image tensors with DINO."""
    device = next(model.parameters()).device
    image_batch_tensor = image_batch_tensor.to(device)
    with torch.no_grad():
        # , torch.cuda.amp.autocast(device_type='cuda',enabled=True)
        embeddings = model.forward_features(image_batch_tensor)['x_norm_clstoken']
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings # Returns embeddings on GPU

def init_worker(model_name,args_dict, error_list):
    global GLOBAL_MODEL,  GLOBAL_ERROR_LIST, ARGS
    # print(f"Initializing worker {os.getpid()}...")
    # Use Manager proxy objects within the worker
    # MANAGER = manager_proxy
    GLOBAL_ERROR_LIST = error_list
    ARGS = argparse.Namespace(**args_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load DINO Model
        GLOBAL_MODEL = torch.hub.load('facebookresearch/dinov2',model_name, pretrained=True)
        GLOBAL_MODEL.eval()
        GLOBAL_MODEL.to(device)

    except Exception as e:
        # Log the error centrally if possible, or raise to stop the worker
        # GLOBAL_ERROR_LIST.append(/)
        import traceback
        traceback.print_exc() # Print traceback for debugging
        raise e # Raising exception might stop the DataLoader pool gracefully

def get_super_bbox(bbox_list):
    x1 = min(box[0] for box in bbox_list)
    y1 = min(box[1] for box in bbox_list)
    x2 = max(box[2] for box in bbox_list)
    y2 = max(box[3] for box in bbox_list)
    return [x1, y1, x2, y2]

def resize_keypoints(keypoints, original_size, new_size):
    """Resizes keypoint coordinates from an original image size to a new image size."""
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy() # Ensure NumPy array on CPU

    if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 2 or keypoints.shape[1] != 2:
        # Handle case where keypoints might be empty or invalid
        return np.empty((0, 2), dtype=np.int32)
        # raise ValueError("keypoints must be a NumPy array of shape (N, 2)") # Or return empty

    original_height, original_width = original_size
    new_height, new_width = new_size

    if original_width <= 0 or original_height <= 0:
         return np.empty((0, 2), dtype=np.int32) # Or raise error
         # raise ValueError("Original dimensions must be positive.")

    keypoints_float = keypoints.astype(np.float64)

    scale_factors = np.array([
        new_width / original_width if original_width > 0 else 1,
        new_height / original_height if original_height > 0 else 1
    ], dtype=np.float64)

    resized_keypoints = keypoints_float * scale_factors

    # Clamp coordinates to be within the new image bounds
    resized_keypoints[:, 0] = np.clip(resized_keypoints[:, 0], 0, new_width - 1)
    resized_keypoints[:, 1] = np.clip(resized_keypoints[:, 1], 0, new_height - 1)

    return resized_keypoints.astype(np.int32)



# Optimized Function to process a single video (now with full sequential frame read)
def process_video_optimized(video_path, batch_size=16):
    """
    Processes a single video, optimized for subprocess execution and full sequential frame reading.
    Args:
        video_path (str): Path to the video file.
        batch_size (int): Batch size for processing frames.
    """
    global GLOBAL_MODEL, ARGS, GLOBAL_ERROR_LIST


    json_path = video_path.replace('dataset','data_cropped_new').replace('.mp4', '_seg.json')
    tensor_path = video_path.replace('dataset','data_tensor').replace('.mp4', '_results.pt')


    try:
        if not os.path.exists(tensor_path): raise FileNotFoundError("Tensor file missing")
        pred_data = torch.load(tensor_path, map_location='cpu') # Load to CPU first
        
        if not os.path.exists(json_path): raise FileNotFoundError("JSON file missing")
        with open(json_path, 'r') as f:
            frame_data = json.load(f) # List of dicts {frame_idx, bbox, segmentation}

        if not frame_data:
            # print(f"Warning [{worker_pid}]: No frame entries in JSON {json_path}")
            return None # Success, but nothing to process

    except Exception as e:
        msg = f"Error loading data for {tensor_path}: {e}"
        print(msg)
        GLOBAL_ERROR_LIST.append(tensor_path)
        return tensor_path

    # --- 3. Prepare for Frame Processing ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Error opening video {video_path}"
        print(msg)
        GLOBAL_ERROR_LIST.append(video_path)
        return tensor_path

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- 4. Read Needed Frames ---
    frames_dict = {}
    needed_frame_indices = sorted([int(key) for key in frame_data.keys()])
    if not needed_frame_indices:
         cap.release()
         return None # No frames specified

    max_frame_idx = needed_frame_indices[-1]
    current_frame_idx = 0
    needed_set = set(needed_frame_indices)


    
    while current_frame_idx <= max_frame_idx:
        ret, frame = cap.read()
        if not ret:
            break # Reached end of video
        if current_frame_idx in needed_set:
            frames_dict[current_frame_idx] = frame # Keep frame in BGR
        current_frame_idx += 1
    cap.release() # Release video capture handle
    

    
    

    # --- 5. Prepare Crops for Batch DINO Inference ---
    dino_input_batches = []
    original_indices = [] # Keep track of which frame/entry corresponds to each crop

    # Two *global* result collectors – stay small because on CPU
    all_dino_preds = []       # list of tensors, each (B, 768)
    all_idxs       = []       # list of python lists


    # tensor_idx_map = {entry['frame_idx']: i for i, entry in enumerate(frame_data)}

    try:

        tensor_frame_idx_list = pred_data['frame_idx'].tolist()
    except:
        msg = f"Error no frame_idx {video_path}"
        print(msg)
        GLOBAL_ERROR_LIST.append(video_path)
        return tensor_path




    # 3. Frame Processing Loop (access frames from indexed_frames)
    for frame_idx, data in frame_data.items():
        frame_idx = int(frame_idx)
        # frame_idx = int(frame_idx) % len(tensor_frame_idx_list)
        if frame_idx not in tensor_frame_idx_list: continue
        tensor_idx = tensor_frame_idx_list.index(frame_idx)
        

        frame = frames_dict.get(frame_idx) # Efficiently retrieve pre-read frame

        masks = []
        bboxs = []
        horse_found = False
        
        for class_dict in data:
            if class_dict['class_id'] == 17:
                horse_found = True
            horse_mask = class_dict['segmentation']
            bbox = class_dict['bbox']
            masks.append(horse_mask)
            bboxs.append(bbox)

        if not horse_found:
            print('Horse not found in frame:', frame_idx)
            continue # Skip if no horse found #TODO: Frames where horses skipped, write in a text/other file. 

        bbox = get_super_bbox(bboxs)
        x1, y1, x2, y2 = expand_bbox(bbox, frame_width, frame_height, ARGS.bbox_expand_ratio)
        if x1 >= x2 or y1 >= y2: continue # Invalid bbox

        # Create mask based on segmentation (and optionally keypoints)
        # This part is kept similar to the original logic
        horse_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        try:
            for mask in masks:
                points = np.asarray(mask, dtype=np.int32)
                cv2.fillPoly(horse_mask, [points], (255))
        except Exception as e:
            print(f"Error processing segmentation for frame {frame_idx}: {e}")
            continue # Skip this frame if segmentation is bad


        kp_2d_orig = pred_data['kp_2d'][tensor_idx] # KPs relative to 256x256 crop
        kp_2d_resized = resize_keypoints(kp_2d_orig, (ARGS.imgsize, ARGS.imgsize), (frame_height, frame_width))
        reference_keypoint_index = 14 # Example
        if reference_keypoint_index < len(kp_2d_resized):
            _, ref_y = kp_2d_resized[reference_keypoint_index]
            ref_y = max(0, min(ref_y, frame_height - 1))
            below_keypoint_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            below_keypoint_mask[ref_y:, :] = 255
            final_keep_mask = cv2.bitwise_or(horse_mask, below_keypoint_mask)
        else:
            final_keep_mask = horse_mask # Fallback if keypoint is missing

        # Apply mask and crop
        masked_frame = cv2.bitwise_and(frame, frame, mask=final_keep_mask)
        crop = masked_frame[y1:y2, x1:x2]

        if crop.size == 0: continue # Skip if crop is empty

        # Prepare for DINO: Convert BGR crop to RGB PIL Image
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_tensor = dino_transform(crop_rgb) # Apply DINO transforms
            dino_input_batches.append(img_tensor)
            original_indices.append(frame_idx) # Store the frame index

            if len(dino_input_batches) >= ARGS.dino_batch_size:
                _flush_dino_buffer(dino_input_batches,
                                   original_indices,
                                   all_dino_preds,
                                   all_idxs)
            # print("in loop len all_dino_pred and idx",len(all_dino_preds),len(all_idxs))

        except Exception as e:
            print(f"Error converting/transforming crop for frame {frame_idx}: {e}")
            continue

    _flush_dino_buffer(dino_input_batches,
                       original_indices,
                       all_dino_preds,
                       all_idxs)

    if not all_dino_preds:                 # nothing was processed
        print(f"No valid crops generated for DINO in {video_path}")
        del frames_dict
        return None


# --- 6. Perform Batched DINO Inference ---


    try:

        flattened_idxs = (np.concatenate([np.asarray(x, np.int32).ravel()
                                          for x in all_idxs])
                          if all_idxs else np.empty(0, np.int32))
        # print("total finasl len",final_results["x_norm_clstoken"].shape[0])


        final_results = {
            'frame_idx': torch.tensor(flattened_idxs),
            'x_norm_clstoken': torch.cat(all_dino_preds, dim=0).cpu()
        }
        print("len final",final_results["x_norm_clstoken"].shape[0])
        assert final_results["frame_idx"].numel() == final_results["x_norm_clstoken"].shape[0], \
            f"Row mismatch in {video_path}"
        dir_name = os.path.dirname(video_path).replace('dataset', 'dino_tensor')
        os.makedirs(dir_name, exist_ok=True)
        save_path = video_path.replace('dataset', 'dino_tensor').replace('.mp4',
                                                                        '_results.pt')
        torch.save(final_results, save_path)

    

    except Exception as e:
         msg = f"Error during batched DINO inference for {video_path}: {e}"
         print(msg)
         GLOBAL_ERROR_LIST.append(video_path)
         del frames_dict
         del dino_input_batches
         del all_dino_preds
         del all_idxs
         gc.collect()
         return tensor_path # Indicate failure

    # Free memory used by input tensors
    del dino_input_batches
    
    return None




def process_video_wrapper(args):
    return process_video_optimized(*args)

# Main execution with multiprocessing
if __name__ == "__main__":
    args = parse_args()
    args = set_default_args(args)

    model_path = "/home/ubuntu/workspace/Dessie/COMBINAREAL/version_8/checkpoints/best.ckpt"
    video_root = '/home/ubuntu/workspace/dataset'
    main_video_list = sorted(list_mp4_files(video_root))

    df = pd.read_csv('/home/ubuntu/workspace/running_videos.csv')
    removed_videos = set(df.values.flatten().tolist())

    video_list = [path for path in main_video_list if path not in removed_videos]

    # video_list = ['/home/ubuntu/workspace/dataset/fastiptondata/2020_July-Horses-of-Racing-Age/APPIAN WAY_lot_6/video.mp4']

    

    num_processes = 2 # Number of subprocesses to use - as requested

    manager = mp.Manager()
    shared_error_list = manager.list() # Create a shared list for errors

    # Prepare arguments for subprocesses (video_path, batch_size)
    process_args = [(video_path, 1) for video_path in video_list]
    

    with mp.Pool(processes=num_processes,
                 initializer=init_worker,
                 initargs=('dinov2_vitb14', vars(args),  shared_error_list)) as pool:
        # Use starmap for multiple arguments to process_video_optimized
        list(tqdm(pool.imap(process_video_wrapper, process_args), total=len(process_args)))


    print(shared_error_list)
    with open('error_file_path.pkl', 'wb') as f:
        pickle.dump(list(shared_error_list), f) # Convert shared list to regular list for pickling
    
    print("Video processing completed using multiprocessing.")
    
