import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from src.trainer.vanillaupdatetrainer import InferenceModel
import torch
import json
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pickle
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Normalization transform
img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])
norm_transform = transforms.Normalize(img_mean, img_std)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_file', type=str, required=False, help='checkpoint for resuming')
    parser.add_argument('--train', type=str, required=False, help='Train or Test')
    parser.add_argument('--data_dir', type=str, default=None,help='')
    parser.add_argument('--model_dir', type=str,
                        default='/home/x_cili/x_cili_lic/DESSIE/code/src/SMAL/smpl_models', help='model dir')
    parser.add_argument('--save_dir', type=str, default='/home/x_cili/x_cili_lic/DESSIE/results/model',help='save dir')
    parser.add_argument('--name', type=str, default='test', help='experiment name')
    parser.add_argument('--version', type=str, required=False, help='experiment version')
    parser.add_argument('--ModelName', type=str, help='model name')
    parser.add_argument('--DatasetName', type=str, default='None', help='')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--imgsize', type=int, default=256, help='number of workers')
    parser.add_argument('--W_shape_prior', default=50., type=float, help='shape prior')
    parser.add_argument('--W_kp_img', default=0.001, type=float, help='kp loss for image')
    parser.add_argument('--W_mask_img', default=0.0001, type=float, help='mask loss for image: 0.0001 or 1')
    parser.add_argument('--W_pose_img', default=0.01, type=float, help='pose prior for image')
    parser.add_argument('--W_l2_shape_1', default=0., type=float, help='Dloss latent label1')
    parser.add_argument('--W_l2_pose_2', default=0., type=float, help='Dloss latent label2')
    parser.add_argument('--W_l2_shape_3', default=0., type=float, help='Dloss latent label3')
    parser.add_argument('--W_l2_pose_3', default=0., type=float, help='Dloss latent label3')
    parser.add_argument('--W_l2_rootrot_1', default=0, type=float, help='Dloss value label1')
    parser.add_argument('--W_l2_rootrot_2', default=0, type=float, help='Dloss value label2')
    parser.add_argument('--lr', type=float, default=5e-05, help='optimizer learning rate')
    parser.add_argument('--max-epochs', type=int, default=1000, help='max. number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='max. number of training epochs')
    parser.add_argument('--PosePath', type=str, default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/pose')
    parser.add_argument("--TEXTUREPath", type=str, default="/home/x_cili/x_cili_lic/DESSIE/data/syndata/TEXTure")
    parser.add_argument('--uv_size', type=int, default=256, help='number of workers')
    parser.add_argument('--data_batch_size', type=int, default=2, help='batch size; before is 36')
    parser.add_argument('--useSynData', action="store_true", help="True: use syndataset")
    parser.add_argument('--useinterval', type=int, default=8, help='number of interval of the data')
    parser.add_argument("--getPairs", action="store_true", default=False,help="get image pair with label")
    parser.add_argument("--TEXT", action="store_true", default=False,help="Text label input")
    parser.add_argument("--DINO_model_name", type=str, default="dino_vits8")
    parser.add_argument("--DINO_frozen", action="store_true", default=False,help="frozen DINO")
    parser.add_argument("--DINO_obtain_token", action="store_true", default=False,help="obtain CLS token or use key")
    parser.add_argument("--GT", action="store_true", default=False,help="obtain gt or not")
    parser.add_argument('--W_gt_shape', default=0, type=float, help='weight for gt')
    parser.add_argument('--W_gt_pose', default=0., type=float, help='weight for gt')
    parser.add_argument('--W_gt_trans', default=0., type=float, help='weight for gt')
    parser.add_argument("--pred_trans", action="store_true", default=False,help="model to predict translation or not")
    parser.add_argument("--background", action="store_true", default=False,help="get image pair with label")
    parser.add_argument("--background_path", default='/home/x_cili/x_cili_lic/DESSIE/data/syndata/coco', help="")
    parser.add_argument("--REALDATASET", default='MagicPony', help="Animal3D or MagicPony")
    parser.add_argument("--REALPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/realimg', help="staths dataset")
    parser.add_argument("--web_images_num", type=int, default=0, help="staths dataset")
    parser.add_argument("--REALMagicPonyPATH", default='/home/x_cili/x_cili_lic/DESSIE/data/magicpony', help="magicpony dataset")
    
    # New optimization parameters
    parser.add_argument("--gpu_batch_size", type=int, default=128, help="Maximum batch size for GPU processing")
    parser.add_argument("--videos_per_batch", type=int, default=8, help="Number of videos to process in parallel")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch")
    parser.add_argument("--max_frames_in_memory", type=int, default=1000, help="Maximum frames to keep in memory")
    
    args = parser.parse_args()
    return args

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
    
    # Convert to pixel coordinates
    screen_x = (size / 2) * (1 + x_proj / (size / 2))
    
    R = np.array([[-1, 0, 0],
                 [0, -1, 0],
                 [0, 0, 1]])
    
    T = np.zeros(3)
    
    # World to Camera transformation
    points_camera = (points @ R.T) + T  # (V, 3)
    
    # Perspective projection
    y_camera = points_camera[:, 1]
    z_camera = points_camera[:, 2]
    
    y_screen = (focal * y_camera) / z_camera
    
    # Scale to image size
    y_image = (y_screen / (size/2) ) * (size / 2) + (size / 2)
    
    # In OpenGL, +Y is up and image +y is down, so flip y coordinate
    y_image = size - y_image
    
    proj_points = np.stack([screen_x, y_image], axis=-1)
    
    return proj_points.astype(np.int32)

def list_mp4_files(root_dir):
    """
    Recursively lists all .mp4 files in a directory, excluding files that already have tensor results.
    """
    mp4_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                tensor_path = os.path.join(dirpath, filename).replace('data', 'data_tensor').replace('.mp4', '_results.pt')
                if not os.path.exists(tensor_path):
                    mp4_files.append(os.path.join(dirpath, filename))
    return mp4_files

def expand_bbox(bbox, frame_width, frame_height, expand_ratio=0.05):
    x1, y1, x2, y2 = bbox
    
    # Calculate width and height of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Calculate expansion amount
    expand_w = int(bbox_width * expand_ratio)
    expand_h = int(bbox_height * expand_ratio)
    
    # Expand the bounding box
    x1_new = max(0, x1 - expand_w)
    y1_new = max(0, y1 - expand_h)
    x2_new = min(frame_width, x2 + expand_w)
    y2_new = min(frame_height, y2 + expand_h)
    
    return [x1_new, y1_new, x2_new, y2_new]

class VideoFrameDataset(Dataset):
    """
    Dataset for efficiently loading frames from multiple videos.
    """
    def __init__(self, video_paths, frame_size=256, max_frames_per_video=None):
        self.video_paths = video_paths
        self.frame_size = frame_size
        self.max_frames_per_video = max_frames_per_video
        
        # Extract frame data for each video
        self.all_frame_data = []
        self.error_videos = []
        
        self._preprocess_videos()
    
    def _preprocess_videos(self):
        """Prepare metadata for each video without loading frames yet"""
        for video_path in self.video_paths:
            json_path = video_path.replace('data', 'data_cropped').replace('.mp4', '_seg.json')
            try:
                # Read frame metadata from JSON
                with open(json_path, 'r') as f:
                    frame_data = json.load(f)
                
                # Limit the number of frames if needed
                if self.max_frames_per_video and len(frame_data) > self.max_frames_per_video:
                    frame_data = frame_data[:self.max_frames_per_video]
                
                # Add metadata for each frame
                for entry in frame_data:
                    self.all_frame_data.append({
                        'video_path': video_path,
                        'frame_idx': entry['frame_idx'],
                        'bbox': entry['bbox']
                    })
            except Exception as e:
                print(f"Error preprocessing {video_path}: {e}")
                self.error_videos.append(video_path)
    
    def __len__(self):
        return len(self.all_frame_data)
    
    def __getitem__(self, idx):
        frame_info = self.all_frame_data[idx]
        video_path = frame_info['video_path']
        frame_idx = frame_info['frame_idx']
        bbox = frame_info['bbox']
        
        # We'll use OpenCV to efficiently extract just the frame we need
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Return a placeholder if frame extraction fails
            print(f"Failed to extract frame {frame_idx} from {video_path}")
            placeholder = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.float32)
            return torch.zeros(3, self.frame_size, self.frame_size), video_path, frame_idx
        
        # Process the frame
        h, w, _ = frame.shape
        x1, y1, x2, y2 = expand_bbox(bbox, w, h)
        cropped_frame = frame[y1:y2, x1:x2]
        cropped_frame = cv2.resize(cropped_frame, (self.frame_size, self.frame_size))[:, :, ::-1]  # Convert BGR to RGB
        cropped_frame = cropped_frame / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(cropped_frame).permute(2, 0, 1).float()
        tensor = norm_transform(tensor)
        
        return tensor, video_path, frame_idx

class VideoFrameDatasetWithFrameCache(Dataset):
    """
    Optimized dataset that caches frames from videos to avoid repeated reads.
    Uses a frame cache for each video to optimize memory usage.
    """
    def __init__(self, video_paths, frame_size=256, max_frames_per_video=None, cache_size=30):
        self.video_paths = video_paths
        self.frame_size = frame_size
        self.max_frames_per_video = max_frames_per_video
        self.cache_size = cache_size
        
        # Prepare video metadata and frame information
        self.video_metadata = {}  # stores frames info for each video
        self.frame_data = []      # flattened list of all frames
        self.error_videos = []
        
        # Frame cache
        self.frame_cache = {}     # video_path -> {frame_idx -> tensor}
        self.cache_lock = threading.Lock()
        
        self._preprocess_videos()
    
    def _preprocess_videos(self):
        """Prepare metadata for all videos without loading frames yet"""
        for video_path in tqdm(self.video_paths, desc="Preprocessing videos"):
            json_path = video_path.replace('data', 'data_cropped').replace('.mp4', '_seg.json')
            try:
                # Read frame metadata from JSON
                with open(json_path, 'r') as f:
                    frame_data = json.load(f)
                
                # Limit frames if needed
                if self.max_frames_per_video and len(frame_data) > self.max_frames_per_video:
                    frame_data = frame_data[:self.max_frames_per_video]
                
                # Store metadata for this video
                self.video_metadata[video_path] = frame_data
                
                # Add all frames to flattened list
                for entry in frame_data:
                    self.frame_data.append({
                        'video_path': video_path,
                        'frame_idx': entry['frame_idx'],
                        'bbox': entry['bbox']
                    })
                
                # Initialize cache for this video
                self.frame_cache[video_path] = {}
                
            except Exception as e:
                print(f"Error preprocessing {video_path}: {e}")
                self.error_videos.append(video_path)
    
    def _cache_frames(self, video_path, frame_idxs):
        """Cache multiple frames from the same video in one read operation"""
        if not frame_idxs:
            return
            
        # Check which frames are not in cache
        missing_idxs = []
        for idx in frame_idxs:
            if idx not in self.frame_cache[video_path]:
                missing_idxs.append(idx)
        
        if not missing_idxs:
            return
            
        # Sort missing indices for efficient sequential reading
        missing_idxs.sort()
        
        # Read video and extract missing frames
        cap = cv2.VideoCapture(video_path)
        current_idx = 0
        frame_dict = {}
        
        for target_idx in missing_idxs:
            # Skip frames until we reach the target
            while current_idx < target_idx:
                cap.grab()  # Just grab frame, don't decode
                current_idx += 1
            
            # Read the target frame
            ret, frame = cap.read()
            current_idx += 1
            
            if ret:
                frame_dict[target_idx] = frame
            else:
                print(f"Failed to read frame {target_idx} from {video_path}")
                
        cap.release()
        
        # Update cache with acquired frames
        with self.cache_lock:
            # Add new frames to cache
            for idx, frame in frame_dict.items():
                self.frame_cache[video_path][idx] = frame
                
            # Keep cache size in check
            cache_keys = list(self.frame_cache[video_path].keys())
            if len(cache_keys) > self.cache_size:
                # Remove oldest frames beyond cache size
                to_remove = cache_keys[:-self.cache_size]
                for idx in to_remove:
                    del self.frame_cache[video_path][idx]
    
    def __len__(self):
        return len(self.frame_data)
    
    def __getitem__(self, idx):
        frame_info = self.frame_data[idx]
        video_path = frame_info['video_path']
        frame_idx = frame_info['frame_idx']
        bbox = frame_info['bbox']
        
        # Check if frame is in cache, if not cache it
        with self.cache_lock:
            frame = self.frame_cache[video_path].get(frame_idx, None)
            
        if frame is None:
            # Cache this frame and a few surrounding frames
            nearby_idxs = [i for i in range(frame_idx-2, frame_idx+3) 
                          if i >= 0 and i < len(self.video_metadata.get(video_path, []))]
            self._cache_frames(video_path, nearby_idxs)
            
            # Check cache again
            with self.cache_lock:
                frame = self.frame_cache[video_path].get(frame_idx, None)
        
        if frame is None:
            # If still not available, create a placeholder
            print(f"Frame {frame_idx} from {video_path} not available")
            placeholder = torch.zeros(3, self.frame_size, self.frame_size)
            return placeholder, video_path, frame_idx
        
        # Process the frame
        h, w, _ = frame.shape
        x1, y1, x2, y2 = expand_bbox(bbox, w, h)
        cropped_frame = frame[y1:y2, x1:x2]
        cropped_frame = cv2.resize(cropped_frame, (self.frame_size, self.frame_size))[:, :, ::-1]  # BGR to RGB
        cropped_frame = cropped_frame / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(cropped_frame).permute(2, 0, 1).float()
        tensor = norm_transform(tensor)
        
        return tensor, video_path, frame_idx

class BatchResultProcessor:
    """
    Processes and accumulates batch results, then saves them efficiently.
    """
    def __init__(self):
        self.results_by_video = {}  # video_path -> {frame_idx -> result}
        self.save_queue = Queue()
        self.save_thread = threading.Thread(target=self._save_worker)
        self.save_thread.daemon = True
        self.save_thread.start()
    
    def add_batch_result(self, pred_data, video_paths, frame_indices):
        """
        Add batch results to the processor.
        
        Args:
            pred_data: Dictionary of prediction results
            video_paths: List of video paths corresponding to batch items
            frame_indices: List of frame indices corresponding to batch items
        """
        # Convert kp_3d to kp_2d
        if "pred_kp3d_crop" in pred_data:
            pred_data["kp_2d"] = [convert_3d_to_2d(kp.cpu().numpy()) for kp in pred_data["pred_kp3d_crop"]]
        
        # Process CPU transfers for latent variables
        for key in ["xf_shape", "xf_pose", "xf_cam"]:
            if key in pred_data:
                pred_data[key] = pred_data[key].cpu()
        
        # Group results by video path
        for i in range(len(video_paths)):
            video_path = video_paths[i]
            frame_idx = frame_indices[i]
            
            if video_path not in self.results_by_video:
                self.results_by_video[video_path] = {}
            
            # Store frame result
            frame_result = {}
            for key, value in pred_data.items():
                if isinstance(value, torch.Tensor) and len(value) == len(video_paths):
                    # Tensor with batch dimension
                    frame_result[key] = value[i:i+1]  # Keep dimension
                elif isinstance(value, list) and len(value) == len(video_paths):
                    # List of values
                    frame_result[key] = [value[i]]
            
            self.results_by_video[video_path][frame_idx] = frame_result
            
            # If we've accumulated enough frames for a video, save it
            if len(self.results_by_video[video_path]) >= 100:  # Save after 100 frames
                self._queue_save(video_path)
    
    def _queue_save(self, video_path):
        """Queue a video for saving in a separate thread"""
        if video_path in self.results_by_video and self.results_by_video[video_path]:
            results = self.results_by_video.pop(video_path)
            self.save_queue.put((video_path, results))
    
    def _save_worker(self):
        """Worker thread that saves results to disk"""
        while True:
            video_path, results = self.save_queue.get()
            
            # Process and combine results
            final_results = self._combine_frame_results(results)
            
            # Save results
            dir_name = os.path.dirname(video_path).replace('data', 'data_tensor')
            os.makedirs(dir_name, exist_ok=True)
            save_path = video_path.replace('data', 'data_tensor').replace('.mp4', '_results.pt')
            torch.save(final_results, save_path)
            
            self.save_queue.task_done()
    
    def _combine_frame_results(self, frame_results):
        """Combine results from multiple frames into a single tensor dictionary"""
        if not frame_results:
            return {}
        
        # Get sorted frame indices
        frame_indices = sorted(frame_results.keys())
        
        # Combine results
        combined = {}
        first_frame = frame_results[frame_indices[0]]
        
        for key in first_frame.keys():
            values = [frame_results[idx][key] for idx in frame_indices]
            
            if isinstance(values[0], list):
                # Flatten lists of tensors
                flat_values = []
                for v in values:
                    flat_values.extend(v)
                combined[key] = flat_values
            else:
                # Concatenate tensors
                combined[key] = torch.cat(values, dim=0)
        
        return combined
    
    def flush_all(self):
        """Save all remaining results"""
        video_paths = list(self.results_by_video.keys())
        for video_path in video_paths:
            self._queue_save(video_path)
        
        # Wait for all saves to complete
        self.save_queue.join()

def batch_inference(model, dataloader):
    """
    Process batches through the model with efficient GPU usage
    """
    model.eval()
    result_processor = BatchResultProcessor()
    error_list = []
    
    with torch.no_grad():
        for batch_frames, batch_video_paths, batch_frame_idxs in tqdm(dataloader, desc="Processing batches"):
            try:
                # Move batch to GPU
                batch_frames = batch_frames.cuda(non_blocking=True)
                print(batch_frames.shape)
                # Run model inference
                data = model.latentpredict(batch_frames)
                pred_data = model.easypredict(
                    xf_shape=data['xf_shape'], 
                    xf_pose=data['xf_pose'], 
                    xf_cam=data['xf_cam'], 
                    xf=None
                )
                
                # Add latent variables from the initial prediction
                pred_data["xf_shape"] = data["xf_shape"]
                pred_data["xf_pose"] = data["xf_pose"]
                pred_data["xf_cam"] = data["xf_cam"]
                
                # Add to result processor
                result_processor.add_batch_result(pred_data, batch_video_paths, batch_frame_idxs)
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                for video_path in batch_video_paths:
                    if video_path not in error_list:
                        error_list.append(video_path)
    
    # Save any remaining results
    result_processor.flush_all()
    return error_list

if __name__ == "__main__":
    # Parse and set arguments
    args = parse_args()
    args = set_default_args(args)

    # Load model
    model_path = "/home/ubuntu/workspace/Dessie/COMBINAREAL/version_8/checkpoints/best.ckpt"
    model = InferenceModel(args).cuda()
    state_dict = torch.load(model_path, map_location="cuda")["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    model.initial_setup()
    
    # Find videos to process
    video_root = '/home/ubuntu/workspace/data'
    video_list = list_mp4_files(video_root)
    
    print(f"Found {len(video_list)} videos to process")
    
    # Create dataset with frame caching
    dataset = VideoFrameDatasetWithFrameCache(
        video_paths=video_list,
        frame_size=256,
        cache_size=50  # Cache 50 frames per video
    )
    
    # Create DataLoader with pinned memory for faster GPU transfers
    dataloader = DataLoader(
        dataset,
        batch_size=args.gpu_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    # Process all videos in batches
    error_list = batch_inference(model, dataloader)
    
    # Add any preprocessing errors
    error_list.extend(dataset.error_videos)
    
    # Save errors
    print(f"Encountered errors with {len(error_list)} videos")
    with open('error_file_path.pkl', 'wb') as f:
        pickle.dump(error_list, f)
    
    print("Video processing completed.")