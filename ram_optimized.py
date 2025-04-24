import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from src.trainer.vanillaupdatetrainer import InferenceModel
import torch
import json
import cv2
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

###############################################################################
# GLOBAL DICTIONARY: will store {video_path: (json_path or None, frame_count)} 
# so we don't parse all JSON in the Dataset __init__.
###############################################################################
GLOBAL_VIDEO_INFO = {}

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
    focal = 5000
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_proj = (focal * x) / z
    screen_x = (size / 2) * (1 + x_proj / (size / 2))

    R = np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0,  1]])
    T = np.zeros(3)
    points_camera = (points @ R.T) + T
    y_camera = points_camera[:, 1]
    z_camera = points_camera[:, 2]
    y_screen = (focal * y_camera) / z_camera
    y_image = (y_screen / (size / 2)) * (size / 2) + (size / 2)
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
    bw = x2 - x1
    bh = y2 - y1
    ew = int(bw * expand_ratio)
    eh = int(bh * expand_ratio)
    x1_new = max(0, x1 - ew)
    y1_new = max(0, y1 - eh)
    x2_new = min(frame_width, x2 + ew)
    y2_new = min(frame_height, y2 + eh)
    return [x1_new, y1_new, x2_new, y2_new]

##############################################################################
# 1) Multiprocessing to gather frame counts from JSON in parallel
##############################################################################
def _process_json_for_video(path, max_frames_per_video=None):
    """
    Worker function for parallel reading: returns (path, json_path or None, frame_count)
    """
    json_path = path.replace('data', 'data_cropped').replace('.mp4', '_seg.json')
    if not os.path.exists(json_path):
        return (path, None, 0)
    try:
        with open(json_path, 'r') as f:
            frame_data = json.load(f)
        length = len(frame_data)
        if max_frames_per_video is not None and length > max_frames_per_video:
            length = max_frames_per_video
        return (path, json_path, length)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return (path, None, 0)

def gather_video_info_in_parallel(video_list, max_frames_per_video):
    """
    Populates the GLOBAL_VIDEO_INFO dict using multiple threads to parse JSON lengths quickly.
    """
    global GLOBAL_VIDEO_INFO
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Launch parallel tasks
        tasks = [executor.submit(_process_json_for_video, v, max_frames_per_video) for v in video_list]
        for t in tasks:
            results.append(t.result())

    # Store results in the global dictionary
    for (vid_path, json_path, length) in results:
        GLOBAL_VIDEO_INFO[vid_path] = (json_path, length)

##############################################################################
# 2) Updated VideoFrameDatasetWithFrameCache that reads from GLOBAL_VIDEO_INFO
##############################################################################
class VideoFrameDatasetWithFrameCache(Dataset):
    """
    Same as before, but we skip the JSON-length loop in _preprocess_videos
    and instead read from GLOBAL_VIDEO_INFO, which we populated in parallel.
    """
    def __init__(self, video_paths, frame_size=256, max_frames_per_video=None, cache_size=30):
        self.video_paths = video_paths
        self.frame_size = frame_size
        self.max_frames_per_video = max_frames_per_video
        self.cache_size = cache_size

        self.error_videos = []
        self.video_json_paths = []
        self.video_frame_counts = []
        self.prefix_sums = []
        self.frame_cache = {}  # video_path -> {"meta": None, "frames": {}}

        self._preprocess_videos()

    def _preprocess_videos(self):
        global GLOBAL_VIDEO_INFO
        running_sum = 0
        for vid_path in self.video_paths:
            # read from the global dict
            json_path, length = GLOBAL_VIDEO_INFO.get(vid_path, (None, 0))
            self.video_json_paths.append(json_path)
            self.video_frame_counts.append(length)
            running_sum += length
            self.prefix_sums.append(running_sum)
            self.frame_cache[vid_path] = {"meta": None, "frames": {}}

    def __len__(self):
        return self.prefix_sums[-1] if self.prefix_sums else 0

    def _find_video_index(self, global_idx):
        lo, hi = 0, len(self.prefix_sums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if global_idx < self.prefix_sums[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __getitem__(self, idx):
        video_idx = self._find_video_index(idx)
        offset_in_video = idx
        if video_idx > 0:
            offset_in_video -= self.prefix_sums[video_idx - 1]

        video_path = self.video_paths[video_idx]
        json_path = self.video_json_paths[video_idx]

        if not json_path:
            # No bounding boxes => blank
            return torch.zeros(3, self.frame_size, self.frame_size), video_path, offset_in_video

        if self.frame_cache[video_path]["meta"] is None:
            # Load actual JSON bounding boxes
            try:
                with open(json_path, 'r') as f:
                    self.frame_cache[video_path]["meta"] = json.load(f)
            except Exception as e:
                print(f"Error loading JSON {json_path}: {e}")
                return torch.zeros(3, self.frame_size, self.frame_size), video_path, offset_in_video

        vid_meta = self.frame_cache[video_path]["meta"]
        if offset_in_video < 0 or offset_in_video >= len(vid_meta):
            print(f"Offset {offset_in_video} out of range for {video_path}.")
            return torch.zeros(3, self.frame_size, self.frame_size), video_path, offset_in_video

        frame_info = vid_meta[offset_in_video]
        frame_idx = frame_info["frame_idx"]
        bbox = frame_info["bbox"]

        # Attempt to read frame from local cache
        if frame_idx in self.frame_cache[video_path]["frames"]:
            frame = self.frame_cache[video_path]["frames"][frame_idx]
        else:
            # Load from disk
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return torch.zeros(3, self.frame_size, self.frame_size), video_path, frame_idx
            # Put in cache
            self.frame_cache[video_path]["frames"][frame_idx] = frame
            if len(self.frame_cache[video_path]["frames"]) > self.cache_size:
                # Evict the oldest item
                oldest_key = next(iter(self.frame_cache[video_path]["frames"]))
                del self.frame_cache[video_path]["frames"][oldest_key]

        # Crop and transform
        h, w, _ = frame.shape
        x1, y1, x2, y2 = expand_bbox(bbox, w, h)
        cropped = frame[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (self.frame_size, self.frame_size))[:, :, ::-1]
        cropped = cropped / 255.0
        tensor = torch.from_numpy(cropped).permute(2, 0, 1).float()
        tensor = norm_transform(tensor)

        return tensor, video_path, frame_idx

##############################################################################
class BatchResultProcessor:
    def __init__(self):
        self.results_by_video = {}
        self.save_queue = Queue()
        self.save_thread = threading.Thread(target=self._save_worker)
        self.save_thread.daemon = True
        self.save_thread.start()
    
    def add_batch_result(self, pred_data, video_paths, frame_indices):
        if "pred_kp3d_crop" in pred_data:
            pred_data["kp_2d"] = [convert_3d_to_2d(kp.cpu().numpy()) for kp in pred_data["pred_kp3d_crop"]]
        
        for key in ["xf_shape", "xf_pose", "xf_cam"]:
            if key in pred_data:
                pred_data[key] = pred_data[key].cpu()
        
        for i in range(len(video_paths)):
            video_path = video_paths[i]
            frame_idx = frame_indices[i]
            if video_path not in self.results_by_video:
                self.results_by_video[video_path] = {}
            frame_result = {}
            for key, value in pred_data.items():
                if isinstance(value, torch.Tensor) and len(value) == len(video_paths):
                    frame_result[key] = value[i:i+1]
                elif isinstance(value, list) and len(value) == len(video_paths):
                    frame_result[key] = [value[i]]
            self.results_by_video[video_path][frame_idx] = frame_result
            
            if len(self.results_by_video[video_path]) >= 100:
                self._queue_save(video_path)
    
    def _queue_save(self, video_path):
        if video_path in self.results_by_video and self.results_by_video[video_path]:
            results = self.results_by_video.pop(video_path)
            self.save_queue.put((video_path, results))
    
    def _save_worker(self):
        while True:
            video_path, results = self.save_queue.get()
            final_results = self._combine_frame_results(results)
            dir_name = os.path.dirname(video_path).replace('data', 'data_tensor')
            os.makedirs(dir_name, exist_ok=True)
            save_path = video_path.replace('data', 'data_tensor').replace('.mp4', '_results.pt')
            torch.save(final_results, save_path)
            self.save_queue.task_done()
    
    def _combine_frame_results(self, frame_results):
        if not frame_results:
            return {}
        frame_indices = sorted(frame_results.keys())
        combined = {}
        first_frame = frame_results[frame_indices[0]]
        for key in first_frame.keys():
            values = [frame_results[idx][key] for idx in frame_indices]
            if isinstance(values[0], list):
                flat_values = []
                for v in values:
                    flat_values.extend(v)
                combined[key] = flat_values
            else:
                combined[key] = torch.cat(values, dim=0)
        return combined
    
    def flush_all(self):
        video_paths = list(self.results_by_video.keys())
        for video_path in video_paths:
            self._queue_save(video_path)
        self.save_queue.join()

def batch_inference(model, dataloader):
    model.eval()
    result_processor = BatchResultProcessor()
    error_list = []
    with torch.no_grad():
        for batch_frames, batch_video_paths, batch_frame_idxs in tqdm(dataloader, desc="Processing batches"):
            try:
                batch_frames = batch_frames.cuda(non_blocking=True)
                # Inference
                data = model.latentpredict(batch_frames)
                pred_data = model.easypredict(
                    xf_shape=data['xf_shape'], 
                    xf_pose=data['xf_pose'], 
                    xf_cam=data['xf_cam'], 
                    xf=None
                )
                pred_data["xf_shape"] = data["xf_shape"]
                pred_data["xf_pose"] = data["xf_pose"]
                pred_data["xf_cam"] = data["xf_cam"]
                # Store
                result_processor.add_batch_result(pred_data, batch_video_paths, batch_frame_idxs)
            except Exception as e:
                print(f"Error processing batch: {e}")
                for vp in batch_video_paths:
                    if vp not in error_list:
                        error_list.append(vp)

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
    print(f"Found {len(video_list)} videos to process.")
    
    # 1) Parallel gather of JSON lengths, stored in GLOBAL_VIDEO_INFO
    gather_video_info_in_parallel(video_list, max_frames_per_video=None)

    # 2) Create dataset with frame caching, but skipping big init overhead
    dataset = VideoFrameDatasetWithFrameCache(
        video_paths=video_list,
        frame_size=256,
        cache_size=50  # for example
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.gpu_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    # Run batched inference
    error_list = batch_inference(model, dataloader)
    
    # Also include any error videos discovered while building dataset
    error_list.extend(dataset.error_videos)
    
    # Save error info
    print(f"Encountered errors with {len(error_list)} videos.")
    with open('error_file_path.pkl', 'wb') as f:
        pickle.dump(error_list, f)
    
    print("Video processing completed.")
