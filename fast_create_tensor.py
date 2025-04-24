import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from src.trainer.vanillaupdatetrainer import InferenceModel
import torch
import json
import cv2
import os
import numpy as np
import multiprocessing as mp
from torchvision import transforms
from tqdm import tqdm
import pickle
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

    # For DINO
    parser.add_argument("--DINO_model_name", type=str, default="dino_vits8")
    parser.add_argument("--DINO_frozen", action="store_true", default=False,help="frozen DINO")
    parser.add_argument("--DINO_obtain_token", action="store_true", default=False,help="obtain CLS token or use key")

    # For GT
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
                tensor_path = os.path.join(dirpath, filename).replace('data', 'data_tensor').replace('.mp4', '_results.pt')
                if not os.path.exists(tensor_path):
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


def process_batch(frames, model):
    tensor_batch = torch.stack(frames).cuda()

    with torch.no_grad():
        data = model.latentpredict(tensor_batch)
        pred_data = model.easypredict(xf_shape=data['xf_shape'], xf_pose=data['xf_pose'], xf_cam=data['xf_cam'], xf=None)

        pred_data["kp_2d"] = [convert_3d_to_2d(kp.cpu().numpy()) for kp in pred_data["pred_kp3d_crop"]]
        pred_data["xf_shape"] = data["xf_shape"].cpu()
        pred_data["xf_pose"] = data["xf_pose"].cpu()
        pred_data["xf_cam"] = data["xf_cam"].cpu()
        # pred_data['frames_idxs'] = frames_idxs

    return pred_data

def init_worker( model_path, args_dict, error_list):
    """
    Initializer function for each pool worker. Loads the YOLO model once per process.
    """
    global GLOBAL_MODEL
    global GLOBAL_ERROR_LIST
    GLOBAL_ERROR_LIST = error_list
    args = argparse.Namespace(**args_dict) # Recreate args namespace from dict
    args = set_default_args(args)
    GLOBAL_MODEL = InferenceModel(args).cuda() # Move model to GPU inside the process
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))["state_dict"] # Load on CPU first, then move to GPU
    GLOBAL_MODEL.load_state_dict(state_dict)
    GLOBAL_MODEL.eval()
    GLOBAL_MODEL.initial_setup()



# Optimized Function to process a single video (now with full sequential frame read)
def process_video_optimized(video_path, batch_size=16):
    """
    Processes a single video, optimized for subprocess execution and full sequential frame reading.
    Args:
        video_path (str): Path to the video file.
        batch_size (int): Batch size for processing frames.
    """
    global GLOBAL_MODEL # Access the globally initialized model
    global GLOBAL_ERROR_LIST
    # 1. Prepare paths and video capture
    json_path = video_path.replace('data', 'data_cropped').replace('.mp4', '_seg.json')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        GLOBAL_ERROR_LIST.append(video_path)
        print(f"Error: Could not open video file {video_path}")
        return

    try:
        with open(json_path, 'r') as f:
            frame_data = json.load(f)
    except Exception as e:
        GLOBAL_ERROR_LIST.append(json_path)
        print(f"Error: JSON file not found {json_path}")
        cap.release()
        return

    # 2. Read all frames sequentially and store them with indices
    indexed_frames = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video
        indexed_frames[frame_idx] = frame
        frame_idx += 1


    frames, frame_indices = [], []
    processed_results = []

    # 3. Frame Processing Loop (access frames from indexed_frames)
    for entry in frame_data:
        frame_idx, bbox = entry["frame_idx"], entry["bbox"]
        frame = indexed_frames.get(frame_idx) # Efficiently retrieve pre-read frame

        if frame is None: # Handle case where frame wasn't pre-read (e.g., read error or frame_idx from json out of range)
            print(f"Warning: Frame {frame_idx} not found in pre-read frames for {video_path}. Skipping.")
            continue

        h,w,_ = frame.shape
        x1, y1, x2, y2 = expand_bbox(bbox, w,h)
        cropped_frame = frame[y1:y2, x1:x2]
        cropped_frame = cv2.resize(cropped_frame, (256, 256))[:, :, ::-1]  # Convert BGR to RGB
        cropped_frame = cropped_frame / 255.0
        tensor = torch.from_numpy(cropped_frame).permute(2, 0, 1).float()
        tensor = norm_transform(tensor)

        frames.append(tensor)
        frame_indices.append(frame_idx)

        if len(frames) >= batch_size:
            processed_results.append(process_batch(frames, GLOBAL_MODEL))
            frames = []

    if frames:
        processed_results.append(process_batch(frames, GLOBAL_MODEL))

    cap.release()

    # 4. Post-processing and Saving Results (remains mostly the same)
    final_results = {}
    if processed_results:
        first_result_keys = processed_results[0].keys()
        for key in first_result_keys:
            values = [res[key] for res in processed_results]
            if isinstance(values[0], list):
                values = [torch.tensor(np.array(v)) for v in values]
            final_results[key] = torch.cat(values, dim=0)

        dir_name = os.path.dirname(video_path).replace('data', 'data_tensor')
        os.makedirs(dir_name, exist_ok=True)
        save_path = video_path.replace('data', 'data_tensor').replace('.mp4', '_results.pt')
        torch.save(final_results, save_path)

    else:
        GLOBAL_ERROR_LIST.append(video_path) 
        print(f"Warning: No results processed for {video_path}")

def process_video_wrapper(args):
    return process_video_optimized(*args)

# Main execution with multiprocessing
if __name__ == "__main__":
    args = parse_args()
    args = set_default_args(args)

    model_path = "/home/ubuntu/workspace/Dessie/COMBINAREAL/version_8/checkpoints/best.ckpt"
    video_root = '/home/ubuntu/workspace/data'
    video_list = list_mp4_files(video_root)

    num_processes = 2 # Number of subprocesses to use - as requested

    manager = mp.Manager()
    shared_error_list = manager.list() # Create a shared list for errors

    # Prepare arguments for subprocesses (video_path, batch_size)
    process_args = [(video_path, 32) for video_path in video_list]
    

    
    with mp.Pool(processes=num_processes,
                 initializer=init_worker,
                 initargs=(model_path, vars(args), shared_error_list)) as pool:
        # Use starmap for multiple arguments to process_video_optimized
        list(tqdm(pool.imap(process_video_wrapper, process_args), total=len(process_args)))


    print(shared_error_list)
    with open('error_file_path.pkl', 'wb') as f:
        pickle.dump(list(shared_error_list), f) # Convert shared list to regular list for pickling
    


    print("Video processing completed using multiprocessing.")
    
