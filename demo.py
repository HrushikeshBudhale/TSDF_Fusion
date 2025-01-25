import torch
from path import Path
from tqdm import tqdm
from utils import get_camera_frustum, save_mesh, apply_transform
from visualizer import FusionVisualizer
import cv2
import numpy as np
import fusion
import argparse

class FusionController:
    def __init__(self, data_dir: Path, step_size=1, device="cpu"):
        self.data_dir = data_dir
        self.step_size = step_size
        self.device = device
        
        print("Loading Data...")
        transforms = np.loadtxt(data_dir/"transforms.csv", skiprows=1, delimiter=",", dtype=np.float32)
        self.image_numbers = transforms[:,0].astype(np.int32)
        self.TOTAL_FRAMES = self.image_numbers.shape[0]
        poses = transforms[:,1:-4].reshape(-1,3,4)
        self.poses = torch.from_numpy(poses)
        
        fx, fy, cx, cy = transforms[0,-4:] # All cameras have same K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.K = torch.from_numpy(K)
        self.tqdm = tqdm(range(self.TOTAL_FRAMES), desc="Fusing Frames")
        self.fused_frames = 0
        
        print("Setting volume bounds...")
        volume_bounds = torch.zeros((3,2))
        for i in tqdm(self.image_numbers, desc="Finding volume bounds"):
            image_stem = f"{int(i):05d}"
            depth = cv2.imread(data_dir/"depth/"+(image_stem+".png"), -1).astype(np.float32)
            depth = torch.from_numpy(depth)
            depth /= 1000.0 # depth is in mm
            depth[depth == 65.535] = 0  # set invalid depth to 0 (specific to dataset)
            points_3d = get_camera_frustum(depth, self.K, self.poses[i])
            volume_bounds[:,0] = torch.minimum(volume_bounds[:,0], torch.amin(points_3d, axis=0))
            volume_bounds[:,1] = torch.maximum(volume_bounds[:,1], torch.amax(points_3d, axis=0))

        print("Creating voxel volume...")
        self.tsdf_vol = fusion.TSDFVolume(volume_bounds, voxel_size=0.025, device=device)
  
    def fuse_frames(self, n_frames=-1):
        if n_frames == -1:
            n_frames = self.TOTAL_FRAMES
            
        if self.fused_frames >= self.TOTAL_FRAMES:
            return False

        while n_frames > 0:
            i = self.fused_frames
            image_number = self.image_numbers[i]
            image_stem = f"{image_number:05d}"
            bgr = cv2.imread(self.data_dir/"rgb/"+(image_stem+".jpg"), -1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = torch.from_numpy(rgb)
            depth = cv2.imread(self.data_dir/"depth/"+(image_stem+".png"), -1).astype(np.float32)
            depth = torch.from_numpy(depth)
            depth /= 1000.0 # depth is in mm
            depth[depth == 65.535] = 0 # set invalid depth to 0 (specific to dataset)
            self.tsdf_vol.fuse_frame(self.K, self.poses[i], rgb, depth, weight=1.0)
            
            n_frames -= self.step_size
            self.fused_frames += self.step_size
            self.tqdm.update(self.step_size)
        return True

def main(args):
    data_dir = Path(args.data_dir)
    visualize = args.visualize == "True"
    output_path = Path(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Initialize fusion controller
    fc = FusionController(data_dir, step_size=2, device=device)
    if visualize:
        viz = FusionVisualizer(fc, update_every=1)
        viz.run()
    else:
        fc.fuse_frames()      
    
    # Save mesh
    print("Saving mesh...")
    v, f, n, c = fc.tsdf_vol.extract_mesh()
    save_mesh(output_path/"output_mesh.ply", v, f, c)
    print("Done")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--visualize", required=False, default="True", type=str, choices=["True", "False"])
    parser.add_argument("--output_dir", required=False, default="./output", type=str)
    
    args = parser.parse_args()
    main(args)


