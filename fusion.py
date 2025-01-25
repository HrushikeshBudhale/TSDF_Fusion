import torch
import numpy as np
from skimage import measure
from utils import apply_transform
torch.set_printoptions(sci_mode=False)


class TSDFVolume:
    def __init__(self, volume_bounds: torch.Tensor, voxel_size: float, device='cpu'):
        assert volume_bounds.shape == (3, 2) # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.volume_bounds = volume_bounds        
        self.voxel_size = voxel_size
        self.device = device
        self.trunc_margin = 5 * voxel_size
        
        # Adjust volume bounds to fit voxel edges
        self.volume_dim = torch.ceil((self.volume_bounds[:,1]-self.volume_bounds[:,0])/self.voxel_size).long()
        self.volume_bounds[:,1] = self.volume_bounds[:,0] + self.volume_dim*self.voxel_size
        self.origin = self.volume_bounds[:,0].to(device)
        
        # volumes
        self.color_volume = torch.zeros((*self.volume_dim,3), dtype=torch.uint8, device=device)
        self.tsdf_volume = torch.ones(*self.volume_dim, dtype=torch.float32, device=device)
        self.weight_volume = torch.zeros_like(self.tsdf_volume, dtype=torch.float32, device=device)
        
        # voxel coordinates
        xv, yv, zv = torch.meshgrid(torch.arange(self.volume_dim[0]), torch.arange(self.volume_dim[1]), torch.arange(self.volume_dim[2]), indexing='ij')
        self.voxel_coords = torch.stack((xv, yv, zv), axis=3).reshape(-1, 3).to(device) # (N, 3)
        
        # world coordinates
        self.world_coords = (self.voxel_coords * self.voxel_size) + self.origin # (N, 3)
        
        print("TSDF Volume initialized")
        print(f"Voxel volume: {self.volume_dim[0]}x{self.volume_dim[1]}x{self.volume_dim[2]}")
    
    def fuse_frame(self, K: torch.Tensor, c2w, rgb, depth, weight=1.0):
        K, c2w, rgb, depth = [x.to(self.device) for x in [K, c2w, rgb, depth]]
        
        # convert world coordinates to uv coordinates
        c2w = torch.cat((c2w, torch.tensor([[0,0,0,1]], dtype=torch.float32, device=c2w.device)), axis=0)
        w2c = torch.inverse(c2w)[:3]
        voxel_camera = apply_transform(w2c, self.world_coords)
        z = voxel_camera[:,2].repeat(3,1).T
        voxel_uv = torch.round((voxel_camera @ K.T) / z).long() # (N, 3)
        px, py = voxel_uv[:,:2].T
        pz = voxel_camera[:,2]
        
        # Eliminate pixels out of view or behind camera
        view_mask = (px >= 0) & (px < depth.shape[1]) & (py >= 0) & (py < depth.shape[0]) & (pz > 0) # (N,)
        valid_px, valid_py = px[view_mask], py[view_mask]
        valid_vx, valid_vy, valid_vz = self.voxel_coords[view_mask, 0], self.voxel_coords[view_mask, 1], self.voxel_coords[view_mask, 2]
        
        # compute TSDF values
        depth_val = depth[valid_py, valid_px]
        sdf_value = depth_val - pz[view_mask]
        tsdf_val = torch.clamp(sdf_value / self.trunc_margin, max=1)
        valid_pts = (-self.trunc_margin < sdf_value) & (sdf_value < self.trunc_margin) & (depth_val > 0) # (N,)
        tsdf_val = tsdf_val[valid_pts]
        
        # Get coordinates of valid voxels
        valid_vx = valid_vx[valid_pts] # (N,)
        valid_vy = valid_vy[valid_pts]
        valid_vz = valid_vz[valid_pts]
        valid_px = valid_px[valid_pts]
        valid_py = valid_py[valid_pts]
        
        # Integrate TSDF volume
        weight_old = self.weight_volume[valid_vx, valid_vy, valid_vz]
        tsdf_old = self.tsdf_volume[valid_vx, valid_vy, valid_vz]
        weight_new = weight_old + weight
        tsdf_new = (tsdf_val * weight + tsdf_old * weight_old) / weight_new
        self.weight_volume[valid_vx, valid_vy, valid_vz] = weight_new
        self.tsdf_volume[valid_vx, valid_vy, valid_vz] = tsdf_new
        
        # Integrate color volume
        weight_old = weight_old.reshape(-1, 1).tile((1,3))
        color_old = self.color_volume[valid_vx, valid_vy, valid_vz].to(torch.float32) # (N, 3)
        weight_new = weight_old + weight
        color_value = rgb[valid_py, valid_px]
        color_new = torch.clamp((color_value * weight + color_old * weight_old) / weight_new, max=255).to(torch.uint8)
        self.color_volume[valid_vx, valid_vy, valid_vz] = color_new
    
    def extract_mesh(self):
        # Extract triangle mesh using Marching Cubes
        vertices, faces, normals, values = measure.marching_cubes(self.tsdf_volume.cpu().numpy())
        vertex_coords = vertices.astype(np.int32)
        vertices = vertices * self.voxel_size + self.origin.cpu().numpy() # (N, 3)
        color_volume = self.color_volume.cpu().numpy()
        colors = color_volume[vertex_coords[:,0], vertex_coords[:,1], vertex_coords[:,2]]
        return vertices, faces, normals, colors