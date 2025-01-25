import torch 

def apply_transform(pose: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    # pose: (3, 4), points: (N, 3)
    points_homogenous = torch.cat((points, torch.ones((points.shape[0], 1), dtype=torch.float32, device=points.device)), axis=1)
    return torch.matmul(pose, points_homogenous.T).T

def get_camera_frustum(depth_image: torch.Tensor, K: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    # depth_image: (H, W), pose: (3, 4), K: (3, 3)
    H, W = depth_image.shape
    max_d = depth_image.max().item()
    frustum_points = torch.tensor([
        [0,0,1],
        [W,0,1],
        [W,H,1],
        [0,H,1],
        [0,0,0],
    ], dtype=torch.float32, device=depth_image.device)
    frustum_points = (torch.inverse(K) @ frustum_points.T).T
    frustum_points *= max_d
    frustum_points = apply_transform(pose, frustum_points)
    return frustum_points

def save_mesh(filename: str, vertices, faces, colors) -> None:
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex {}\n'.format(vertices.shape[0]))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('element face {}\n'.format(faces.shape[0]))
            f.write('property list uchar int vertex_index\n')
            f.write('end_header\n')
            
            # write vertices
            for i in range(vertices.shape[0]):
                f.write(f'{vertices[i,0]} {vertices[i,1]} {vertices[i,2]} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n')
            
            # write faces
            for i in range(faces.shape[0]):
                f.write(f'3 {faces[i,0]} {faces[i,1]} {faces[i,2]}\n')
                
        print(f"Mesh saved to {filename}")