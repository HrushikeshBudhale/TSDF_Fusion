import open3d as o3d
import numpy as np

class FusionVisualizer:
    def __init__(self, fc, update_every=1):
        self.fc = fc
        self.update_every = update_every # Extract mesh every nth frame
        
        self.remaining_calls = 1
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="TSDF", width=1280, height=720, visible=True)
        self.vis.get_render_option().mesh_show_back_face = True
        # self.vis.get_render_option()
        self.vis.register_animation_callback(self.update_viz)
        self.vis.register_key_callback(ord(" "), self.toggle_fusion)
        self.vis.register_key_callback(ord("N"), self.step)
        
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        self.mesh = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)
        self.camera = None
        
        # set viewpoint
        self.vis.get_view_control().set_zoom(1.5)
        self.vis.get_view_control().set_front([0, 0, -1])
        self.vis.get_view_control().set_up([0, -1, 0])
        
        print("Fusion Visualizer initialized")
        print("Press 'space' to start/pause fusion, 'N' to step, 'Q' to quit")
        
    def get_camera(self, camera_index, scale=0.1):
        pose = np.eye(4)
        pose[:3] = self.fc.poses[camera_index].cpu().numpy() # (3,4), (3,3)
        K =self.fc.K.cpu().numpy() # (3,3)
        
        # get fov
        fov = 2 * np.arctan2(K[0,2], K[0,0])
        W2H_ratio = K[0,0] / K[1,1]
        w = fov*W2H_ratio
        h = w/W2H_ratio
        points = np.array([
            [0,0,0],
            [w,h,1],
            [w,-h,1],
            [-w,-h,1],
            [-w,h,1],
        ]) * scale
        lines = np.array([
            [0,1], [0,2], [0,3], [0,4],
            [1,2], [2,3], [3,4], [4,1],
        ])
        frustum = o3d.geometry.LineSet()
        frustum.points = o3d.utility.Vector3dVector(points)
        frustum.lines = o3d.utility.Vector2iVector(lines)
        frustum.colors = o3d.utility.Vector3dVector(np.zeros((lines.shape[0],3)))
        frustum.transform(pose)
        return frustum
        
    def update_viz(self, vis):
        if self.remaining_calls == 0:
            return
        self.remaining_calls -= 1

        fused = self.fc.fuse_frames(n_frames=1)
        if fused and self.fc.fused_frames % self.update_every == 0:
            # update mesh
            vertices, faces, normals, colors = self.fc.tsdf_vol.extract_mesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(faces)
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
            self.mesh.compute_vertex_normals()
            self.vis.add_geometry(self.mesh, reset_bounding_box=False) 
        
            # update camera    
            self.vis.remove_geometry(self.camera, reset_bounding_box=False)
            self.camera = self.get_camera(self.fc.fused_frames-self.fc.step_size)
            self.vis.add_geometry(self.camera, reset_bounding_box=False)
        
        vis.poll_events()
        vis.update_renderer()
        
    def step(self, vis):
        self.remaining_calls = 1
        self.update_every = 1
    
    def toggle_fusion(self, vis):
        if self.remaining_calls == 0:
            self.remaining_calls = np.inf
            self.update_every = 1
        else:
            self.remaining_calls = 0
            self.update_every = 1
    
    def run(self):
        self.vis.run()
        self.vis.destroy_window()

