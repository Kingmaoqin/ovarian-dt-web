# env/render_smoke_test.py
import os, numpy as np, open3d as o3d
from open3d.visualization.rendering import Camera
os.makedirs("outputs", exist_ok=True)

W, H = 800, 600
print("Open3D:", o3d.__version__)

# 造一个椭球点云
th = np.linspace(0, 2*np.pi, 800)
ph = np.linspace(0, np.pi, 400)
TH, PH = np.meshgrid(th, ph)
a,b,c = 1.0, 0.8, 0.6
pts = np.stack([
    a*np.sin(PH)*np.cos(TH),
    b*np.sin(PH)*np.sin(TH),
    c*np.cos(PH)
], axis=-1).reshape(-1,3).astype(np.float32)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

# 离屏渲染（是否无头由后端+环境变量决定）
renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
renderer.scene.set_background([1,1,1,1])

mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.base_color = (0.85, 0.2, 0.25, 1.0)
mat.point_size = 2.0
renderer.scene.add_geometry("pc", pcd, mat)  # 0.19 用 add_geometry

# 相机
bb = pcd.get_axis_aligned_bounding_box()
center = bb.get_center()
extent = max(bb.get_extent())
eye = center + np.array([0, 0, 3.0*extent])
up  = [0, 1, 0]
renderer.scene.camera.set_projection(60.0, W/float(H), 0.1, 100.0, Camera.FovType.Vertical)
renderer.scene.camera.look_at(center, eye, up)

img = renderer.render_to_image()
o3d.io.write_image("outputs/open3d_headless_smoke.png", img)
print("[OK] wrote outputs/open3d_headless_smoke.png")
