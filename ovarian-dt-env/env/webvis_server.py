# webvis_server.py — Open3D 0.19 WebRTC server-side render (int point_size)
import os
import open3d as o3d
from open3d.visualization import rendering as r

# ——(可选) 绑定 IP/端口：不设置则走默认 localhost:8888——
os.environ.setdefault("WEBRTC_IP", "0.0.0.0")   # 仅内网/安全环境使用；公网请配 HTTPS 反代
os.environ.setdefault("WEBRTC_PORT", "8888")

# ——启用 WebRTC 可视化后端——
o3d.visualization.webrtc_server.enable_webrtc()

# ——读取示例点云（换成你的实际路径也行）——
pcd1 = o3d.io.read_point_cloud("data/pointclouds/patientA_visit1.ply")
pcd2 = o3d.io.read_point_cloud("data/pointclouds/patientA_visit2.ply")

# ——为不同几何设置材质（point_size 必须是 int）——
mat1 = r.MaterialRecord(); mat1.shader = "defaultUnlit"; mat1.point_size = 2
mat2 = r.MaterialRecord(); mat2.shader = "defaultUnlit"; mat2.point_size = 4

# ——打开 Web 可视化（浏览器访问 http://<IP>:<PORT>）——
o3d.visualization.draw(
    [
        {"name": "visit1", "geometry": pcd1, "material": mat1, "is_visible": True},
        {"name": "visit2", "geometry": pcd2, "material": mat2, "is_visible": True},
    ],
    title="DT Web Viewer",
    width=1280, height=800,
    show_skybox=False,
)
