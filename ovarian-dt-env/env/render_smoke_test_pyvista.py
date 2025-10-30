# env/render_smoke_test_pyvista.py
import os
os.makedirs("outputs", exist_ok=True)

import numpy as np
import pyvista as pv

# 若你在交互式会话，还可启用:
try:
    pv.start_xvfb()  # per docs, Linux only
except Exception as e:
    print("[WARN] start_xvfb failed:", e)

# 造一个球面点云
theta = np.linspace(0, 2*np.pi, 1200)
phi = np.linspace(0, np.pi, 600)
theta, phi = np.meshgrid(theta, phi)
r = 1.0
x = r*np.sin(phi)*np.cos(theta)
y = r*np.sin(phi)*np.sin(theta)
z = r*np.cos(phi)
pts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

# 转为 PolyData（glyph 更美观）
cloud = pv.PolyData(pts)

plotter = pv.Plotter(off_screen=True)  # per API
plotter.set_background("white")
plotter.add_mesh(cloud, point_size=3.0, render_points_as_spheres=True, color="crimson")
plotter.camera_position = "xy"

png_path = "outputs/pyvista_xvfb_smoke.png"
plotter.show(screenshot=png_path)
print("[OK] Wrote", png_path)
