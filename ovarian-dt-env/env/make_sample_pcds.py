# make_sample_pcds.py  — NumPy 2.x 兼容版
import os, json, numpy as np, open3d as o3d

out_dir = "data/pointclouds"
os.makedirs(out_dir, exist_ok=True)

def make_ellipsoid(a=1.0, b=0.8, c=0.6, n_theta=900, n_phi=450, jitter=0.002):
    th = np.linspace(0, 2*np.pi, n_theta, dtype=np.float32)
    ph = np.linspace(0, np.pi,     n_phi, dtype=np.float32)
    TH, PH = np.meshgrid(th, ph, indexing="xy")
    x = a*np.sin(PH)*np.cos(TH)
    y = b*np.sin(PH)*np.sin(TH)
    z = c*np.cos(PH)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
    if jitter and jitter > 0:
        pts += np.random.normal(scale=jitter, size=pts.shape).astype(np.float32)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # 颜色：用 z 轴做简单归一化（NumPy 2.x 兼容）
    zvals = pts[:, 2]
    zmin, zmax = float(zvals.min()), float(zvals.max())
    denom = (zmax - zmin) if (zmax > zmin) else 1.0
    norm_z = (zvals - zmin) / (denom + 1e-8)
    colors = np.stack([0.5 + 0.5*norm_z, 0.1 + 0.2*norm_z, 0.2 + 0.2*norm_z], axis=1).astype(np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def approx_volume(pcd: o3d.geometry.PointCloud) -> float:
    # 演示用：AABB 体积（真实项目请用分割体素体积）
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    return float(extent[0] * extent[1] * extent[2])

# visit1：基线
pcd1 = make_ellipsoid(a=1.0,  b=0.85, c=0.65)
f1 = os.path.join(out_dir, "patientA_visit1.ply")
o3d.io.write_point_cloud(f1, pcd1)

# visit2：稍大模拟进展
pcd2 = make_ellipsoid(a=1.12, b=0.95, c=0.72)
f2 = os.path.join(out_dir, "patientA_visit2.ply")
o3d.io.write_point_cloud(f2, pcd2)

timeline = {
    "patient_id": "patientA",
    "visits": [
        {"visit": 1, "days_from_diagnosis": 0,   "ply": "patientA_visit1.ply", "approx_vol": approx_volume(pcd1)},
        {"visit": 2, "days_from_diagnosis": 90,  "ply": "patientA_visit2.ply", "approx_vol": approx_volume(pcd2)},
    ]
}
with open(os.path.join(out_dir, "timeline.json"), "w") as f:
    json.dump(timeline, f, indent=2)

print("Wrote:")
print(" -", f1)
print(" -", f2)
print(" -", os.path.join(out_dir, "timeline.json"))
