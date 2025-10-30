#!/usr/bin/env bash
set -euo pipefail

# 0) 载入 conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[WARN] Could not find conda.sh. Ensure conda is installed and on PATH."
fi

# 1) 创建 conda 环境
echo "[INFO] Creating conda env ov-dt ..."
conda env remove -n ov-dt -y >/dev/null 2>&1 || true
conda env create -f env/environment.yml

echo "[INFO] Activating env ..."
conda activate ov-dt

# 2) （可选）安装系统依赖 —— 若你有 sudo
#    A) Open3D Headless(推荐优先) 需要 OSMesa (libosmesa)
#    B) PyVista 备选方案使用 Xvfb 离屏
if command -v sudo >/dev/null 2>&1; then
  echo "[INFO] sudo is available. Installing system packages for headless rendering ..."
  # Ubuntu/Debian 系
  if grep -qiE 'ubuntu|debian' /etc/os-release; then
    sudo apt-get update
    # OSMesa for Open3D CPU headless (per Open3D docs)
    sudo apt-get install -y libosmesa6-dev
    # Xvfb for PyVista offscreen (per PyVista docs)
    sudo apt-get install -y xvfb libgl1-mesa-glx
  fi
else
  echo "[INFO] sudo not available. Skipping system packages."
  echo "[INFO] Will try Open3D CPU headless via conda OSMesa packages."
fi

# 3) 打印关键信息
echo "[INFO] Python:"
python --version
echo "[INFO] Open3D version:"
python -c "import open3d as o3d; print(o3d.__version__)"
echo "[INFO] PyVista/VTK versions:"
python - <<'PY'
import pyvista as pv, vtk
print("pyvista", pv.__version__)
print("vtk", vtk.vtkVersion.GetVTKVersion())
PY

echo "[INFO] Environment setup done."
echo "[INFO] Next: run smoke tests."
