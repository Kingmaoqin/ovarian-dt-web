# 卵巢癌数字孪生系统 (Ovarian Digital Twin)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

完整的卵巢癌数字孪生数据管线与基线建模系统，从医学影像到 3D 可视化与预测分析。

## 项目概述

本项目提供从医学影像数据（DICOM）到 3D 几何模型、多模态数据融合、预测建模与 Web 可视化的完整管线：

```
DICOM 影像 → 分割 → 网格/点云 → 数据融合 → 预测模型 → Web 可视化
```

### 主要功能

- **数据收集**: TCIA 影像下载、临床数据标准化
- **几何重建**: DICOM → 3D 网格/点云（Marching Cubes + Poisson）
- **数据融合**: 几何度量 + 人口统计 + 化验数据
- **预测建模**:
  - 横断面分类（XGBoost/RandomForest）
  - 生存分析（Cox 比例风险模型）
- **可视化**: Web 前端时间轴展示（支持 VTK.js 集成）

## 目录结构

```
ovarian-dt-web/
├── dt_pipeline/              # 核心管线模块
│   ├── __init__.py
│   ├── tcia_client.py        # TCIA 数据下载
│   ├── clinical_io.py        # 临床数据读取
│   ├── dicom_reader.py       # DICOM 读取与重建
│   ├── segmentation.py       # 分割（阈值/区域生长/mask）
│   ├── mesh_generator.py     # 网格生成（Marching Cubes/Poisson）
│   ├── pointcloud_generator.py  # 点云生成与下采样
│   └── joiner.py             # 数据融合
├── scripts/                  # 命令行脚本
│   ├── fetch_tcia.py         # 下载 TCIA 数据
│   ├── build_geometry.py     # 生成几何模型
│   ├── run_classify.py       # 横断面分类
│   ├── run_survival.py       # 生存分析
│   ├── sync_to_web.py        # 同步到前端
│   └── export_snapshot.py    # 导出 3D 快照
├── ovarian-dt-web/           # Web 前端
│   ├── index.html            # 主页面
│   ├── assets/               # 静态资源
│   └── data/                 # 数据目录
│       └── timeline.json     # 时间轴元数据
├── conf/                     # 配置文件
│   └── config.yaml
├── raw/                      # 原始数据
│   ├── tcia/                 # DICOM 数据
│   └── clinical/             # 临床数据（demographics.csv, labtests.csv）
├── work/                     # 工作目录
│   ├── feats/                # 特征矩阵
│   └── models/               # 训练好的模型
├── examples/                 # 示例
│   └── minicase/             # 最小测试案例
├── Makefile                  # 便捷命令
└── README.md                 # 本文档
```

## 环境要求

- **Python**: 3.10+
- **Conda 环境**: ov-dt

### 依赖包

```bash
# 创建 Conda 环境
conda create -n ov-dt python=3.10
conda activate ov-dt

# 安装依赖
pip install open3d numpy pandas scikit-image scikit-learn \
    xgboost lifelines pydicom tqdm requests pyyaml
```

或使用 requirements.txt（如已创建）：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 初始化项目

```bash
make setup
```

### 2. 下载 TCIA 数据（示例）

```bash
python scripts/fetch_tcia.py \
    --collection TCGA-OV \
    --patient TCGA-04-1331 \
    --out raw/tcia
```

或列出可用数据集：

```bash
python scripts/fetch_tcia.py --list-collections
```

### 3. 准备临床数据

将人口统计学数据和化验数据放在 `raw/clinical/`：

- `demographics.csv`: patient_id, visit_date, age, sex, race, bmi, ...
- `labtests.csv`: patient_id, visit_date, a1c, egfr, wbc, ca125, ...

参考 `dt_pipeline/clinical_io.py` 查看支持的列名。

### 4. 生成几何模型

为每次就诊生成 3D 网格和点云：

```bash
python scripts/build_geometry.py \
    --patient PAT123 \
    --dicom-dir raw/tcia/PAT123/study_1/series_1 \
    --visit 1 \
    --date 2024-05-01 \
    --threshold -200 300
```

**参数说明**:
- `--threshold LOWER UPPER`: 阈值分割范围（HU 值）
- `--mask PATH`: 使用外部 mask（.npy 或 .nii.gz）
- `--poisson`: 启用 Poisson 表面重建（更平滑）
- `--voxel SIZE`: 点云下采样体素大小（默认 1.5mm）

**输出**:
- `ovarian-dt-web/data/PAT123/visit_1/tumor.{ply,vtp,stl}`
- `ovarian-dt-web/data/timeline.json`（更新）

### 5. 数据融合

合并几何度量、人口统计和化验数据：

```bash
python -m dt_pipeline.joiner \
    --timeline ovarian-dt-web/data/timeline.json \
    --demographics raw/clinical/demographics.csv \
    --labtests raw/clinical/labtests.csv \
    --output work/feats
```

**输出**:
- `work/feats/tabular_X.csv`, `y.csv`（分类任务）
- `work/feats/surv_X.csv`, `duration.csv`, `event.csv`（生存分析）

### 6. 训练预测模型

#### 横断面分类（预测下一次体积是否增大）

```bash
python scripts/run_classify.py \
    --data work/feats/tabular_X.csv \
    --labels work/feats/y.csv \
    --model xgb \
    --out work/models/xgb.pkl
```

**评估指标**: AUC-ROC, Accuracy, PR-AUC（5-Fold CV）

#### 生存分析（时间到进展/复发）

```bash
python scripts/run_survival.py \
    --x work/feats/surv_X.csv \
    --duration work/feats/duration.csv \
    --event work/feats/event.csv \
    --out work/models/coxph.pkl
```

**评估指标**: C-index, Log-rank test

### 7. 可视化

同步数据到前端：

```bash
python scripts/sync_to_web.py
```

在浏览器中打开 `ovarian-dt-web/index.html`，或启动 HTTP 服务器：

```bash
cd ovarian-dt-web
python -m http.server 8000
# 访问 http://localhost:8000
```

### 8. 导出快照（可选）

生成 3D 模型的标准角度截图：

```bash
python scripts/export_snapshot.py \
    --patient PAT123 \
    --visit 1 \
    --out snapshot.png
```

## 完整管线示例

一条龙命令（假设数据已准备好）：

```bash
# 1. 为多个就诊生成几何模型
for visit in 1 2 3; do
    python scripts/build_geometry.py \
        --patient PAT123 \
        --dicom-dir raw/tcia/PAT123/study_${visit}/series_1 \
        --visit ${visit} \
        --date 2024-0${visit}-01 \
        --threshold -200 300
done

# 2. 数据融合
python -m dt_pipeline.joiner \
    --timeline ovarian-dt-web/data/timeline.json \
    --demographics raw/clinical/demographics.csv \
    --labtests raw/clinical/labtests.csv \
    --output work/feats

# 3. 训练模型
python scripts/run_classify.py \
    --data work/feats/tabular_X.csv \
    --labels work/feats/y.csv \
    --model xgb --out work/models/xgb.pkl

python scripts/run_survival.py \
    --x work/feats/surv_X.csv \
    --duration work/feats/duration.csv \
    --event work/feats/event.csv \
    --out work/models/coxph.pkl

# 4. 同步到前端
python scripts/sync_to_web.py
```

## 配置

编辑 `conf/config.yaml` 以调整默认参数：

- TCIA API Key
- 分割阈值
- 网格平滑参数
- 点云体素大小
- 模型超参数

## 测试

运行最小测试案例：

```bash
make demo
```

详见 `examples/minicase/README.md`。

检查依赖：

```bash
make check-deps
```

## 脚本参考

所有脚本支持 `--help` 查看详细用法：

```bash
python scripts/build_geometry.py --help
python scripts/run_classify.py --help
python scripts/run_survival.py --help
```

### 主要脚本

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `fetch_tcia.py` | 下载 TCIA 数据 | collection, patient | DICOM 文件 |
| `build_geometry.py` | 生成几何模型 | DICOM, mask | .ply, .vtp, .stl, timeline.json |
| `joiner.py` | 数据融合 | timeline.json, 临床数据 | tabular_X.csv, y.csv, surv_X.csv, ... |
| `run_classify.py` | 分类模型 | X, y | xgb.pkl, results.json |
| `run_survival.py` | 生存模型 | X, duration, event | coxph.pkl, summary.csv |
| `sync_to_web.py` | 同步到前端 | timeline.json, 几何文件 | ovarian-dt-web/data/ |
| `export_snapshot.py` | 导出快照 | patient, visit | .png |

## 数据格式

### timeline.json

```json
[
  {
    "patient_id": "PAT123",
    "visits": [
      {
        "visit": 1,
        "date": "2024-05-01",
        "ply": "PAT123/visit_1/tumor.ply",
        "vtp": "PAT123/visit_1/tumor.vtp",
        "vol_cc": 123.45,
        "surface_area_cm2": 234.56,
        "num_points": 15000
      },
      {
        "visit": 2,
        "date": "2024-08-01",
        "ply": "PAT123/visit_2/tumor.ply",
        "vtp": "PAT123/visit_2/tumor.vtp",
        "vol_cc": 145.12,
        "surface_area_cm2": 256.78,
        "num_points": 16000
      }
    ]
  }
]
```

### demographics.csv

| patient_id | visit_date | age | sex | race | bmi |
|------------|------------|-----|-----|------|-----|
| PAT123 | 2024-05-01 | 55 | F | White | 24.5 |

### labtests.csv

| patient_id | visit_date | a1c | egfr | wbc | ca125 |
|------------|------------|-----|------|-----|-------|
| PAT123 | 2024-05-01 | 5.7 | 90 | 6.5 | 125.3 |

## 技术细节

### 分割方法

1. **阈值分割**: 简单快速，适合 CT 数据（基于 HU 值）
2. **区域生长**: 从种子点扩展，适合局部清晰的区域
3. **外部 Mask**: 使用医生标注或 3D Slicer 导出的 mask

### 网格生成

- **Marching Cubes**: skimage.measure.marching_cubes
- **Poisson 重建**: Open3D（可选，更平滑但可能丢失细节）

### 点云处理

- **采样**: 从网格均匀采样或从体数据直接采样
- **下采样**: Voxel 下采样（Open3D）
- **法向量**: KD-Tree 邻域估计

### 基线模型

- **分类**: XGBoost（主）/ RandomForest（备选）
- **生存**: Cox 比例风险模型（lifelines）

## 注意事项

1. **数据量**: 建模需要足够样本（≥30 患者），单个患者仅用于管线验证
2. **分割质量**: 阈值需根据具体数据调整，或使用外部 mask
3. **计算资源**: Poisson 重建（depth ≥ 9）需要较多内存
4. **数据隐私**: 处理真实医疗数据需遵守 HIPAA/GDPR 等法规

## 扩展与改进

### 集成深度学习分割

替换 `segmentation.py` 中的方法：

```python
# 例如使用 nnU-Net
from nnunet.inference import predict
mask = predict(volume, model_path)
```

### 3D 可视化增强

在 `index.html` 中集成 VTK.js 或 Three.js：

```html
<script src="https://unpkg.com/vtk.js"></script>
<!-- 添加 VTK 渲染器代码 -->
```

### 多模态融合

扩展 `joiner.py` 支持基因组、病理图像等数据。

## 常见问题

**Q: DICOM 读取失败？**

A: 检查文件完整性，或使用 `--spacing-override` 手动指定间距。

**Q: 分割结果为空？**

A: 调整 `--threshold` 范围，或检查 DICOM 是否为目标模态（CT/MRI）。

**Q: 模型训练报错？**

A: 确保样本数足够，检查特征列是否有过多缺失值。

**Q: 前端无法加载 3D 模型？**

A: 当前版本为占位实现，需集成 VTK.js 或 Three.js 渲染器。

## 贡献

欢迎贡献代码、报告 bug 或提出功能请求！

## 许可证

MIT License

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{ovarian_dt_2024,
  title = {Ovarian Cancer Digital Twin Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ovarian-dt-web}
}
```

## 联系方式

- 项目维护者: [Your Name]
- Email: your.email@example.com
- Issues: https://github.com/yourusername/ovarian-dt-web/issues

---

**© 2024 Ovarian Digital Twin Project**
