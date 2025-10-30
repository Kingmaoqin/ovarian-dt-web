# 最小测试案例

这是一个最小的演示案例，用于验证整个数据管线是否正常工作。

## 准备测试数据

由于无法直接提供真实的 DICOM 数据，您需要准备以下数据：

### 1. DICOM 数据

将测试 DICOM 序列放在：

```
raw/tcia/DEMO001/study_1/series_1/*.dcm
```

或者使用 TCIA 下载脚本：

```bash
python scripts/fetch_tcia.py \
    --collection TCGA-OV \
    --patient TCGA-04-1331 \
    --out raw/tcia \
    --max-patients 1
```

### 2. 临床数据（可选）

创建示例人口统计学数据：

**raw/clinical/demographics.csv**

```csv
patient_id,visit_date,age,sex,race,bmi
DEMO001,2024-01-01,55,F,White,24.5
DEMO001,2024-04-01,55,F,White,24.2
```

**raw/clinical/labtests.csv**

```csv
patient_id,visit_date,ca125,wbc,hemoglobin
DEMO001,2024-01-01,125.3,6.5,12.8
DEMO001,2024-04-01,98.2,6.8,13.1
```

## 运行完整管线

### 步骤 1: 生成几何模型（第1次就诊）

```bash
python scripts/build_geometry.py \
    --patient DEMO001 \
    --dicom-dir raw/tcia/DEMO001/study_1/series_1 \
    --visit 1 \
    --date 2024-01-01 \
    --threshold -200 300
```

### 步骤 2: 生成几何模型（第2次就诊）

如果有第二次就诊的数据：

```bash
python scripts/build_geometry.py \
    --patient DEMO001 \
    --dicom-dir raw/tcia/DEMO001/study_2/series_1 \
    --visit 2 \
    --date 2024-04-01 \
    --threshold -200 300
```

### 步骤 3: 数据融合

```bash
python -m dt_pipeline.joiner \
    --timeline ovarian-dt-web/data/timeline.json \
    --demographics raw/clinical/demographics.csv \
    --labtests raw/clinical/labtests.csv \
    --output work/feats
```

### 步骤 4: 训练分类模型（需要多个样本）

```bash
python scripts/run_classify.py \
    --data work/feats/tabular_X.csv \
    --labels work/feats/y.csv \
    --model xgb \
    --out work/models/xgb.pkl
```

### 步骤 5: 训练生存模型（需要多个患者）

```bash
python scripts/run_survival.py \
    --x work/feats/surv_X.csv \
    --duration work/feats/duration.csv \
    --event work/feats/event.csv \
    --out work/models/coxph.pkl
```

### 步骤 6: 同步到前端

```bash
python scripts/sync_to_web.py
```

### 步骤 7: 查看可视化

在浏览器中打开：

```
file:///path/to/ovarian-dt-web/index.html
```

或启动一个简单的 HTTP 服务器：

```bash
cd ovarian-dt-web
python -m http.server 8000
```

然后访问：http://localhost:8000

## 导出快照

生成 3D 模型的截图：

```bash
python scripts/export_snapshot.py \
    --patient DEMO001 \
    --visit 1 \
    --out examples/minicase/snapshot_visit1.png
```

## 预期输出

完整运行后，应该生成以下文件：

```
ovarian-dt-web/data/
├── timeline.json
└── DEMO001/
    ├── visit_1/
    │   ├── tumor.ply
    │   ├── tumor.vtp
    │   ├── tumor.stl
    │   └── mask.npy
    └── visit_2/
        ├── tumor.ply
        ├── tumor.vtp
        ├── tumor.stl
        └── mask.npy

work/
├── feats/
│   ├── tabular_X.csv
│   ├── y.csv
│   ├── surv_X.csv
│   ├── duration.csv
│   └── event.csv
└── models/
    ├── xgb.pkl
    ├── xgb.results.json
    ├── coxph.pkl
    ├── coxph.summary.csv
    └── coxph.results.json
```

## 故障排除

### DICOM 读取失败

- 检查 DICOM 文件是否完整
- 尝试使用 `--spacing-override` 手动指定体素间距

### 分割失败

- 调整阈值范围：`--threshold <lower> <upper>`
- 或使用外部 mask：`--mask path/to/mask.npy`

### 模型训练失败

- 确保有足够的样本（分类模型需要 ≥10 个样本，生存模型需要 ≥5 个患者）
- 检查特征列是否包含缺失值

## 注意事项

- 单个患者的数据不足以训练有效的预测模型，仅用于验证管线功能
- 实际应用中需要至少 30-50 个患者的数据
- 确保 DICOM 文件质量良好，否则分割结果可能不准确
