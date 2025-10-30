# 使用本地 DICOM 文件

由于 TCIA API 访问可能需要特殊权限或认证，您可以使用本地的 DICOM 文件来运行管线。

## 方法 1: 手动下载 TCIA 数据

### 步骤 1: 访问 TCIA 网站

访问 [TCIA 浏览器](https://nbia.cancerimagingarchive.net/nbia-search/)

### 步骤 2: 搜索数据集

1. 选择 "Collections" 选项卡
2. 找到 "TCGA-OV" (卵巢癌数据集)
3. 点击进入

### 步骤 3: 下载 DICOM 文件

1. 选择患者（例如 TCGA-04-1331）
2. 选择研究（Study）
3. 点击 "Cart" 添加到购物车
4. 下载 NBIA Data Retriever（TCIA 官方下载工具）
5. 使用 Data Retriever 下载数据

### 步骤 4: 组织文件

下载后，将 DICOM 文件组织到以下结构：

```
raw/tcia/
└── TCGA-04-1331/              # 患者 ID
    └── study_1/               # 研究编号
        └── series_1/          # 序列编号
            ├── 1.dcm
            ├── 2.dcm
            └── ...
```

## 方法 2: 使用示例 DICOM 数据

如果您没有真实的卵巢癌数据，可以使用其他公开的 DICOM 数据进行测试：

### 选项 A: 使用其他 TCIA 数据集

一些不需要认证的数据集：
- **LIDC-IDRI**: 肺癌数据（完全公开）
- **Breast-MRI-NACT-Pilot**: 乳腺癌 MRI
- **Pancreas-CT**: 胰腺 CT

### 选项 B: 使用在线 DICOM 样本

1. [DICOM Library](https://www.dicomlibrary.com/) - 免费 DICOM 样本
2. [Medical Connections](https://www.medicalconnections.co.uk/FreeStuff) - 测试数据

### 选项 C: 生成模拟数据（仅用于测试管线）

```python
# 创建一个简单的模拟 DICOM 用于测试
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from datetime import datetime

def create_dummy_dicom(output_path, slice_idx):
    """创建模拟 DICOM 文件"""
    # 创建一个简单的 256x256 图像
    pixel_array = np.random.randint(0, 255, (256, 256), dtype=np.uint16)

    # 创建 DICOM dataset
    file_meta = Dataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # 添加必需的 DICOM 标签
    ds.PatientID = "TEST001"
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.SeriesNumber = 1
    ds.InstanceNumber = slice_idx
    ds.ImagePositionPatient = [0, 0, slice_idx * 1.0]  # Z 间距 1mm
    ds.SliceLocation = slice_idx * 1.0
    ds.PixelSpacing = [1.0, 1.0]  # 1mm x 1mm
    ds.SliceThickness = 1.0
    ds.Rows = 256
    ds.Columns = 256
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(output_path)

# 创建一个测试序列（50 个切片）
from pathlib import Path
output_dir = Path("raw/tcia/TEST001/study_1/series_1")
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(1, 51):
    create_dummy_dicom(output_dir / f"{i}.dcm", i)

print(f"已创建 50 个模拟 DICOM 文件: {output_dir}")
```

## 方法 3: 使用您自己的医学影像数据

如果您有访问医院 PACS 系统的权限或自己的医学影像数据：

### 要求：
1. **DICOM 格式**: 文件必须是 DICOM (.dcm) 格式
2. **CT 或 MRI**: 最好是 CT（支持 HU 值）或 MRI
3. **多切片**: 至少 20-50 个切片以形成 3D 体
4. **元数据完整**: 包含 PixelSpacing, SliceThickness, ImagePositionPatient

### 导出步骤：
1. 从 PACS 导出 DICOM 文件
2. 使用 3D Slicer 或其他工具检查数据完整性
3. 组织到正确的目录结构

## 使用本地数据运行管线

一旦您有了 DICOM 文件（任何来源），就可以直接运行几何重建：

```bash
# 使用本地 DICOM 文件
python scripts/build_geometry.py \
    --patient TEST001 \
    --dicom-dir raw/tcia/TEST001/study_1/series_1 \
    --visit 1 \
    --date 2024-01-01 \
    --threshold -200 300
```

**注意**:
- `--threshold` 参数需要根据您的数据类型调整
- CT 数据通常使用 HU 值范围（-200 到 300 适合软组织）
- MRI 数据可能需要不同的阈值

## 获取分割 Mask（推荐）

为了获得更准确的结果，建议使用专业工具进行分割：

### 使用 3D Slicer

1. 下载并安装 [3D Slicer](https://www.slicer.org/)
2. 加载 DICOM 数据
3. 使用 "Segment Editor" 手动或半自动分割肿瘤
4. 导出为 NIfTI (.nii.gz) 或 numpy (.npy) 格式
5. 使用 `--mask` 参数：

```bash
python scripts/build_geometry.py \
    --patient TEST001 \
    --dicom-dir raw/tcia/TEST001/study_1/series_1 \
    --mask path/to/tumor_mask.nii.gz \
    --visit 1 \
    --date 2024-01-01
```

### 使用 ITK-SNAP

另一个流行的医学影像分割工具：
- 下载: http://www.itksnap.org/
- 加载 DICOM，手动分割
- 导出 mask

## 故障排除

### DICOM 读取失败

```bash
# 测试 DICOM 读取
python -m dt_pipeline.dicom_reader --dicom-dir raw/tcia/TEST001/study_1/series_1
```

### 分割结果为空

```bash
# 检查强度范围
python -c "
import numpy as np
from dt_pipeline.dicom_reader import DICOMReader
reader = DICOMReader()
reader.load_series('raw/tcia/TEST001/study_1/series_1')
vol = reader.get_volume()
print(f'强度范围: [{vol.min()}, {vol.max()}]')
"

# 根据输出调整 --threshold 参数
```

## 多个患者示例

如果您有多个患者的数据：

```bash
# 患者 1
python scripts/build_geometry.py --patient PAT001 --dicom-dir raw/tcia/PAT001/study_1/series_1 --visit 1 --date 2024-01-01 --threshold -200 300
python scripts/build_geometry.py --patient PAT001 --dicom-dir raw/tcia/PAT001/study_2/series_1 --visit 2 --date 2024-04-01 --threshold -200 300

# 患者 2
python scripts/build_geometry.py --patient PAT002 --dicom-dir raw/tcia/PAT002/study_1/series_1 --visit 1 --date 2024-01-15 --threshold -200 300
python scripts/build_geometry.py --patient PAT002 --dicom-dir raw/tcia/PAT002/study_2/series_1 --visit 2 --date 2024-05-15 --threshold -200 300

# ... 更多患者
```

然后继续数据融合和建模步骤。

## 需要帮助？

如果您在使用本地 DICOM 数据时遇到问题：

1. 检查日志文件: `build_geometry.log`
2. 使用 `--verbose` 或 `-v` 标志获取详细输出
3. 查看 `examples/minicase/README.md` 中的示例
