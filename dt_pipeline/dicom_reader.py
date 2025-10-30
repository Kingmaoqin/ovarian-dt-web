"""
DICOM 读取模块

使用 pydicom 读取 DICOM 序列，重建 3D 体数据。

主要功能：
- 读取 DICOM 序列
- 按 ImagePositionPatient / SliceLocation 排序
- 重建 3D 体 (Z, Y, X)
- 提取体素间距 (spacing)
- HU 值处理
"""

import logging
import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class DICOMReader:
    """DICOM 序列读取器"""

    def __init__(self):
        self.dicom_files: List[Path] = []
        self.slices: List[pydicom.Dataset] = []
        self.volume: Optional[np.ndarray] = None
        self.spacing: Optional[Tuple[float, float, float]] = None
        self.origin: Optional[Tuple[float, float, float]] = None
        self.metadata: Dict[str, Any] = {}

    def load_series(self, dicom_dir: Path) -> 'DICOMReader':
        """
        加载 DICOM 序列

        Args:
            dicom_dir: DICOM 文件目录

        Returns:
            self
        """
        logger.info(f"加载 DICOM 序列: {dicom_dir}")

        if not dicom_dir.exists():
            raise FileNotFoundError(f"目录不存在: {dicom_dir}")

        # 查找所有 DICOM 文件
        self.dicom_files = []
        for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']:
            self.dicom_files.extend(dicom_dir.glob(ext))

        # 如果没有找到带扩展名的文件，尝试读取所有文件
        if not self.dicom_files:
            for file in dicom_dir.iterdir():
                if file.is_file():
                    try:
                        pydicom.dcmread(file, stop_before_pixels=True)
                        self.dicom_files.append(file)
                    except Exception:
                        pass

        if not self.dicom_files:
            raise ValueError(f"目录中未找到 DICOM 文件: {dicom_dir}")

        logger.info(f"找到 {len(self.dicom_files)} 个 DICOM 文件")

        # 读取 DICOM 切片
        self.slices = []
        for file_path in self.dicom_files:
            try:
                ds = pydicom.dcmread(file_path)
                self.slices.append(ds)
            except Exception as e:
                logger.warning(f"读取失败 {file_path}: {e}")

        if not self.slices:
            raise ValueError("无法读取任何 DICOM 文件")

        logger.info(f"成功读取 {len(self.slices)} 个切片")

        # 排序切片
        self._sort_slices()

        # 提取元数据
        self._extract_metadata()

        return self

    def _sort_slices(self):
        """根据 ImagePositionPatient 或 SliceLocation 排序切片"""
        # 尝试使用 ImagePositionPatient 的 Z 坐标
        try:
            positions = []
            for ds in self.slices:
                if hasattr(ds, 'ImagePositionPatient'):
                    # Z 坐标是第三个元素
                    z = float(ds.ImagePositionPatient[2])
                    positions.append((z, ds))
                elif hasattr(ds, 'SliceLocation'):
                    z = float(ds.SliceLocation)
                    positions.append((z, ds))
                else:
                    # 使用 InstanceNumber 作为后备
                    z = float(ds.get('InstanceNumber', 0))
                    positions.append((z, ds))

            # 按 Z 坐标排序
            positions.sort(key=lambda x: x[0])
            self.slices = [ds for z, ds in positions]

            logger.info(f"切片已排序，Z 范围: {positions[0][0]:.2f} ~ {positions[-1][0]:.2f}")

        except Exception as e:
            logger.warning(f"切片排序失败: {e}，使用原始顺序")

    def _extract_metadata(self):
        """提取序列元数据"""
        if not self.slices:
            return

        ds = self.slices[0]  # 使用第一个切片的元数据

        self.metadata = {
            'PatientID': str(ds.get('PatientID', 'Unknown')),
            'StudyDate': str(ds.get('StudyDate', '')),
            'StudyDescription': str(ds.get('StudyDescription', '')),
            'SeriesDescription': str(ds.get('SeriesDescription', '')),
            'Modality': str(ds.get('Modality', '')),
            'Manufacturer': str(ds.get('Manufacturer', '')),
            'SliceThickness': float(ds.get('SliceThickness', 1.0)),
            'Rows': int(ds.get('Rows', 0)),
            'Columns': int(ds.get('Columns', 0)),
            'NumberOfSlices': len(self.slices),
        }

        # 体素间距 (spacing)
        pixel_spacing = ds.get('PixelSpacing', [1.0, 1.0])
        slice_thickness = float(ds.get('SliceThickness', 1.0))

        # 计算实际切片间距
        if len(self.slices) > 1:
            try:
                z1 = float(self.slices[0].ImagePositionPatient[2])
                z2 = float(self.slices[1].ImagePositionPatient[2])
                actual_spacing = abs(z2 - z1)
                if actual_spacing > 0:
                    slice_thickness = actual_spacing
            except Exception:
                pass

        # spacing 顺序: (slice_spacing, row_spacing, col_spacing)
        # 对应于 (Z, Y, X)
        self.spacing = (
            slice_thickness,
            float(pixel_spacing[0]),
            float(pixel_spacing[1])
        )

        # 原点 (origin)
        if hasattr(ds, 'ImagePositionPatient'):
            self.origin = tuple(float(x) for x in ds.ImagePositionPatient)
        else:
            self.origin = (0.0, 0.0, 0.0)

        logger.info(f"体素间距 (Z, Y, X): {self.spacing}")
        logger.info(f"原点: {self.origin}")
        logger.info(f"体积形状: {self.metadata['NumberOfSlices']} × "
                   f"{self.metadata['Rows']} × {self.metadata['Columns']}")

    def get_volume(
        self,
        apply_hu: bool = True,
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        重建 3D 体

        Args:
            apply_hu: 是否转换为 HU 值（CT 扫描）
            dtype: 输出数据类型

        Returns:
            3D numpy 数组，形状 (Z, Y, X)
        """
        if self.volume is not None:
            return self.volume

        if not self.slices:
            raise ValueError("未加载 DICOM 序列")

        logger.info("重建 3D 体...")

        # 读取第一个切片以获取形状
        first_slice = self.slices[0]
        rows = int(first_slice.Rows)
        cols = int(first_slice.Columns)
        num_slices = len(self.slices)

        # 创建 3D 数组
        volume = np.zeros((num_slices, rows, cols), dtype=dtype)

        # 填充体素
        for i, ds in enumerate(self.slices):
            # 获取像素数据
            pixels = ds.pixel_array.astype(dtype)

            # 应用 Rescale Slope 和 Intercept（HU 值转换）
            if apply_hu and hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixels = pixels * slope + intercept

            volume[i, :, :] = pixels

        self.volume = volume

        logger.info(f"3D 体重建完成: {volume.shape}, dtype={volume.dtype}")
        logger.info(f"强度范围: [{volume.min():.2f}, {volume.max():.2f}]")

        return volume

    def get_spacing(self) -> Tuple[float, float, float]:
        """
        获取体素间距

        Returns:
            (Z, Y, X) 体素间距 (mm)
        """
        if self.spacing is None:
            raise ValueError("未加载 DICOM 序列")
        return self.spacing

    def get_origin(self) -> Tuple[float, float, float]:
        """
        获取原点坐标

        Returns:
            (X, Y, Z) 原点坐标
        """
        if self.origin is None:
            raise ValueError("未加载 DICOM 序列")
        return self.origin

    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        return self.metadata

    def save_volume_npy(self, output_path: Path):
        """
        保存 3D 体为 NPY 文件

        Args:
            output_path: 输出文件路径
        """
        if self.volume is None:
            self.get_volume()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self.volume)
        logger.info(f"保存 3D 体: {output_path}")


def main():
    """测试 DICOM 读取"""
    import argparse

    parser = argparse.ArgumentParser(description="DICOM 序列读取测试")
    parser.add_argument("--dicom-dir", type=Path, required=True, help="DICOM 文件目录")
    parser.add_argument("--output", type=Path, help="输出 NPY 文件路径")
    parser.add_argument("--no-hu", action="store_true", help="不转换为 HU 值")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 读取 DICOM
    reader = DICOMReader()
    reader.load_series(args.dicom_dir)

    # 获取体积
    volume = reader.get_volume(apply_hu=not args.no_hu)
    spacing = reader.get_spacing()
    metadata = reader.get_metadata()

    print("\n" + "=" * 60)
    print("DICOM 序列信息")
    print("=" * 60)
    print(f"患者 ID:        {metadata['PatientID']}")
    print(f"检查日期:       {metadata['StudyDate']}")
    print(f"检查描述:       {metadata['StudyDescription']}")
    print(f"序列描述:       {metadata['SeriesDescription']}")
    print(f"模态:           {metadata['Modality']}")
    print(f"体积形状:       {volume.shape}")
    print(f"体素间距 (mm):  {spacing}")
    print(f"强度范围:       [{volume.min():.2f}, {volume.max():.2f}]")
    print("=" * 60)

    # 保存
    if args.output:
        reader.save_volume_npy(args.output)


if __name__ == "__main__":
    main()
