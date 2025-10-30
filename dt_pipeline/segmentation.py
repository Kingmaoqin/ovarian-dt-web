"""
分割模块

提供多种分割方法：
- 阈值分割
- 区域生长
- 加载外部 mask (NIfTI/NPY)

这是占位实现，便于后续替换为医生标注或 3D Slicer 导出的结果。
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from skimage import measure, morphology
from scipy import ndimage

logger = logging.getLogger(__name__)


class Segmenter:
    """医学影像分割器"""

    def __init__(self):
        self.mask: Optional[np.ndarray] = None

    def threshold_segmentation(
        self,
        volume: np.ndarray,
        lower: float,
        upper: float,
        min_size: int = 100,
        fill_holes: bool = True
    ) -> np.ndarray:
        """
        阈值分割

        Args:
            volume: 3D 体数据
            lower: 下限阈值
            upper: 上限阈值
            min_size: 最小连通域体积（体素数）
            fill_holes: 是否填充空洞

        Returns:
            二值 mask
        """
        logger.info(f"阈值分割: [{lower}, {upper}]")

        # 应用阈值
        mask = (volume >= lower) & (volume <= upper)

        # 移除小连通域
        if min_size > 0:
            mask = morphology.remove_small_objects(mask, min_size=min_size)
            logger.info(f"移除小于 {min_size} 体素的连通域")

        # 填充空洞
        if fill_holes:
            mask = ndimage.binary_fill_holes(mask)
            logger.info("填充空洞")

        # 保留最大连通域（假设肿瘤是最大的）
        mask = self._keep_largest_component(mask)

        positive_voxels = np.sum(mask)
        logger.info(f"分割得到 {positive_voxels} 个阳性体素")

        self.mask = mask
        return mask

    def region_growing(
        self,
        volume: np.ndarray,
        seed_point: Tuple[int, int, int],
        threshold: float = 50.0,
        connectivity: int = 1
    ) -> np.ndarray:
        """
        区域生长分割

        Args:
            volume: 3D 体数据
            seed_point: 种子点 (z, y, x)
            threshold: 生长阈值（与种子点的强度差）
            connectivity: 连通性 (1: 6-邻域, 2: 18-邻域, 3: 26-邻域)

        Returns:
            二值 mask
        """
        logger.info(f"区域生长分割: 种子点={seed_point}, 阈值={threshold}")

        # 检查种子点
        if not (0 <= seed_point[0] < volume.shape[0] and
                0 <= seed_point[1] < volume.shape[1] and
                0 <= seed_point[2] < volume.shape[2]):
            raise ValueError(f"种子点超出范围: {seed_point}")

        seed_value = volume[seed_point]
        logger.info(f"种子点强度: {seed_value}")

        # 创建 mask
        mask = np.zeros(volume.shape, dtype=bool)
        mask[seed_point] = True

        # 定义邻域结构
        if connectivity == 1:
            struct = ndimage.generate_binary_structure(3, 1)  # 6-邻域
        elif connectivity == 2:
            struct = ndimage.generate_binary_structure(3, 2)  # 18-邻域
        else:
            struct = ndimage.generate_binary_structure(3, 3)  # 26-邻域

        # 迭代生长
        changed = True
        iteration = 0
        max_iterations = 1000

        while changed and iteration < max_iterations:
            # 膨胀 mask
            dilated = ndimage.binary_dilation(mask, structure=struct)

            # 找到新的候选体素
            candidates = dilated & ~mask

            # 检查候选体素是否满足阈值条件
            candidate_values = volume[candidates]
            valid = np.abs(candidate_values - seed_value) <= threshold

            # 更新 mask
            new_mask = mask.copy()
            candidate_coords = np.where(candidates)
            for i in range(len(candidate_coords[0])):
                if valid[i]:
                    z, y, x = candidate_coords[0][i], candidate_coords[1][i], candidate_coords[2][i]
                    new_mask[z, y, x] = True

            # 检查是否有变化
            changed = not np.array_equal(mask, new_mask)
            mask = new_mask
            iteration += 1

        positive_voxels = np.sum(mask)
        logger.info(f"区域生长完成: {iteration} 次迭代, {positive_voxels} 个阳性体素")

        self.mask = mask
        return mask

    def load_mask(
        self,
        mask_path: Path,
        expected_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        加载外部 mask

        支持格式：
        - .npy: NumPy 二值数组
        - .nii / .nii.gz: NIfTI 格式（需要 nibabel）

        Args:
            mask_path: mask 文件路径
            expected_shape: 期望的形状（可选，用于验证）

        Returns:
            二值 mask
        """
        logger.info(f"加载 mask: {mask_path}")

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask 文件不存在: {mask_path}")

        # 根据扩展名加载
        suffix = mask_path.suffix.lower()

        if suffix == '.npy':
            mask = np.load(mask_path)

        elif suffix in ['.nii', '.gz']:
            try:
                import nibabel as nib
                nii = nib.load(mask_path)
                mask = nii.get_fdata()
            except ImportError:
                raise ImportError("加载 NIfTI 需要安装 nibabel: pip install nibabel")

        else:
            raise ValueError(f"不支持的 mask 格式: {suffix}")

        # 转换为布尔类型
        mask = mask.astype(bool)

        # 验证形状
        if expected_shape is not None and mask.shape != expected_shape:
            logger.warning(f"Mask 形状不匹配: {mask.shape} != {expected_shape}")

        positive_voxels = np.sum(mask)
        logger.info(f"加载完成: {mask.shape}, {positive_voxels} 个阳性体素")

        self.mask = mask
        return mask

    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        保留最大连通域

        Args:
            mask: 二值 mask

        Returns:
            只包含最大连通域的 mask
        """
        labeled = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled)

        if not regions:
            logger.warning("未找到连通域")
            return mask

        # 找到最大区域
        largest_region = max(regions, key=lambda r: r.area)

        # 创建新 mask
        new_mask = labeled == largest_region.label

        logger.info(f"保留最大连通域: {largest_region.area} 体素 "
                   f"(共 {len(regions)} 个连通域)")

        return new_mask

    def get_mask(self) -> np.ndarray:
        """获取当前 mask"""
        if self.mask is None:
            raise ValueError("尚未生成 mask")
        return self.mask

    def save_mask(self, output_path: Path):
        """
        保存 mask 为 NPY 文件

        Args:
            output_path: 输出文件路径
        """
        if self.mask is None:
            raise ValueError("尚未生成 mask")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self.mask)
        logger.info(f"保存 mask: {output_path}")

    def compute_statistics(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        计算分割区域的统计信息

        Args:
            volume: 3D 体数据
            mask: 二值 mask（可选，使用当前 mask）

        Returns:
            统计信息字典
        """
        if mask is None:
            mask = self.mask

        if mask is None:
            raise ValueError("尚未生成 mask")

        # 提取分割区域的体素值
        roi_values = volume[mask]

        stats = {
            'num_voxels': int(np.sum(mask)),
            'mean_intensity': float(np.mean(roi_values)),
            'std_intensity': float(np.std(roi_values)),
            'min_intensity': float(np.min(roi_values)),
            'max_intensity': float(np.max(roi_values)),
            'median_intensity': float(np.median(roi_values)),
        }

        return stats


def main():
    """测试分割功能"""
    import argparse

    parser = argparse.ArgumentParser(description="分割功能测试")
    parser.add_argument("--volume", type=Path, required=True, help="3D 体 NPY 文件")
    parser.add_argument("--method", choices=['threshold', 'region'], default='threshold', help="分割方法")
    parser.add_argument("--lower", type=float, default=-200, help="阈值下限")
    parser.add_argument("--upper", type=float, default=300, help="阈值上限")
    parser.add_argument("--seed", type=int, nargs=3, help="区域生长种子点 (z y x)")
    parser.add_argument("--output", type=Path, help="输出 mask 文件")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 加载体积
    logger.info(f"加载体积: {args.volume}")
    volume = np.load(args.volume)
    logger.info(f"体积形状: {volume.shape}")

    # 分割
    segmenter = Segmenter()

    if args.method == 'threshold':
        mask = segmenter.threshold_segmentation(volume, args.lower, args.upper)

    elif args.method == 'region':
        if args.seed is None:
            # 使用体积中心作为默认种子点
            seed = tuple(s // 2 for s in volume.shape)
            logger.info(f"使用默认种子点: {seed}")
        else:
            seed = tuple(args.seed)

        mask = segmenter.region_growing(volume, seed)

    # 统计
    stats = segmenter.compute_statistics(volume, mask)

    print("\n" + "=" * 60)
    print("分割结果统计")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    print("=" * 60)

    # 保存
    if args.output:
        segmenter.save_mask(args.output)


if __name__ == "__main__":
    main()
