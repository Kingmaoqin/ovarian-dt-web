"""
点云生成模块

从网格或体数据生成点云，并进行下采样与法向量估计。

主要功能：
- 从网格均匀采样点
- 从体数据直接采样
- Voxel 下采样
- 法向量估计
- 保存为 PLY 格式
"""

import logging
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class PointCloudGenerator:
    """点云生成器"""

    def __init__(self):
        self.pointcloud: Optional[o3d.geometry.PointCloud] = None

    def from_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        num_points: int = 10000,
        uniform: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        从网格采样点云

        Args:
            mesh: Open3D TriangleMesh
            num_points: 采样点数
            uniform: 是否均匀采样（否则按面积采样）

        Returns:
            Open3D PointCloud
        """
        logger.info(f"从网格采样点云: {num_points} 个点 (均匀={uniform})")

        if uniform:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        else:
            pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

        logger.info(f"采样完成: {len(pcd.points)} 个点")

        self.pointcloud = pcd
        return pcd

    def from_volume(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sample_rate: float = 1.0
    ) -> o3d.geometry.PointCloud:
        """
        从体数据直接采样点云

        Args:
            volume: 3D 体数据
            mask: 二值 mask
            spacing: 体素间距 (Z, Y, X) in mm
            sample_rate: 采样率 (0.0-1.0)

        Returns:
            Open3D PointCloud
        """
        logger.info(f"从体数据采样点云 (采样率={sample_rate})")

        # 获取 mask 中的阳性体素坐标
        coords = np.argwhere(mask)

        # 下采样（可选）
        if sample_rate < 1.0:
            num_samples = int(len(coords) * sample_rate)
            indices = np.random.choice(len(coords), num_samples, replace=False)
            coords = coords[indices]

        # 转换为物理坐标
        # coords 是 (N, 3)，每行是 (z, y, x)
        # 乘以 spacing 得到物理坐标
        points = coords.astype(np.float32) * np.array(spacing).reshape(1, 3)

        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 可选：添加强度作为颜色
        intensities = volume[coords[:, 0], coords[:, 1], coords[:, 2]]
        # 归一化到 [0, 1]
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
        colors = np.stack([intensities, intensities, intensities], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        logger.info(f"采样完成: {len(pcd.points)} 个点")

        self.pointcloud = pcd
        return pcd

    def voxel_downsample(
        self,
        pcd: Optional[o3d.geometry.PointCloud] = None,
        voxel_size: float = 1.0
    ) -> o3d.geometry.PointCloud:
        """
        Voxel 下采样

        Args:
            pcd: Open3D PointCloud（可选，使用当前点云）
            voxel_size: 体素大小 (mm)

        Returns:
            下采样后的点云
        """
        if pcd is None:
            pcd = self.pointcloud

        if pcd is None:
            raise ValueError("尚未生成点云")

        original_count = len(pcd.points)
        logger.info(f"Voxel 下采样: 体素大小={voxel_size} mm")

        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        logger.info(f"下采样完成: {original_count} -> {len(pcd_down.points)} 个点 "
                   f"(保留 {len(pcd_down.points) / original_count * 100:.1f}%)")

        self.pointcloud = pcd_down
        return pcd_down

    def estimate_normals(
        self,
        pcd: Optional[o3d.geometry.PointCloud] = None,
        radius: float = 2.0,
        max_nn: int = 30
    ) -> o3d.geometry.PointCloud:
        """
        估计点云法向量

        Args:
            pcd: Open3D PointCloud（可选，使用当前点云）
            radius: 搜索半径 (mm)
            max_nn: 最大近邻数

        Returns:
            包含法向量的点云
        """
        if pcd is None:
            pcd = self.pointcloud

        if pcd is None:
            raise ValueError("尚未生成点云")

        logger.info(f"估计法向量: radius={radius} mm, max_nn={max_nn}")

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=max_nn
            )
        )

        # 一致化法向量方向
        pcd.orient_normals_consistent_tangent_plane(k=15)

        logger.info("法向量估计完成")

        return pcd

    def save_ply(
        self,
        output_path: Path,
        pcd: Optional[o3d.geometry.PointCloud] = None,
        binary: bool = True
    ):
        """
        保存点云为 PLY 格式

        Args:
            output_path: 输出文件路径
            pcd: Open3D PointCloud（可选，使用当前点云）
            binary: 是否使用二进制格式
        """
        if pcd is None:
            pcd = self.pointcloud

        if pcd is None:
            raise ValueError("尚未生成点云")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(
            str(output_path),
            pcd,
            write_ascii=not binary
        )
        logger.info(f"保存点云: {output_path} ({len(pcd.points)} 个点)")

    def get_pointcloud(self) -> o3d.geometry.PointCloud:
        """获取当前点云"""
        if self.pointcloud is None:
            raise ValueError("尚未生成点云")
        return self.pointcloud

    def compute_statistics(self, pcd: Optional[o3d.geometry.PointCloud] = None) -> dict:
        """
        计算点云统计信息

        Args:
            pcd: Open3D PointCloud（可选，使用当前点云）

        Returns:
            统计信息字典
        """
        if pcd is None:
            pcd = self.pointcloud

        if pcd is None:
            raise ValueError("尚未生成点云")

        points = np.asarray(pcd.points)

        stats = {
            'num_points': len(points),
            'has_normals': pcd.has_normals(),
            'has_colors': pcd.has_colors(),
            'bounding_box_min': points.min(axis=0).tolist(),
            'bounding_box_max': points.max(axis=0).tolist(),
            'centroid': points.mean(axis=0).tolist(),
        }

        return stats


def main():
    """测试点云生成"""
    import argparse

    parser = argparse.ArgumentParser(description="点云生成测试")
    parser.add_argument("--mesh", type=Path, help="网格文件 (STL)")
    parser.add_argument("--volume", type=Path, help="体数据 NPY 文件")
    parser.add_argument("--mask", type=Path, help="二值 mask NPY 文件")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="体素间距 (Z Y X) in mm")
    parser.add_argument("--num-points", type=int, default=10000,
                       help="采样点数（从网格）")
    parser.add_argument("--voxel-size", type=float, default=1.5,
                       help="下采样体素大小 (mm)")
    parser.add_argument("--output", type=Path, required=True, help="输出 PLY 文件")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    generator = PointCloudGenerator()

    # 从网格生成
    if args.mesh:
        logger.info(f"从网格生成点云: {args.mesh}")
        mesh = o3d.io.read_triangle_mesh(str(args.mesh))
        pcd = generator.from_mesh(mesh, num_points=args.num_points)

    # 从体数据生成
    elif args.volume and args.mask:
        logger.info(f"从体数据生成点云")
        volume = np.load(args.volume)
        mask = np.load(args.mask)
        pcd = generator.from_volume(volume, mask, spacing=tuple(args.spacing))

    else:
        raise ValueError("需要指定 --mesh 或 --volume + --mask")

    # 下采样
    pcd = generator.voxel_downsample(voxel_size=args.voxel_size)

    # 估计法向量
    pcd = generator.estimate_normals()

    # 统计信息
    stats = generator.compute_statistics()

    print("\n" + "=" * 60)
    print("点云信息")
    print("=" * 60)
    print(f"点数:       {stats['num_points']}")
    print(f"法向量:     {'是' if stats['has_normals'] else '否'}")
    print(f"颜色:       {'是' if stats['has_colors'] else '否'}")
    print(f"边界框:     {stats['bounding_box_min']} ~ {stats['bounding_box_max']}")
    print(f"质心:       {stats['centroid']}")
    print("=" * 60)

    # 保存
    generator.save_ply(args.output)


if __name__ == "__main__":
    main()
