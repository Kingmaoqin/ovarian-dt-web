"""
网格生成模块

使用 Marching Cubes 从体素 mask 生成三角网格，
可选使用 Poisson 重建生成平滑表面。

主要功能：
- Marching Cubes 表面提取
- 网格平滑
- Poisson 表面重建
- 计算体积、表面积
- 保存为 STL / VTP 格式
"""

import logging
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple
from skimage import measure

logger = logging.getLogger(__name__)


class MeshGenerator:
    """网格生成器"""

    def __init__(self):
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None

    def marching_cubes(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        level: float = 0.5,
        smooth: bool = True,
        smooth_iterations: int = 5
    ) -> o3d.geometry.TriangleMesh:
        """
        使用 Marching Cubes 算法从 mask 生成网格

        Args:
            mask: 二值 mask
            spacing: 体素间距 (Z, Y, X) in mm
            level: 等值面阈值
            smooth: 是否平滑网格
            smooth_iterations: 平滑迭代次数

        Returns:
            Open3D TriangleMesh
        """
        logger.info("Marching Cubes 表面提取...")

        # 确保 mask 是浮点类型
        volume = mask.astype(np.float32)

        # 运行 Marching Cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                volume,
                level=level,
                spacing=spacing,
                allow_degenerate=False
            )
        except Exception as e:
            logger.error(f"Marching Cubes 失败: {e}")
            raise

        logger.info(f"生成网格: {len(verts)} 顶点, {len(faces)} 三角形")

        # 创建 Open3D 网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # 计算法向量
        mesh.compute_vertex_normals()

        # 平滑（可选）
        if smooth and smooth_iterations > 0:
            mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
            mesh.compute_vertex_normals()
            logger.info(f"网格平滑: {smooth_iterations} 次迭代")

        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.normals = np.asarray(mesh.vertex_normals)

        return mesh

    def poisson_reconstruction(
        self,
        pointcloud: o3d.geometry.PointCloud,
        depth: int = 9,
        density_threshold: float = 0.01
    ) -> o3d.geometry.TriangleMesh:
        """
        使用 Poisson 重建从点云生成平滑表面

        Args:
            pointcloud: Open3D PointCloud（需包含法向量）
            depth: Poisson 深度（越大越精细，但越慢）
            density_threshold: 密度阈值（用于裁剪低支持区域）

        Returns:
            Open3D TriangleMesh
        """
        logger.info(f"Poisson 表面重建 (depth={depth})...")

        if not pointcloud.has_normals():
            logger.info("估计点云法向量...")
            pointcloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=2.0,
                    max_nn=30
                )
            )
            pointcloud.orient_normals_consistent_tangent_plane(k=15)

        # Poisson 重建
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pointcloud,
                depth=depth,
                width=0,
                scale=1.1,
                linear_fit=False
            )
        except Exception as e:
            logger.error(f"Poisson 重建失败: {e}")
            raise

        # 根据密度裁剪
        if density_threshold > 0:
            densities = np.asarray(densities)
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            logger.info(f"密度阈值裁剪: {density_threshold}")

        # 计算法向量
        mesh.compute_vertex_normals()

        logger.info(f"Poisson 重建完成: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角形")

        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        self.normals = np.asarray(mesh.vertex_normals)

        return mesh

    def compute_volume(self, mesh: Optional[o3d.geometry.TriangleMesh] = None) -> float:
        """
        计算网格体积（单位：立方毫米）

        Args:
            mesh: Open3D TriangleMesh（可选，使用当前 mesh）

        Returns:
            体积 (mm³)
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("尚未生成网格")

        # 确保网格是水密的
        if not mesh.is_watertight():
            logger.warning("网格不是水密的，体积计算可能不准确")

        volume = mesh.get_volume()

        # 如果体积为负，取绝对值
        volume = abs(volume)

        logger.info(f"网格体积: {volume:.2f} mm³ ({volume / 1000:.2f} cm³)")

        return volume

    def compute_surface_area(self, mesh: Optional[o3d.geometry.TriangleMesh] = None) -> float:
        """
        计算网格表面积（单位：平方毫米）

        Args:
            mesh: Open3D TriangleMesh（可选，使用当前 mesh）

        Returns:
            表面积 (mm²)
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("尚未生成网格")

        surface_area = mesh.get_surface_area()

        logger.info(f"网格表面积: {surface_area:.2f} mm² ({surface_area / 100:.2f} cm²)")

        return surface_area

    def save_stl(self, output_path: Path, mesh: Optional[o3d.geometry.TriangleMesh] = None):
        """
        保存网格为 STL 格式

        Args:
            output_path: 输出文件路径
            mesh: Open3D TriangleMesh（可选，使用当前 mesh）
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("尚未生成网格")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        logger.info(f"保存 STL: {output_path}")

    def save_vtp(self, output_path: Path, mesh: Optional[o3d.geometry.TriangleMesh] = None):
        """
        保存网格为 VTP 格式（VTK PolyData）

        Args:
            output_path: 输出文件路径
            mesh: Open3D TriangleMesh（可选，使用当前 mesh）
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("尚未生成网格")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open3D 不直接支持 VTP，我们手动写入
        # 这里使用简化的 XML 格式
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        with open(output_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <PolyData>\n')
            f.write(f'    <Piece NumberOfPoints="{len(vertices)}" NumberOfPolys="{len(faces)}">\n')

            # Points
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            for v in vertices:
                f.write(f'          {v[0]} {v[1]} {v[2]}\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')

            # Polys
            f.write('      <Polys>\n')
            f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            for face in faces:
                f.write(f'          {face[0]} {face[1]} {face[2]}\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            for i in range(len(faces)):
                f.write(f'          {(i + 1) * 3}\n')
            f.write('        </DataArray>\n')
            f.write('      </Polys>\n')

            f.write('    </Piece>\n')
            f.write('  </PolyData>\n')
            f.write('</VTKFile>\n')

        logger.info(f"保存 VTP: {output_path}")

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        """获取当前网格"""
        if self.mesh is None:
            raise ValueError("尚未生成网格")
        return self.mesh


def main():
    """测试网格生成"""
    import argparse

    parser = argparse.ArgumentParser(description="网格生成测试")
    parser.add_argument("--mask", type=Path, required=True, help="二值 mask NPY 文件")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       help="体素间距 (Z Y X) in mm")
    parser.add_argument("--smooth", type=int, default=5, help="平滑迭代次数")
    parser.add_argument("--output-stl", type=Path, help="输出 STL 文件")
    parser.add_argument("--output-vtp", type=Path, help="输出 VTP 文件")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 加载 mask
    logger.info(f"加载 mask: {args.mask}")
    mask = np.load(args.mask)
    logger.info(f"Mask 形状: {mask.shape}")

    # 生成网格
    generator = MeshGenerator()
    mesh = generator.marching_cubes(
        mask,
        spacing=tuple(args.spacing),
        smooth=args.smooth > 0,
        smooth_iterations=args.smooth
    )

    # 计算度量
    volume = generator.compute_volume()
    surface_area = generator.compute_surface_area()

    print("\n" + "=" * 60)
    print("网格信息")
    print("=" * 60)
    print(f"顶点数:     {len(mesh.vertices)}")
    print(f"三角形数:   {len(mesh.triangles)}")
    print(f"体积:       {volume:.2f} mm³ ({volume / 1000:.2f} cm³)")
    print(f"表面积:     {surface_area:.2f} mm² ({surface_area / 100:.2f} cm²)")
    print(f"水密性:     {'是' if mesh.is_watertight() else '否'}")
    print("=" * 60)

    # 保存
    if args.output_stl:
        generator.save_stl(args.output_stl)

    if args.output_vtp:
        generator.save_vtp(args.output_vtp)


if __name__ == "__main__":
    main()
