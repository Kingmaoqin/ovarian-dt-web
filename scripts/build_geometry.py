#!/usr/bin/env python3
"""
几何重建脚本

从 DICOM 序列生成网格和点云，并更新 timeline.json。

完整管线：
  DICOM → 3D 体 → 分割 mask → 网格（STL/VTP）→ 点云（PLY）→ timeline.json

用法:
    python scripts/build_geometry.py \\
        --patient PAT123 \\
        --dicom-dir raw/tcia/PAT123/study_xxx/series_yyy \\
        --visit 1 \\
        --date 2024-05-01

    python scripts/build_geometry.py \\
        --patient PAT123 \\
        --dicom-dir raw/tcia/PAT123/study_xxx/series_yyy \\
        --mask path/to/mask.nii.gz \\
        --visit 1 \\
        --date 2024-05-01

    python scripts/build_geometry.py \\
        --patient PAT123 \\
        --dicom-dir raw/tcia/PAT123/study_xxx/series_yyy \\
        --threshold -200 300 \\
        --poisson \\
        --voxel 1.5
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dt_pipeline.dicom_reader import DICOMReader
from dt_pipeline.segmentation import Segmenter
from dt_pipeline.mesh_generator import MeshGenerator
from dt_pipeline.pointcloud_generator import PointCloudGenerator


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('build_geometry.log')
        ]
    )


def update_timeline(
    timeline_path: Path,
    patient_id: str,
    visit: int,
    date: str,
    ply_path: str,
    vtp_path: str,
    vol_cc: float,
    surface_area_cm2: float,
    num_points: int
):
    """
    更新 timeline.json

    Args:
        timeline_path: timeline.json 路径
        patient_id: 患者 ID
        visit: 就诊编号
        date: 就诊日期
        ply_path: 点云文件相对路径
        vtp_path: 网格文件相对路径
        vol_cc: 体积 (cm³)
        surface_area_cm2: 表面积 (cm²)
        num_points: 点数
    """
    logger = logging.getLogger(__name__)

    # 加载现有 timeline
    if timeline_path.exists():
        with open(timeline_path, 'r') as f:
            timeline = json.load(f)
    else:
        timeline = []

    # 查找患者
    patient_record = None
    for record in timeline:
        if record['patient_id'] == patient_id:
            patient_record = record
            break

    # 如果患者不存在，创建新记录
    if patient_record is None:
        patient_record = {
            'patient_id': patient_id,
            'visits': []
        }
        timeline.append(patient_record)

    # 查找或创建就诊记录
    visit_record = None
    for v in patient_record['visits']:
        if v['visit'] == visit:
            visit_record = v
            break

    if visit_record is None:
        visit_record = {'visit': visit}
        patient_record['visits'].append(visit_record)

    # 更新就诊记录
    visit_record.update({
        'date': date,
        'ply': ply_path,
        'vtp': vtp_path,
        'vol_cc': round(vol_cc, 2),
        'surface_area_cm2': round(surface_area_cm2, 2),
        'num_points': num_points,
        'updated_at': datetime.now().isoformat()
    })

    # 按就诊编号排序
    patient_record['visits'].sort(key=lambda x: x['visit'])

    # 保存
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timeline_path, 'w', encoding='utf-8') as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)

    logger.info(f"更新 timeline.json: {patient_id}, visit {visit}")


def main():
    parser = argparse.ArgumentParser(
        description="从 DICOM 生成网格和点云",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用阈值分割
  python scripts/build_geometry.py --patient PAT123 \\
      --dicom-dir raw/tcia/PAT123/study_1/series_1 \\
      --visit 1 --date 2024-05-01 \\
      --threshold -200 300

  # 使用外部 mask
  python scripts/build_geometry.py --patient PAT123 \\
      --dicom-dir raw/tcia/PAT123/study_1/series_1 \\
      --mask path/to/mask.nii.gz \\
      --visit 1 --date 2024-05-01

  # 使用 Poisson 重建
  python scripts/build_geometry.py --patient PAT123 \\
      --dicom-dir raw/tcia/PAT123/study_1/series_1 \\
      --visit 1 --date 2024-05-01 \\
      --threshold -200 300 --poisson
        """
    )

    # 必需参数
    parser.add_argument("--patient", required=True, help="患者 ID")
    parser.add_argument("--dicom-dir", type=Path, required=True, help="DICOM 文件目录")
    parser.add_argument("--visit", type=int, required=True, help="就诊编号")
    parser.add_argument("--date", required=True, help="就诊日期 (YYYY-MM-DD)")

    # 分割参数
    seg_group = parser.add_mutually_exclusive_group()
    seg_group.add_argument("--mask", type=Path, help="外部 mask 文件 (NPY/NIfTI)")
    seg_group.add_argument("--threshold", type=float, nargs=2, metavar=('LOWER', 'UPPER'),
                          help="阈值分割范围（例如：-200 300）")

    # 几何参数
    parser.add_argument("--spacing-override", type=float, nargs=3, metavar=('Z', 'Y', 'X'),
                       help="覆盖体素间距 (mm)")
    parser.add_argument("--voxel", type=float, default=1.5,
                       help="点云下采样体素大小 (mm, 默认: 1.5)")
    parser.add_argument("--poisson", action="store_true",
                       help="使用 Poisson 重建生成平滑表面")
    parser.add_argument("--poisson-depth", type=int, default=9,
                       help="Poisson 深度 (默认: 9)")

    # 输出参数
    parser.add_argument("--out-dir", type=Path, default=Path("ovarian-dt-web/data"),
                       help="输出根目录 (默认: ovarian-dt-web/data)")
    parser.add_argument("--timeline", type=Path, default=Path("ovarian-dt-web/data/timeline.json"),
                       help="timeline.json 路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 70)
    print("几何重建管线")
    print("=" * 70)
    print(f"患者:       {args.patient}")
    print(f"就诊:       Visit {args.visit} ({args.date})")
    print(f"DICOM:      {args.dicom_dir}")
    print("=" * 70 + "\n")

    # 创建输出目录
    output_dir = args.out_dir / args.patient / f"visit_{args.visit}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # ========== 步骤 1: 读取 DICOM ==========
    print("[1/5] 读取 DICOM 序列...")
    reader = DICOMReader()
    reader.load_series(args.dicom_dir)
    volume = reader.get_volume(apply_hu=True)
    spacing = reader.get_spacing()

    # 覆盖 spacing（如果指定）
    if args.spacing_override:
        spacing = tuple(args.spacing_override)
        logger.info(f"使用覆盖的体素间距: {spacing}")

    print(f"  ✓ 体积形状: {volume.shape}")
    print(f"  ✓ 体素间距: {spacing} mm")
    print(f"  ✓ 强度范围: [{volume.min():.1f}, {volume.max():.1f}]")

    # ========== 步骤 2: 分割 ==========
    print("\n[2/5] 分割...")
    segmenter = Segmenter()

    if args.mask:
        # 加载外部 mask
        print(f"  加载外部 mask: {args.mask}")
        mask = segmenter.load_mask(args.mask, expected_shape=volume.shape)

    elif args.threshold:
        # 阈值分割
        lower, upper = args.threshold
        print(f"  阈值分割: [{lower}, {upper}]")
        mask = segmenter.threshold_segmentation(volume, lower, upper, min_size=100)

    else:
        # 默认：使用中心点区域生长（占位）
        print("  默认分割: 区域生长（从中心点）")
        seed_point = tuple(s // 2 for s in volume.shape)
        mask = segmenter.region_growing(volume, seed_point, threshold=100)

    stats = segmenter.compute_statistics(volume, mask)
    print(f"  ✓ 阳性体素: {stats['num_voxels']}")
    print(f"  ✓ 平均强度: {stats['mean_intensity']:.1f}")

    # 保存 mask
    mask_path = output_dir / "mask.npy"
    segmenter.save_mask(mask_path)

    # ========== 步骤 3: 生成网格 ==========
    print("\n[3/5] 生成网格...")
    mesh_gen = MeshGenerator()

    if args.poisson:
        # Poisson 重建流程
        print(f"  Poisson 重建 (depth={args.poisson_depth})...")

        # 先用 Marching Cubes 生成初始网格
        mesh_mc = mesh_gen.marching_cubes(mask, spacing, smooth=True)

        # 从网格采样点云
        pcd_gen = PointCloudGenerator()
        pcd_for_poisson = pcd_gen.from_mesh(mesh_mc, num_points=50000)
        pcd_for_poisson = pcd_gen.estimate_normals(radius=2.0)

        # Poisson 重建
        mesh = mesh_gen.poisson_reconstruction(
            pcd_for_poisson,
            depth=args.poisson_depth,
            density_threshold=0.01
        )

    else:
        # Marching Cubes
        print("  Marching Cubes...")
        mesh = mesh_gen.marching_cubes(mask, spacing, smooth=True, smooth_iterations=5)

    # 计算度量
    volume_mm3 = mesh_gen.compute_volume(mesh)
    volume_cc = volume_mm3 / 1000.0
    surface_area_mm2 = mesh_gen.compute_surface_area(mesh)
    surface_area_cm2 = surface_area_mm2 / 100.0

    print(f"  ✓ 顶点: {len(mesh.vertices)}, 三角形: {len(mesh.triangles)}")
    print(f"  ✓ 体积: {volume_cc:.2f} cm³")
    print(f"  ✓ 表面积: {surface_area_cm2:.2f} cm²")

    # 保存网格
    stl_path = output_dir / "tumor.stl"
    vtp_path = output_dir / "tumor.vtp"
    mesh_gen.save_stl(stl_path, mesh)
    mesh_gen.save_vtp(vtp_path, mesh)

    # ========== 步骤 4: 生成点云 ==========
    print("\n[4/5] 生成点云...")
    pcd_gen = PointCloudGenerator()

    # 从网格采样
    pcd = pcd_gen.from_mesh(mesh, num_points=20000, uniform=True)

    # 下采样
    pcd = pcd_gen.voxel_downsample(voxel_size=args.voxel)

    # 估计法向量
    pcd = pcd_gen.estimate_normals(radius=2.0, max_nn=30)

    pcd_stats = pcd_gen.compute_statistics()
    print(f"  ✓ 点数: {pcd_stats['num_points']}")
    print(f"  ✓ 法向量: {'是' if pcd_stats['has_normals'] else '否'}")

    # 保存点云
    ply_path = output_dir / "tumor.ply"
    pcd_gen.save_ply(ply_path)

    # ========== 步骤 5: 更新 timeline.json ==========
    print("\n[5/5] 更新 timeline.json...")

    # 计算相对路径
    ply_rel = f"{args.patient}/visit_{args.visit}/tumor.ply"
    vtp_rel = f"{args.patient}/visit_{args.visit}/tumor.vtp"

    update_timeline(
        timeline_path=args.timeline,
        patient_id=args.patient,
        visit=args.visit,
        date=args.date,
        ply_path=ply_rel,
        vtp_path=vtp_rel,
        vol_cc=volume_cc,
        surface_area_cm2=surface_area_cm2,
        num_points=pcd_stats['num_points']
    )

    print(f"  ✓ timeline.json 已更新")

    # ========== 完成 ==========
    print("\n" + "=" * 70)
    print("几何重建完成！")
    print("=" * 70)
    print(f"输出目录:   {output_dir}")
    print(f"文件:")
    print(f"  - tumor.stl    (网格)")
    print(f"  - tumor.vtp    (网格, VTK)")
    print(f"  - tumor.ply    (点云)")
    print(f"  - mask.npy     (分割 mask)")
    print(f"\n度量:")
    print(f"  - 体积:      {volume_cc:.2f} cm³")
    print(f"  - 表面积:    {surface_area_cm2:.2f} cm²")
    print(f"  - 点数:      {pcd_stats['num_points']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
