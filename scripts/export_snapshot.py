#!/usr/bin/env python3
"""
导出 3D 模型快照

使用 Open3D 离屏渲染生成标准角度的截图。

用法:
    python scripts/export_snapshot.py --patient PAT123 --visit 2 --out snapshot.png
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import open3d as o3d


def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def render_snapshot(
    ply_path: Path,
    output_path: Path,
    width: int = 1920,
    height: int = 1080
):
    """
    渲染快照

    Args:
        ply_path: PLY 文件路径
        output_path: 输出 PNG 路径
        width: 图像宽度
        height: 图像高度
    """
    logger = logging.getLogger(__name__)

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")

    logger.info(f"加载点云: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))

    logger.info(f"点云包含 {len(pcd.points)} 个点")

    # 创建可视化器（离屏）
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    # 添加点云
    vis.add_geometry(pcd)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.15])  # 深色背景
    opt.point_size = 2.0

    # 重置视图
    vis.reset_view_point(True)

    # 渲染并保存
    vis.poll_events()
    vis.update_renderer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(output_path), do_render=True)

    vis.destroy_window()

    logger.info(f"快照已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="导出 3D 模型快照")
    parser.add_argument("--patient", required=True, help="患者 ID")
    parser.add_argument("--visit", type=int, required=True, help="就诊编号")
    parser.add_argument("--out", type=Path, required=True, help="输出 PNG 文件")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ovarian-dt-web/data"),
        help="数据目录 (默认: ovarian-dt-web/data)"
    )
    parser.add_argument("--width", type=int, default=1920, help="图像宽度 (默认: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="图像高度 (默认: 1080)")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 构建 PLY 路径
    ply_path = args.data_dir / args.patient / f"visit_{args.visit}" / "tumor.ply"

    print("\n" + "=" * 60)
    print("导出 3D 模型快照")
    print("=" * 60)
    print(f"患者:       {args.patient}")
    print(f"就诊:       Visit {args.visit}")
    print(f"PLY 文件:   {ply_path}")
    print(f"输出:       {args.out}")
    print("=" * 60 + "\n")

    try:
        render_snapshot(ply_path, args.out, args.width, args.height)
        print("\n✓ 快照导出完成！")
    except Exception as e:
        logger.error(f"导出失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
